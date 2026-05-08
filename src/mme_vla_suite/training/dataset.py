import os
import json
import logging
import numpy as np
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import random
import re

from openpi.training import config as _config
from openpi.training.data_loader import Dataset
from mme_vla_suite.shared.mem_buffer import MemoryBuffer, MemoryBufferRecurrent
from mme_vla_suite.shared.sampling_density import compute_frame_indices, DEFAULT_MAX_FRAMES
from mme_vla_suite.shared.sampling_rules import get_sampling_sources
import pickle

# Global episode index -> task name mapping (alphabetical, 100 episodes each)
_TASK_ORDER = [
    "BinFill", "ButtonUnmask", "ButtonUnmaskSwap", "InsertPeg",
    "MoveCube", "PatternLock", "PickHighlight", "PickXtimes",
    "RouteStick", "StopCube", "SwingXtimes", "VideoPlaceButton",
    "VideoPlaceOrder", "VideoRepick", "VideoUnmask", "VideoUnmaskSwap",
]

def _episode_to_task(global_ep_idx: int) -> str:
    """Map global episode index to task name."""
    task_idx = global_ep_idx // 100
    return _TASK_ORDER[task_idx]

random.seed(0)
logger = logging.getLogger(__name__)


def load_vector_file(vector_path: str, step_idx: int) -> tuple[dict, int]:
    with open(vector_path, "rb") as f:
        return np.load(f, allow_pickle=True).item(), step_idx
    
    

class SampleDataset(Dataset):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.stats = json.load(open(os.path.join(self.dataset_path, "meta", "stats.json")))
    
    def __len__(self):
        if "execution_samples" in self.stats:
            return self.stats["execution_samples"]
        else:
            return self.stats["total_samples"]
    
    def __getitem__(self, idx):
        with open(os.path.join(self.dataset_path,  "data", f"{idx}.pkl"), "rb") as f:
            return pickle.load(f)


class RoboMMEDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        data_config: _config.DataConfig,
        history_config: DictConfig | None,
        action_horizon: int,
        compute_norm_stats: bool = False,
    ):
        self.history_config = history_config
        
        
        self.action_horizon = action_horizon
        self.dataset = SampleDataset(dataset_path)
        self.feature_dir = Path(self.dataset.dataset_path) / "features"

        if self.history_config is not None:
            self.img_emb_dim = self.history_config.memory_feature.img.input_dim
            self.pos_emb_dim = self.history_config.memory_feature.pos.input_dim
            self.state_emb_dim = self.history_config.memory_feature.state.input_dim
            self.num_views = self.history_config.num_views
            self.streaming_obs_horizon = self.history_config.streaming_obs_horizon
        
            if self.history_config.representation_type == "perceptual":
                self.mem_buffer = MemoryBuffer(
                    num_views=self.num_views,
                    img_emb_dim=self.img_emb_dim,
                    pos_emb_dim=self.pos_emb_dim,
                    state_emb_dim=self.state_emb_dim,
                    token_drop_stride=self.streaming_obs_horizon // 2,
                )
            elif self.history_config.representation_type == "recurrent":
                self.mem_buffer = MemoryBufferRecurrent(
                    num_views=self.num_views,
                    img_emb_dim=self.img_emb_dim,
                    pos_emb_dim=self.pos_emb_dim,
                    state_emb_dim=self.state_emb_dim,
                    input_obs_horizon=self.streaming_obs_horizon,
                    max_recur_steps=self.history_config.recurrent_memory.max_recur_steps,
                    max_video_steps=self.history_config.recurrent_memory.max_pretraj_steps,
                )
            else:
                # symbolic memory does not need mem_buffer
                self.mem_buffer = None
        else:
            logger.info("=== Do not use history ===")
        
        self.compute_norm_stats = compute_norm_stats
        
        if not compute_norm_stats:
            self.state_norm_stats = data_config.norm_stats['state']
            self.use_quantiles = data_config.use_quantile_norm
        
    def _load_keyframe_idxs(self, epis_idx: int) -> list[int]:
        path = os.path.join(self.feature_dir, f"episode_{epis_idx}", "keyframe_idxs.json")
        with open(path) as f:
            return json.load(f)

    def _gather_history_feat(self, indices_to_load: list[int], epis_idx: int):
        history_feats = {}
        history_paths = []
        for idx in indices_to_load:
            history_feats[idx] = {}
            history_paths.append(
                os.path.join(self.feature_dir, f"episode_{epis_idx}", f"token_emb_{idx}.npy")
            )
        
        if self.history_config.representation_type == "recurrent":
            # load_vector_fn = load_vector_file_lru4096 if self.use_lru else load_vector_file
            load_vector_fn = load_vector_file
            max_workers = min(48, max(4, len(history_paths)))
        else:
            # load_vector_fn = load_vector_file_lru1024 if self.use_lru else load_vector_file
            load_vector_fn = load_vector_file
            max_workers = min(36, max(4, len(history_paths)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(load_vector_fn, path, idx): path
                for path, idx in zip(history_paths, indices_to_load)
            }
            for future in as_completed(future_to_path):
                np_dict, idx = future.result()
                history_feats[idx] = np_dict
        return history_feats
    
    
    def prepare_token_drop(self, epis_idx, step_idx):
        token_budget = self.history_config.budget
        kept_indices = json.load(open(os.path.join(self.feature_dir, f"episode_{epis_idx}", "kept_indices.json")))
        
        return self.mem_buffer.prepare_token_dropping(
            step_idx, token_budget, self._gather_history_feat, 
            kept_indices=kept_indices, epis_idx=epis_idx)
        

    def prepare_frame_sampling(self,  epis_idx,  step_idx):
        token_per_image = self.history_config.token_per_image
        token_budget = self.history_config.budget

        return self.mem_buffer.prepare_frame_sampling(
            step_idx, token_budget, token_per_image, self._gather_history_feat,
            epis_idx=epis_idx)

    def prepare_oracle_keyframe_sampling(self, epis_idx, step_idx):
        token_per_image = self.history_config.token_per_image
        token_budget = self.history_config.budget
        keyframe_idxs = self._load_keyframe_idxs(epis_idx)

        return self.mem_buffer.prepare_oracle_keyframe_sampling(
            step_idx, token_budget, token_per_image, keyframe_idxs,
            self._gather_history_feat, epis_idx=epis_idx)

    def _load_density_metadata(self, epis_idx: int) -> tuple[list[dict], dict[int, list[int]]]:
        """Load precomputed segments and density keyframes for an episode."""
        ep_dir = os.path.join(self.feature_dir, f"episode_{epis_idx}")
        with open(os.path.join(ep_dir, "segments.json")) as f:
            segments = json.load(f)
        with open(os.path.join(ep_dir, "density_keyframes.json")) as f:
            raw = json.load(f)
        keyframes_by_seg = {int(k): v for k, v in raw.items()}
        return segments, keyframes_by_seg

    def prepare_density_sampling(self, epis_idx, step_idx, task_goal):
        token_per_image = self.history_config.token_per_image
        token_budget = self.history_config.budget
        max_frames = token_budget // (token_per_image * self.num_views)

        task_name = _episode_to_task(epis_idx)
        segments, keyframes_by_seg = self._load_density_metadata(epis_idx)
        source_map = get_sampling_sources(task_name, segments, task_goal=task_goal)

        # Find current exec segment to get source_seg_indices
        current_seg_idx = None
        for seg in segments:
            if seg["start_frame"] <= step_idx <= seg["end_frame"]:
                current_seg_idx = seg["idx"]
                break
        source_seg_indices = source_map.get(current_seg_idx, []) if current_seg_idx is not None else []

        indices = compute_frame_indices(
            task_name=task_name,
            step_idx=step_idx,
            segments=segments,
            source_seg_indices=source_seg_indices,
            keyframes_by_seg=keyframes_by_seg,
            task_goal=task_goal,
            max_frames=max_frames,
        )

        return self.mem_buffer.prepare_frame_sampling_with_indices(
            indices, token_budget, token_per_image,
            self._gather_history_feat, epis_idx=epis_idx,
        )

    def prepare_token_recurrent(self, epis_idx, step_idx, exec_start_idx):      
        self.mem_buffer.exec_start_idx = exec_start_idx
        
        return self.mem_buffer.prepare_token_recurrent(
            step_idx, exec_start_idx, self._gather_history_feat,
            epis_idx=epis_idx)
        
    
    def __len__(self):
        return len(self.dataset)
    
    
    def _normalize_state(self, state):
        if self.compute_norm_stats:
            return state
        else:
            if self.use_quantiles:
                return (state - self.state_norm_stats.q01) / (self.state_norm_stats.q99 - self.state_norm_stats.q01 + 1e-6) * 2.0 - 1.0
            else:
                return (state - self.state_norm_stats.mean) / (self.state_norm_stats.std + 1e-6)
    
    def _truncated_gaussian_noise(self, max_val: int, std: float = None) -> int:
        """Generate truncated Gaussian noise centered at 0, clamped to [-max_val, max_val]."""
        if std is None:
            std = max_val / 2.5  # ~95% of samples within range before clamping
        noise = random.gauss(0, std)
        noise = max(-max_val, min(max_val, noise))  # clamp to [-max_val, max_val]
        return int(round(noise))

    def add_grounding_augmentation(self, subgoal: str, noise_range: int = 8) -> str:
        if not subgoal:
            return subgoal
        
        matches = re.findall(r'at <(\d+), (\d+)>', subgoal)
        
        if len(matches) == 0 or len(matches) > 1:
            return subgoal
        
        x, y = matches[0]
        x = int(x)
        y = int(y)
        noise_x = self._truncated_gaussian_noise(noise_range)
        noise_y = self._truncated_gaussian_noise(noise_range)

        new_subgoal = subgoal.replace(f'at <{x}, {y}>', f'at <{x + noise_x}, {y + noise_y}>')
        return new_subgoal
        

    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        data["actions"] = data["actions"][:self.action_horizon]
        
        # During online evaluation, the ground-truth subgoal may change earlier than when it was recorded.
        # To make the model robust to this temporal shift and avoid train/test distribution mismatch,
        # we randomly sample from either subgoal or subgoal_online (early change).
        if self.history_config is not None \
            and self.history_config.representation_type == "symbolic" \
            and self.history_config.symbolic_memory.type in ["simple_subgoal", "grounded_subgoal"] \
            and random.random() < 0.5: 
            data["simple_subgoal"] = data["simple_subgoal_online"]
            data["grounded_subgoal"] = data["grounded_subgoal_online"]
        data.pop("simple_subgoal_online")
        data.pop("grounded_subgoal_online")
        
        if self.history_config is not None and self.history_config.representation_type == "symbolic":
            data["grounded_subgoal"] = self.add_grounding_augmentation(data["grounded_subgoal"], noise_range=8)
            data["simple_subgoal"] = self.add_grounding_augmentation(data["simple_subgoal"], noise_range=8)
 
        
        if self.history_config is not None:
            # use history
            epis_idx = data["epis_idx"].item()
            step_idx = data["step_idx"].item()
            exec_start_idx = data["exec_start_idx"].item()

            if self.history_config.representation_type == "perceptual":
                if self.history_config.perceptual_memory.type == "token_dropping":
                    (
                        static_img_emb,
                        static_pos_emb,
                        static_state_emb,
                        static_mask, # >=64
                    ) = self.prepare_token_drop(epis_idx, step_idx)
                elif self.history_config.perceptual_memory.type == "oracle_keyframe_sampling":
                    (
                        static_img_emb,
                        static_pos_emb,
                        static_state_emb,
                        static_mask,
                    ) = self.prepare_oracle_keyframe_sampling(epis_idx, step_idx)
                elif self.history_config.perceptual_memory.type == "density_sampling":
                    (
                        static_img_emb,
                        static_pos_emb,
                        static_state_emb,
                        static_mask,
                    ) = self.prepare_density_sampling(epis_idx, step_idx, data["prompt"])
                else:
                    (
                        static_img_emb,
                        static_pos_emb,
                        static_state_emb,
                        static_mask, # slow >=64, mid >=16,fast >=4
                    ) = self.prepare_frame_sampling(epis_idx, step_idx)
                
                data["static_image_emb"] = static_img_emb
                data["static_pos_emb"] = static_pos_emb
                data["static_state_emb"] = self._normalize_state(static_state_emb)
                data["static_mask"] = static_mask

            elif self.history_config.representation_type == "recurrent":             
                recur_img_emb, recur_pos_emb, recur_state_emb, recur_mask = (
                    self.prepare_token_recurrent(
                        epis_idx, step_idx, exec_start_idx)
                )

                data["recur_image_emb"] = recur_img_emb
                data["recur_pos_emb"] = recur_pos_emb
                data["recur_state_emb"] = self._normalize_state(recur_state_emb)
                data["recur_mask"] = recur_mask  # 1 ~ max_exec_steps

            elif self.history_config.representation_type == "symbolic":
                pass
            else:
                raise ValueError(
                    f"Not supported representation type: {self.history_config.representation_type}"
                )
        
        
        for key in [
            "static_image_emb",
            "static_pos_emb",
            "static_state_emb",
            "static_mask",
            
            "recur_image_emb",
            "recur_pos_emb",
            "recur_state_emb",
            "recur_mask",
            
            "simple_subgoal",
            "grounded_subgoal",
            "prompt"
        ]:
            if key not in data:
                data[key] = None        

        return data