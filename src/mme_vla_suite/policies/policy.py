from collections.abc import Sequence
import time
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0
from mme_vla_suite.shared.mem_buffer import MemoryBuffer, MemoryBufferRecurrent
from mme_vla_suite.shared.sampling_density import compute_frame_indices
from mme_vla_suite.shared.sampling_rules import get_sampling_sources

# Optional: RL selector for frame selection
try:
    from mme_vla_suite.selector.model import FrameSelector
    from mme_vla_suite.selector.config import SelectorConfig
    _SELECTOR_AVAILABLE = True
except ImportError:
    _SELECTOR_AVAILABLE = False

# Optional: QKFS selector for frame selection
try:
    from mme_vla_suite.qkfs.model import QKFS
    from mme_vla_suite.qkfs.config import QKFSConfig
    from mme_vla_suite.qkfs.inference import load_qkfs as _load_qkfs, select_frames_qkfs
    _QKFS_AVAILABLE = True
except ImportError:
    _QKFS_AVAILABLE = False

# Optional: SigLIP text encoder for QKFS instruction embeddings
_SIGLIP_TEXT_MODEL = None
_SIGLIP_TEXT_PROCESSOR = None

def _encode_instruction_siglip(text: str) -> np.ndarray:
    """Encode instruction text using SigLIP text encoder. Cached per session."""
    global _SIGLIP_TEXT_MODEL, _SIGLIP_TEXT_PROCESSOR
    if _SIGLIP_TEXT_MODEL is None:
        import logging
        logging.getLogger(__name__).info("Loading SigLIP text encoder...")
        from transformers import AutoTokenizer, AutoModel
        _SIGLIP_TEXT_PROCESSOR = AutoTokenizer.from_pretrained(
            "google/siglip-so400m-patch14-384")
        _SIGLIP_TEXT_MODEL = AutoModel.from_pretrained(
            "google/siglip-so400m-patch14-384").text_model.eval()
        import torch
        _SIGLIP_TEXT_MODEL = _SIGLIP_TEXT_MODEL.to(torch.float32)
    import torch
    inputs = _SIGLIP_TEXT_PROCESSOR(text, return_tensors="pt",
                                     padding=True, truncation=True)
    with torch.no_grad():
        out = _SIGLIP_TEXT_MODEL(**inputs)
        emb = out.pooler_output[0].cpu().numpy().astype(np.float32)
    return emb

class MME_VLA_Policy:
    def __init__(
        self,
        model: HistoryPi0,
        *,
        seed: int = 42,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        norm_stats: dict[str, _transforms.NormStats] | None = None,
        use_quantiles: bool = False,
    ):
        self._model = model
        self._seed = seed
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._vision_encode = nnx_utils.module_jit(model.vision_encode)
        
        
        self.config = model.history_config
        self.mem_buffer = None

        self.state_norm_stats = norm_stats['state']
        self.use_quantiles = use_quantiles

        # RL selector (loaded separately if available)
        self._frame_selector = None
        self._selector_config = None

        # QKFS selector (loaded separately if available)
        self._qkfs_model = None
        self._qkfs_config = None

        self.reset()
        
    
    def _prepare_mem_buffer(self):
        if self.config is None or self.config.representation_type == "symbolic":
            self.mem_buffer = None
        elif self.config.representation_type == "recurrent":
            self.mem_buffer = MemoryBufferRecurrent(
                num_views=self.config.num_views,
                img_emb_dim=self.config.memory_feature.img.input_dim,
                pos_emb_dim=self.config.memory_feature.pos.input_dim,
                state_emb_dim=self.config.memory_feature.state.input_dim,
                input_obs_horizon=self.config.streaming_obs_horizon,
                max_recur_steps=self.config.recurrent_memory.max_recur_steps,
                max_video_steps=self.config.recurrent_memory.max_pretraj_steps,
                prepare_buffer=True, vision_enc_fn=self._vision_encode,
            )
        else:
            self.mem_buffer = MemoryBuffer(
                num_views=self.config.num_views,
                img_emb_dim=self.config.memory_feature.img.input_dim,
                pos_emb_dim=self.config.memory_feature.pos.input_dim,
                state_emb_dim=self.config.memory_feature.state.input_dim,
                compute_token_drop_score = self.config.perceptual_memory.type == "token_dropping",
                token_drop_stride=self.config.streaming_obs_horizon // 2,
                prepare_buffer=True, vision_enc_fn=self._vision_encode,
            )

    @override
    def infer(self, obs: dict) -> dict:
        if self.config is not None and self.config.representation_type != "symbolic":
            assert len(self.mem_buffer._history_feats) > 0, \
                "history feats is empty, add buffer first"
                                        
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._prepare_history(inputs)
        inputs = self._input_transform(inputs)
        observation = HistAugObservation.from_dict(
            jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        )
        self._rng, sample_rng = jax.random.split(self._rng)
    
        start_time = time.monotonic()
        outputs = {
            "state": observation.state,
            "actions": self._sample_actions(sample_rng, observation, **self._sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)      
        outputs = self._output_transform(outputs)
        outputs["infer_time_ms"] = model_time * 1000
        
        return outputs
    
    @override
    def reset(self) -> None:
        del self.mem_buffer
        self._prepare_mem_buffer()
        self.step_idx = -1
        self.exec_start_idx = 0
        self.keyframe_idxs = []
        self._rng = jax.random.key(self._seed)
        # Density sampling state
        self._density_task_name = None
        self._density_task_goal = None
        self._density_segments = []
        self._density_current_label = None  # (phase, label) tuple
        # QKFS cached instruction embedding
        self._qkfs_instruction_emb = None
        self._qkfs_instruction_text = None
            
    
    def add_buffer(self, obs: dict) -> None:
        if self.mem_buffer is None:
            return
        images = obs["images"]
        states = obs["state"]
        if obs.get("exec_start_idx", 0) > 0: # has video
            self.exec_start_idx = obs["exec_start_idx"]

        step_idx_list = list(range(self.step_idx+1, self.step_idx + len(images) + 1))
        self.mem_buffer.add_buffer(images, states, step_idx_list)

        # Track keyframe indices for oracle keyframe sampling
        if "is_subgoal_boundary" in obs:
            boundaries = obs["is_subgoal_boundary"]
            for i, step_idx in enumerate(step_idx_list):
                if boundaries[i]:
                    self.keyframe_idxs.append(step_idx)

        # Track task name for density sampling
        if "task_name" in obs and obs["task_name"] is not None:
            self._density_task_name = obs["task_name"]

        # Build segments incrementally for density sampling
        if "subgoal_labels" in obs and obs["subgoal_labels"] is not None:
            labels = obs["subgoal_labels"]
            for i, step_idx in enumerate(step_idx_list):
                label = labels[i].strip().lower() if labels[i] else ""
                if not label:
                    continue
                is_demo = step_idx < self.exec_start_idx
                phase = "demo" if is_demo else "exec"
                key = (phase, label)

                if key != self._density_current_label:
                    # Close previous segment
                    if self._density_current_label is not None and self._density_segments:
                        self._density_segments[-1]["end_frame"] = step_idx - 1
                        self._density_segments[-1]["num_frames"] = (
                            step_idx - self._density_segments[-1]["start_frame"]
                        )
                    # Start new segment
                    self._density_segments.append({
                        "idx": len(self._density_segments),
                        "phase": phase,
                        "label": label,
                        "start_frame": step_idx,
                        "end_frame": step_idx,  # updated as we go
                        "num_frames": 1,
                    })
                    self._density_current_label = key
                else:
                    # Extend current segment
                    if self._density_segments:
                        self._density_segments[-1]["end_frame"] = step_idx
                        self._density_segments[-1]["num_frames"] = (
                            step_idx - self._density_segments[-1]["start_frame"] + 1
                        )

        self.step_idx += len(images)

    def _normalize_state(self, state):
        if self.use_quantiles:
            return (state - self.state_norm_stats.q01) / (self.state_norm_stats.q99 - self.state_norm_stats.q01 + 1e-6) * 2.0 - 1.0
        else:
            return (state - self.state_norm_stats.mean) / (self.state_norm_stats.std + 1e-6)

    def _prepare_history(self, inputs: dict) -> dict:
        if self.config is None or self.config.representation_type == "symbolic":
            return inputs
        
        if self.config.representation_type == "recurrent":
            history_feats_gather_fn = self.mem_buffer.default_history_feats_gather_fn
            recur_image_emb, recur_pos_emb, recur_state_emb, recur_mask = \
                self.mem_buffer.prepare_token_recurrent(
                    self.step_idx, self.exec_start_idx, history_feats_gather_fn)
            inputs["recur_image_emb"] = recur_image_emb
            inputs["recur_pos_emb"] = recur_pos_emb
            inputs["recur_state_emb"] = self._normalize_state(recur_state_emb)
            inputs["recur_mask"] = recur_mask
        elif self.config.representation_type == "perceptual":
            # Use QKFS selector if loaded
            if self._qkfs_model is not None:
                return self._prepare_history_with_qkfs(inputs)

            # Use RL selector if loaded, otherwise fall back to configured method
            if self._frame_selector is not None:
                return self._prepare_history_with_selector(inputs)

            history_feats_gather_fn = self.mem_buffer.default_history_feats_gather_fn
            token_budget = self.config.budget

            if self.config.perceptual_memory.type == "token_dropping":
                static_image_emb, static_pos_emb, static_state_emb, static_mask = \
                    self.mem_buffer.prepare_token_dropping(
                        self.step_idx, token_budget, history_feats_gather_fn)
            elif self.config.perceptual_memory.type == "oracle_keyframe_sampling":
                token_per_image = self.config.token_per_image
                static_image_emb, static_pos_emb, static_state_emb, static_mask = \
                    self.mem_buffer.prepare_oracle_keyframe_sampling(
                        self.step_idx, token_budget, token_per_image, self.keyframe_idxs,
                        history_feats_gather_fn)
            elif self.config.perceptual_memory.type == "density_sampling":
                token_per_image = self.config.token_per_image
                static_image_emb, static_pos_emb, static_state_emb, static_mask = \
                    self._prepare_density_history(
                        token_budget, token_per_image, history_feats_gather_fn,
                        task_goal=inputs.get("prompt", ""))
            else:
                token_per_image = self.config.token_per_image
                static_image_emb, static_pos_emb, static_state_emb, static_mask = \
                    self.mem_buffer.prepare_frame_sampling(
                        self.step_idx, token_budget, token_per_image, history_feats_gather_fn)

            inputs["static_image_emb"] = static_image_emb
            inputs["static_pos_emb"] = static_pos_emb
            inputs["static_state_emb"] = self._normalize_state(static_state_emb)
            inputs["static_mask"] = static_mask
        else:
            raise ValueError(f"Not supported representation type: {self.config.representation_type}")
        
    
        return inputs

    def _prepare_density_history(self, token_budget, token_per_image, history_feats_gather_fn,
                                   task_goal=""):
        """Compute density-based frame indices and load features."""
        max_frames = token_budget // (token_per_image * self.config.num_views)
        task_name = self._density_task_name
        segments = self._density_segments

        if not task_name or not segments:
            # Fallback to uniform if no segment info available
            return self.mem_buffer.prepare_frame_sampling(
                self.step_idx, token_budget, token_per_image, history_feats_gather_fn)

        # Get source segments for current step
        source_map = get_sampling_sources(task_name, segments, task_goal=task_goal)

        current_seg_idx = None
        for seg in segments:
            if seg["start_frame"] <= self.step_idx <= seg["end_frame"]:
                current_seg_idx = seg["idx"]
                break
        source_seg_indices = source_map.get(current_seg_idx, []) if current_seg_idx is not None else []

        # Empty keyframes — strategies fall back to uniform within segments
        keyframes_by_seg = {}

        indices = compute_frame_indices(
            task_name=task_name,
            step_idx=self.step_idx,
            segments=segments,
            source_seg_indices=source_seg_indices,
            keyframes_by_seg=keyframes_by_seg,
            task_goal=task_goal,
            max_frames=max_frames,
        )

        return self.mem_buffer.prepare_frame_sampling_with_indices(
            indices, token_budget, token_per_image, history_feats_gather_fn)

    def load_selector(self, selector_checkpoint_path: str, selector_config: Any = None):
        """Load a trained RL frame selector for inference.

        Args:
            selector_checkpoint_path: path to directory containing selector_params.pkl
            selector_config: SelectorConfig instance (or loaded from checkpoint)
        """
        if not _SELECTOR_AVAILABLE:
            raise ImportError("mme_vla_suite.selector not available")

        import pickle
        import flax.nnx as nnx

        if selector_config is None:
            with open(f"{selector_checkpoint_path}/config.pkl", "rb") as f:
                selector_config = pickle.load(f)

        self._selector_config = selector_config
        self._frame_selector = FrameSelector(selector_config, rngs=nnx.Rngs(0))

        with open(f"{selector_checkpoint_path}/selector_params.pkl", "rb") as f:
            saved_state = pickle.load(f)
        nnx.update(self._frame_selector, saved_state)

    def _prepare_history_with_selector(self, inputs: dict) -> dict:
        """Use the RL selector to choose frames instead of uniform sampling."""
        import jax

        history_feats_gather_fn = self.mem_buffer.default_history_feats_gather_fn
        token_budget = self.config.budget
        token_per_image = self.config.token_per_image

        # Get candidate global embeddings (use 8x8=64 tokens for best fidelity)
        cand_global = self.mem_buffer.get_candidate_global_embeddings(
            self.step_idx, pool_size=64
        )
        num_cands = cand_global.shape[0]
        N = self._selector_config.max_candidates

        # Pad candidates
        cand_embs = np.zeros((N, self._selector_config.candidate_emb_dim), dtype=np.float32)
        cand_times = np.zeros((N, 1), dtype=np.float32)
        cand_mask = np.zeros(N, dtype=np.bool_)
        n = min(num_cands, N)
        cand_embs[:n] = cand_global[:n]
        cand_mask[:n] = True
        for t in range(n):
            cand_times[t, 0] = (self.step_idx - t) / max(self.step_idx, 1)

        # Build query
        front_view_emb = cand_global[min(self.step_idx, num_cands - 1)]
        # At inference time the key is "observation/state", at training time it's "state"
        raw_state = inputs.get("state", inputs.get("observation/state"))
        proprio = self._normalize_state(raw_state)

        # Instruction embedding (use prompt tokens from the VLA)
        instr_emb = front_view_emb  # fallback; ideally use LLM embed

        progress = np.array([self.step_idx / max(self.step_idx + 100, 1)], dtype=np.float32)

        # Run selector (deterministic top-k at eval)
        selected = self._frame_selector.select_eval(
            jnp.asarray(front_view_emb),
            jnp.asarray(instr_emb),
            jnp.asarray(proprio),
            jnp.asarray(progress),
            jnp.asarray(cand_embs),
            jnp.asarray(cand_times),
            jnp.asarray(cand_mask),
        )
        selected_indices = sorted(int(i) for i in jax.device_get(selected) if i < num_cands)

        # Use the standard packing with selected indices
        static_image_emb, static_pos_emb, static_state_emb, static_mask = \
            self.mem_buffer.prepare_frame_sampling_with_indices(
                selected_indices, token_budget, token_per_image, history_feats_gather_fn)

        inputs["static_image_emb"] = static_image_emb
        inputs["static_pos_emb"] = static_pos_emb
        inputs["static_state_emb"] = self._normalize_state(static_state_emb)
        inputs["static_mask"] = static_mask
        return inputs

    def load_qkfs_selector(self, qkfs_checkpoint_path: str):
        """Load a trained QKFS selector for inference.

        Args:
            qkfs_checkpoint_path: path to directory containing qkfs_params.pkl and config.pkl
        """
        if not _QKFS_AVAILABLE:
            raise ImportError("mme_vla_suite.qkfs not available")

        self._qkfs_model, self._qkfs_config = _load_qkfs(qkfs_checkpoint_path)

    def _prepare_history_with_qkfs(self, inputs: dict) -> dict:
        """Use QKFS to select frames for perceptual memory."""
        history_feats_gather_fn = self.mem_buffer.default_history_feats_gather_fn
        token_budget = self.config.budget
        token_per_image = self.config.token_per_image

        # Get all past global embeddings and proprios from memory buffer
        all_past_embs = self.mem_buffer.get_candidate_global_embeddings(
            self.step_idx, pool_size=64
        )
        num_past = all_past_embs.shape[0]

        if num_past == 0:
            return self.mem_buffer.prepare_frame_sampling(
                self.step_idx, token_budget, token_per_image, history_feats_gather_fn)

        # Collect past proprios from memory buffer
        all_past_proprios = np.zeros((num_past, self._qkfs_config.proprio_dim), dtype=np.float32)
        for t in range(num_past):
            if t in self.mem_buffer._history_feats:
                feat = self.mem_buffer._history_feats[t]
                if "state_emb" in feat:
                    all_past_proprios[t] = feat["state_emb"]

        # Current observation embedding
        current_obs_emb = all_past_embs[min(self.step_idx, num_past - 1)]

        # Instruction embedding — SigLIP text encoding of the episode prompt
        prompt = inputs.get("prompt", "")
        if prompt and prompt != self._qkfs_instruction_text:
            self._qkfs_instruction_emb = _encode_instruction_siglip(prompt)
            self._qkfs_instruction_text = prompt
        if self._qkfs_instruction_emb is not None:
            instruction_emb = self._qkfs_instruction_emb
        else:
            # Fallback: zero vector (should not happen with valid prompts)
            instruction_emb = np.zeros(self._qkfs_config.instruction_emb_dim, dtype=np.float32)

        # Current proprio
        raw_state = inputs.get("state", inputs.get("observation/state"))
        current_proprio = np.asarray(raw_state, dtype=np.float32)

        # Run QKFS selection
        selected_indices = select_frames_qkfs(
            model=self._qkfs_model,
            config=self._qkfs_config,
            instruction_emb=instruction_emb,
            current_obs_emb=current_obs_emb,
            all_past_embs=all_past_embs,
            all_past_proprios=all_past_proprios,
            current_proprio=current_proprio,
        )

        selected_list = sorted(int(i) for i in selected_indices if i < num_past)

        # Pack selected frames into VLA memory format
        static_image_emb, static_pos_emb, static_state_emb, static_mask = \
            self.mem_buffer.prepare_frame_sampling_with_indices(
                selected_list, token_budget, token_per_image, history_feats_gather_fn)

        inputs["static_image_emb"] = static_image_emb
        inputs["static_pos_emb"] = static_pos_emb
        inputs["static_state_emb"] = self._normalize_state(static_state_emb)
        inputs["static_mask"] = static_mask
        return inputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata