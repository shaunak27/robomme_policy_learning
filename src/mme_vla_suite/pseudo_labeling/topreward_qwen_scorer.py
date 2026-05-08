"""TOPReward-style scorer using Qwen3-VL-8B.

Computes log P("True" | video_prefix, prompt) for task completion assessment.
Follows the reference TOPReward implementation: includes "True" in the input
text, masks prompt tokens with -100, and extracts log-probs via gather on the
shifted logits.

Does NOT generate text or parse output.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_SIZE = 244  # TOPReward resizes all frames to 244x244
DEFAULT_FPS = 2.0  # TOPReward default fps for video content
SOURCE_FPS = 20.0  # RoboMME control frequency (frames stored at 20Hz)

# Prompt text that goes BEFORE the instruction (inside the video message).
# The instruction + "True" are appended separately after stripping EOS.
PROMPT_PREFIX = (
    "The above video shows a robot manipulation trajectory that completes "
    "the following task: "
)

PROMPT_PREFIX_WINDOW = (
    "The above video shows a short robot manipulation trajectory that completes "
    "the following task: "
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model_and_processor(
    model_name: str,
    device: str = "cuda",
    dtype: str = "auto",
) -> tuple[Any, Any, Any]:
    """Load Qwen3-VL model, processor, and tokenizer."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if dtype == "auto":
        torch_dtype = "auto"
    else:
        torch_dtype = getattr(torch, dtype, torch.bfloat16)

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
    except (ImportError, ValueError) as e:
        logger.warning("flash_attention_2 unavailable (%s), using default attention", e)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
        )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = processor.tokenizer

    return model, processor, tokenizer


# ---------------------------------------------------------------------------
# Frame preprocessing (matches TOPReward's to_pil)
# ---------------------------------------------------------------------------


def _prepare_frames(
    frames: list[np.ndarray],
    img_size: int = IMG_SIZE,
) -> list[Image.Image]:
    """Convert numpy frames to resized PIL images, matching TOPReward."""
    pil_frames = []
    for f in frames:
        if isinstance(f, Image.Image):
            pil = f
        else:
            # Normalize float images to uint8 if needed
            if f.dtype in (np.float32, np.float64) and f.max() <= 1.0:
                f = (f * 255).astype(np.uint8)
            pil = Image.fromarray(f, "RGB")
        pil_frames.append(pil.resize((img_size, img_size)))
    return pil_frames


# ---------------------------------------------------------------------------
# Core scoring — matches TOPReward QwenClient.compute_instruction_reward
# ---------------------------------------------------------------------------


def score_true_logprob(
    model: Any,
    processor: Any,
    frames: list[np.ndarray],
    subtask_label: str,
    prompt_prefix: str = PROMPT_PREFIX,
    reduction: str = "mean",
    fps: float = DEFAULT_FPS,
) -> dict:
    """Compute TOPReward for one video or prefix.

    Follows the reference implementation exactly:
    1. Build video message with prompt_prefix as the text content.
    2. Apply chat template with add_generation_prompt=False.
    3. Strip the EOS token from the templated text.
    4. Append the instruction suffix including " True" as raw text.
    5. Tokenize the full text (with "True" included).
    6. Mask all tokens except the last one (the "True" token) with -100.
    7. Forward pass, extract log-prob of "True" via gather.

    Returns
    -------
    dict with keys:
        logp_true : float   (the reward value)
        reward : float       (same as logp_true)
        token_count : int    (number of unmasked tokens scored)
        reduction : str
    """
    device = next(model.parameters()).device

    # Resize frames to 244x244 (matching TOPReward)
    pil_frames = _prepare_frames(frames)

    # Build the instruction suffix: "{subtask_label}. Decide ... True"
    instruction_suffix = (
        f"{subtask_label}. Decide whether the above statement is True or "
        f"not. The answer is: True"
    )

    # Build the video message (prompt_prefix only — instruction appended later)
    content = [
        {"type": "video", "video": pil_frames, "fps": fps},
        {"type": "text", "text": prompt_prefix},
    ]
    user_messages = [{"role": "user", "content": content}]

    # Apply chat template WITHOUT generation prompt
    prompt_chat = processor.apply_chat_template(
        user_messages, tokenize=False, add_generation_prompt=False,
    )

    # Strip EOS token so "True" doesn't land after a turn boundary
    eos_token = processor.tokenizer.eos_token
    if eos_token is not None:
        prompt_chat = prompt_chat.split(eos_token)[0]

    # Append instruction suffix (includes " True" at the end)
    full_text = f"{prompt_chat}{instruction_suffix}"

    # Process vision info (extract pixel values from the messages)
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(user_messages)

    # Tokenize the full text + vision
    inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Build labels: mask everything except the last token ("True")
    labels = inputs["input_ids"].clone()
    prompt_length = inputs["input_ids"].shape[1] - 1
    labels[:, :prompt_length] = -100
    if "attention_mask" in inputs:
        labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)

    # Extract per-token log-probs (shifted: logits predict next token)
    logits = outputs.logits[:, :-1, :]      # (1, seq_len-1, vocab)
    target_labels = labels[:, 1:]            # (1, seq_len-1)
    log_probs = F.log_softmax(logits, dim=-1)

    mask = target_labels != -100
    safe_targets = target_labels.masked_fill(~mask, 0)
    token_log_probs = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    masked_log_probs = token_log_probs[mask]

    # Reduce
    if reduction == "sum":
        reward = masked_log_probs.sum().item()
    else:
        reward = masked_log_probs.mean().item()

    token_count = int(mask.sum().item())

    return {
        "logp_true": reward,
        "reward": reward,
        "token_count": token_count,
        "reduction": reduction,
    }


# ---------------------------------------------------------------------------
# Uniform frame sampling
# ---------------------------------------------------------------------------


def subsample_to_fps(
    frames: list[np.ndarray],
    source_fps: float = SOURCE_FPS,
    target_fps: float = 4.0,
    min_frames: int = 4,
) -> list[np.ndarray]:
    """Subsample frames from source_fps to target_fps.

    E.g. 172 frames at 20Hz → 34 frames at 4fps.
    Short prefixes are floored at min_frames.
    """
    n = len(frames)
    budget = max(min_frames, int(round(n * target_fps / source_fps)))
    budget = min(budget, n)
    if budget >= n:
        return list(frames)
    indices = np.linspace(0, n - 1, budget).astype(int)
    return [frames[i] for i in indices]


# ---------------------------------------------------------------------------
# Prefix scoring
# ---------------------------------------------------------------------------


def score_prefixes(
    model: Any,
    processor: Any,
    frames: list[np.ndarray],
    subtask_label: str,
    num_prefixes: int = 16,
    target_fps: float = 4.0,
    source_fps: float = SOURCE_FPS,
    reduction: str = "mean",
    fps: float = DEFAULT_FPS,
) -> dict:
    """Score K video prefixes using TOPReward.

    Parameters
    ----------
    model, processor : loaded Qwen VL model + processor
    frames : full segment frames (list of H,W,3 uint8)
    subtask_label : the subtask text label
    num_prefixes : number of prefix endpoints to evaluate
    target_fps : subsample each prefix to this fps before feeding to Qwen.
        Source data is at source_fps (20Hz for RoboMME).
    source_fps : fps of the source frames (default 20Hz).
    reduction : "mean" (default, matches TOPReward) or "sum"
    fps : fps metadata passed to Qwen processor (default 2.0)

    Returns
    -------
    dict with keys:
        prefix_end_indices : list[int]
        logp_true : list[float]
        raw_reward : list[float]
        reward_norm : list[float]
        frames_sent_per_prefix : list[int]
    """
    T = len(frames)
    min_prefix_frames = max(4, int(0.1 * T))
    prefix_end_indices = np.linspace(min_prefix_frames, T - 1, num_prefixes).astype(int)
    # Deduplicate (matching TOPReward's sorted(set(...)))
    prefix_end_indices = sorted(set(int(x) for x in prefix_end_indices))

    logp_true_list: list[float] = []
    frames_sent_list: list[int] = []

    for i, end_idx in enumerate(prefix_end_indices):
        prefix_frames = frames[: end_idx + 1]

        # Subsample from source_fps to target_fps
        sampled = subsample_to_fps(
            prefix_frames, source_fps=source_fps, target_fps=target_fps,
        )

        result = score_true_logprob(
            model, processor, sampled, subtask_label,
            prompt_prefix=PROMPT_PREFIX,
            reduction=reduction,
            fps=fps,
        )
        logp_true_list.append(result["logp_true"])
        frames_sent_list.append(len(sampled))

        logger.info(
            "Prefix %d/%d (end=%d, n_frames=%d/%d): reward=%.4f (%s, %d tok)",
            i + 1,
            len(prefix_end_indices),
            end_idx,
            len(sampled),
            len(prefix_frames),
            result["reward"],
            reduction,
            result["token_count"],
        )

    raw_reward = np.array(logp_true_list, dtype=np.float64)

    # Normalize (matching TOPReward's normalize_rewards)
    if len(raw_reward) <= 1:
        reward_norm = np.ones_like(raw_reward)
    else:
        rmin, rmax = raw_reward.min(), raw_reward.max()
        if rmax == rmin:
            reward_norm = np.ones_like(raw_reward)
        else:
            reward_norm = (raw_reward - rmin) / (rmax - rmin)

    return {
        "prefix_end_indices": [int(x) for x in prefix_end_indices],
        "logp_true": logp_true_list,
        "raw_reward": raw_reward.tolist(),
        "reward_norm": reward_norm.tolist(),
        "frames_sent_per_prefix": frames_sent_list,
    }


def score_window(
    model: Any,
    processor: Any,
    frames: list[np.ndarray],
    window_start: int,
    window_end: int,
    subtask_label: str,
    target_fps: float = 4.0,
    source_fps: float = SOURCE_FPS,
    reduction: str = "mean",
    fps: float = DEFAULT_FPS,
) -> dict:
    """Score a single attempt window using the window prompt template."""
    window_frames = frames[window_start: window_end + 1]

    sampled = subsample_to_fps(
        window_frames, source_fps=source_fps, target_fps=target_fps,
    )

    return score_true_logprob(
        model, processor, sampled, subtask_label,
        prompt_prefix=PROMPT_PREFIX_WINDOW,
        reduction=reduction,
        fps=fps,
    )
