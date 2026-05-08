"""MLLM (Gemini) scorer for candidate action-attempt windows.

The MLLM is the main success/failure teacher.  For each candidate window,
it receives a before/center/after triple (or short video clip) and
classifies the window with action-specific success criteria.

The MLLM returns structured JSON including:
- is_action_happening
- is_success
- is_failed_attempt
- confidence
- reason (free-text explanation)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import imageio
import numpy as np

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from mme_vla_suite.pseudo_labeling.candidate_detector import CandidateWindow


@dataclass
class MLLMWindowResult:
    """MLLM classification for one candidate window."""

    window: CandidateWindow
    is_success: bool | None
    is_action_happening: bool | None
    is_failed_attempt: bool | None
    confidence: float
    reason: str
    raw_responses: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Action-specific success criteria
# ---------------------------------------------------------------------------

def _get_success_criterion(subtask_label: str) -> str:
    """Return an action-specific success criterion for the MLLM prompt."""
    label = subtask_label.strip().lower()

    # Pick / grasp / lift
    if any(v in label for v in ("pick up", "grasp", "lift", "pickup")):
        return (
            "Success means the target object ends up held by the gripper "
            "and lifted off the surface. Touching, pushing, or briefly "
            "moving the object is NOT success."
        )

    # Place in bin
    if "bin" in label and any(v in label for v in ("put", "place", "drop")):
        return (
            "Success means the object is released inside the bin and "
            "remains there. Dropping outside the bin or passing over the "
            "bin is NOT success."
        )

    # Place on target
    if "target" in label and any(v in label for v in ("put", "place", "move")):
        return (
            "Success means the object is placed on the correct target "
            "location and released. Placing on the wrong target or "
            "dropping nearby is NOT success."
        )

    # Button press
    if "button" in label or "press" in label:
        return (
            "Success means the button is visibly pressed/depressed by "
            "the gripper. Merely touching near the button or hovering "
            "above it is NOT success."
        )

    # Peg insertion
    if "insert" in label and "peg" in label:
        return (
            "Success means the peg is aligned and inserted into the "
            "correct box side/opening. Touching the box or missing the "
            "opening is NOT success."
        )

    # Pattern / trace
    if any(v in label for v in ("pattern", "retrace", "trace")):
        return (
            "Success means the stick follows the demonstrated pattern "
            "and reaches the correct endpoint. Partial movement or "
            "deviation from the path is NOT success."
        )

    # Route / navigate
    if any(v in label for v in ("route", "navigate")):
        return (
            "Success means the stick follows the demonstrated route "
            "around obstacles and reaches the end. Colliding with "
            "obstacles or taking the wrong path is NOT success."
        )

    # Swing / move between targets
    if any(v in label for v in ("swing", "right-to-left", "left-to-right")):
        return (
            "Success means the cube reaches the specified side target. "
            "Moving toward the wrong target or stopping midway is NOT "
            "success."
        )

    # Stop cube
    if "stop" in label and "cube" in label:
        return (
            "Success means the cube is stopped at the correct target. "
            "Pressing the button when the cube is not at the target is "
            "NOT success."
        )

    # Container / unmask
    if "container" in label or "unmask" in label:
        return (
            "Success means the correct container is picked up, revealing "
            "the hidden cube. Picking the wrong container is NOT success."
        )

    # Generic fallback
    return (
        f"Success means the subtask \"{subtask_label}\" has been fully "
        f"completed. The intended outcome must be visually achieved, not "
        f"just attempted."
    )


def _build_window_query(subtask_label: str) -> str:
    """Build the MLLM query for classifying a candidate window."""
    criterion = _get_success_criterion(subtask_label)

    return f"""You are analyzing a robot manipulation video showing a temporal window (before, during, and after frames) for the subtask: "{subtask_label}".

SUCCESS CRITERION: {criterion}

Answer each question with "yes", "no", or "unsure":
1. Is the robot actively attempting to {subtask_label} in the middle frames?
2. Is the action "{subtask_label}" successfully completed by the end of this window?
3. If the robot is attempting the action, did the attempt fail?

Provide your classification in this exact JSON format:
{{"is_action_happening": true/false, "is_success": true/false, "is_failed_attempt": true/false, "confidence": 0.0-1.0, "reason": "brief explanation of what you see"}}

Rules:
- is_success=true means the success criterion above is MET by the end frame.
- is_failed_attempt=true means the robot tried but the criterion is NOT met.
- is_success and is_failed_attempt cannot both be true.
- confidence should reflect how clearly you can see the outcome."""


def _parse_mllm_response(response_text: str) -> dict:
    """Parse the MLLM JSON response."""
    json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            return {
                "is_action_happening": bool(result.get("is_action_happening", False)),
                "is_success": bool(result.get("is_success", False)),
                "is_failed_attempt": bool(result.get("is_failed_attempt", False)),
                "confidence": float(result.get("confidence", 0.5)),
                "reason": str(result.get("reason", "")),
            }
        except (json.JSONDecodeError, ValueError):
            pass

    return {
        "is_action_happening": None,
        "is_success": None,
        "is_failed_attempt": None,
        "confidence": 0.2,
        "reason": "failed to parse response",
    }


class MLLMScorer:
    """Classifies candidate windows using Gemini MLLM."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
        tmp_dir: str = "/tmp/mllm_scorer",
        rate_limit_delay: float = 0.5,
    ) -> None:
        if genai is None:
            raise ImportError("google-generativeai is required.")

        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=(
                "You are an expert robot manipulation analyst. "
                "You analyze short video clips or image sequences of robot "
                "actions and classify whether the robot successfully "
                "completed a subtask or failed. Be precise and focus on "
                "the physical outcome of the action, not the intent."
            ),
        )
        self.tmp_dir = tmp_dir
        os.makedirs(tmp_dir, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay

    def score_windows(
        self,
        candidates: list[CandidateWindow],
        all_frames: list[np.ndarray],
        frame_indices: list[int],
        subtask_label: str,
        use_triples: bool = True,
    ) -> list[MLLMWindowResult]:
        """Score each candidate window using the MLLM.

        Args:
            candidates: Candidate windows to classify.
            all_frames: All RGB frames in the segment.
            frame_indices: Global timestep indices.
            subtask_label: The atomic subtask label.
            use_triples: Use image triples (faster) vs video clips.

        Returns:
            List of MLLMWindowResult, one per candidate.
        """
        results: list[MLLMWindowResult] = []

        for cand in candidates:
            if use_triples:
                result = self._score_triple(
                    cand, all_frames, subtask_label
                )
            else:
                result = self._score_video(
                    cand, all_frames, subtask_label
                )
            results.append(result)
            time.sleep(self.rate_limit_delay)

        return results

    def _score_triple(
        self,
        cand: CandidateWindow,
        all_frames: list[np.ndarray],
        subtask_label: str,
    ) -> MLLMWindowResult:
        """Score using before/center/after image triple."""
        n = len(all_frames)
        center = cand.center_frame
        before = max(0, center - 3)
        after = min(n - 1, center + 3)
        # Also include after+N for postcondition check
        after_ext = min(n - 1, center + 6)

        indices = [before, center, after]
        if after_ext != after:
            indices.append(after_ext)

        frame_labels = ["BEFORE", "DURING (contact peak)", "AFTER"]
        if len(indices) == 4:
            frame_labels.append("AFTER+3 (postcondition check)")

        selected_frames = [all_frames[i] for i in indices]

        # Save as images
        paths = []
        for i, (frame, fl) in enumerate(zip(selected_frames, frame_labels)):
            p = os.path.join(
                self.tmp_dir,
                f"triple_{cand.center_global}_{i}.png",
            )
            imageio.imwrite(p, frame)
            paths.append(p)

        try:
            uploaded = []
            for p in paths:
                f = genai.upload_file(path=p)
                uploaded.append(f)

            # Wait for processing
            for _ in range(20):
                all_ready = True
                for j in range(len(uploaded)):
                    uploaded[j] = genai.get_file(uploaded[j].name)
                    if uploaded[j].state.name == "PROCESSING":
                        all_ready = False
                if all_ready:
                    break
                time.sleep(0.3)

            label_desc = ", ".join(frame_labels)
            preamble = (
                f"I'm showing you {len(indices)} frames from a robot "
                f"manipulation sequence: {label_desc}.\n\n"
            )
            query = preamble + _build_window_query(subtask_label)

            response = self.model.generate_content([query, *uploaded])
            response_text = response.text
            parsed = _parse_mllm_response(response_text)

            for f in uploaded:
                try:
                    genai.delete_file(f.name)
                except Exception:
                    pass

            return MLLMWindowResult(
                window=cand,
                is_success=parsed.get("is_success"),
                is_action_happening=parsed.get("is_action_happening"),
                is_failed_attempt=parsed.get("is_failed_attempt"),
                confidence=parsed.get("confidence", 0.5),
                reason=parsed.get("reason", ""),
                raw_responses={
                    "query": query,
                    "response": response_text,
                    "parsed": parsed,
                },
            )

        except Exception as e:
            return MLLMWindowResult(
                window=cand,
                is_success=None,
                is_action_happening=None,
                is_failed_attempt=None,
                confidence=0.0,
                reason=f"error: {e}",
                raw_responses={"error": str(e)},
            )

    def _score_video(
        self,
        cand: CandidateWindow,
        all_frames: list[np.ndarray],
        subtask_label: str,
    ) -> MLLMWindowResult:
        """Score using a short video clip."""
        start = max(0, cand.center_frame - 4)
        end = min(len(all_frames) - 1, cand.center_frame + 4)
        window_frames = all_frames[start: end + 1]

        if len(window_frames) < 2:
            return MLLMWindowResult(
                window=cand, is_success=None, is_action_happening=None,
                is_failed_attempt=None, confidence=0.0,
                reason="window too small",
                raw_responses={"error": "window too small"},
            )

        clip_path = os.path.join(
            self.tmp_dir,
            f"window_{cand.center_global}.mp4",
        )
        imageio.mimsave(clip_path, window_frames, fps=5)

        try:
            video_file = genai.upload_file(path=clip_path)
            for _ in range(30):
                video_file = genai.get_file(video_file.name)
                if video_file.state.name != "PROCESSING":
                    break
                time.sleep(0.5)

            if video_file.state.name == "FAILED":
                return MLLMWindowResult(
                    window=cand, is_success=None, is_action_happening=None,
                    is_failed_attempt=None, confidence=0.0,
                    reason="video processing failed",
                    raw_responses={"error": "video processing failed"},
                )

            query = _build_window_query(subtask_label)
            response = self.model.generate_content([query, video_file])
            response_text = response.text
            parsed = _parse_mllm_response(response_text)

            try:
                genai.delete_file(video_file.name)
            except Exception:
                pass

            return MLLMWindowResult(
                window=cand,
                is_success=parsed.get("is_success"),
                is_action_happening=parsed.get("is_action_happening"),
                is_failed_attempt=parsed.get("is_failed_attempt"),
                confidence=parsed.get("confidence", 0.5),
                reason=parsed.get("reason", ""),
                raw_responses={
                    "query": query,
                    "response": response_text,
                    "parsed": parsed,
                },
            )

        except Exception as e:
            return MLLMWindowResult(
                window=cand, is_success=None, is_action_happening=None,
                is_failed_attempt=None, confidence=0.0,
                reason=f"error: {e}",
                raw_responses={"error": str(e)},
            )
