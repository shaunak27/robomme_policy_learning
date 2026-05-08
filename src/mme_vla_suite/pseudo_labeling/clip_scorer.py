"""CLIP-based weak static predicate scoring.

CLIP scores frames against state-predicate text prompts. It provides
weak evidence for pre/contact/post states but should NOT be trusted to
directly assign phase labels.  In particular:

- CLIP is used for candidate attempt window detection.
- CLIP provides weak pre/post/contact evidence for fusion.
- failed_attempt is NOT scored by CLIP (it requires temporal context).
- Final labels are derived by temporal constraints in label_fusion.py.

Only the front camera image is scored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# The 3 phases CLIP actually scores (failed_attempt is temporal-only)
CLIP_PHASE_NAMES = [
    "precondition",
    "contact_or_transition",
    "postcondition_success",
]

# Full 4-phase vocabulary used by the rest of the pipeline
PHASE_NAMES = [
    "precondition",
    "contact_or_transition",
    "postcondition_success",
    "failed_attempt",
]


def generate_phase_prompts(subtask_label: str) -> dict[str, list[str]]:
    """Generate RoboMME-specific state-predicate prompts per phase.

    Important:
        These prompts are meant to provide weak CLIP-style static evidence.
        They should NOT be treated as reliable success/failure labels by
        themselves. In particular, "failed_attempt" is best derived temporally
        as: contact/transition evidence without stable postcondition.

    RoboMME task families covered:
        - cube/block pick, place, put in bin, put on target
        - repeated pick/place and swing-to-target actions
        - button press / stop button / stop cube at target
        - containers hiding colored cubes
        - highlighted cubes
        - video-reference tasks: same cube, demo target, demo order
        - peg insertion
        - pattern tracing / route following with stick
    """
    label = subtask_label.strip().lower()
    verb, obj = _parse_action_object(label)

    prompts: dict[str, list[str]] = {
        "precondition": [],
        "contact_or_transition": [],
        "postcondition_success": [],
        "failed_attempt": [],
    }

    def add(phase: str, *items: str) -> None:
        prompts[phase].extend([x for x in items if x])

    def dedupe() -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for phase, items in prompts.items():
            seen = set()
            out[phase] = []
            for p in items:
                p = re.sub(r"\s+", " ", p.strip().lower())
                if p and p not in seen:
                    out[phase].append(p)
                    seen.add(p)
        return out

    # ---------------------------------------------------------------------
    # Generic weak prompts, used only as fallback/background evidence.
    # Avoid abstract wording like "successfully completed" as much as possible.
    # ---------------------------------------------------------------------
    add(
        "precondition",
        f"the relevant object for {label} is visible before the action",
        f"the robot has not yet changed the state of the {obj}",
        f"the {obj} is still in its initial state",
    )
    add(
        "contact_or_transition",
        f"the robot gripper is near the {obj}",
        f"the robot is interacting with the {obj}",
        f"the {obj} is moving because of the robot",
    )
    add(
        "postcondition_success",
        f"the {obj} is in the final state after the action",
        f"the robot has changed the state of the {obj}",
    )
    add(
        "failed_attempt",
        f"the robot touches the {obj} but the {obj} remains in its initial state",
        f"the robot interacts with the {obj} but does not achieve the final state",
    )

    # ---------------------------------------------------------------------
    # Button / stop-button predicates.
    # ---------------------------------------------------------------------
    is_button = "button" in label or "buttons" in label
    is_stop_cube = "stop cube" in label or "stop the cube" in label or (
        "cube" in label and "target" in label and "stop" in label
    )

    if is_button:
        button_obj = "buttons" if "both buttons" in label or "buttons" in label else "button"
        add(
            "precondition",
            f"the {button_obj} is visible and not pressed",
            f"the robot gripper is away from the {button_obj}",
            f"the {button_obj} has not been pressed yet",
        )
        add(
            "contact_or_transition",
            f"the robot gripper is touching the {button_obj}",
            f"the robot is pressing down on the {button_obj}",
            f"the {button_obj} is being depressed by the robot",
        )
        add(
            "postcondition_success",
            f"the {button_obj} is pressed down",
            f"the robot has pressed the {button_obj}",
        )
        add(
            "failed_attempt",
            f"the robot reaches for the {button_obj} but does not press it",
            f"the robot touches near the {button_obj} but the {button_obj} is not depressed",
        )

    if is_stop_cube:
        add(
            "precondition",
            "the cube is still moving and has not been stopped at the target",
            "the cube is approaching the target but the stop button has not been pressed",
        )
        add(
            "contact_or_transition",
            "the robot is pressing the stop button while the cube is at the target",
            "the cube is reaching the target and the robot is pressing the button",
        )
        add(
            "postcondition_success",
            "the cube is stopped at the target",
            "the cube is stationary at the target after the button press",
        )
        add(
            "failed_attempt",
            "the button is pressed when the cube is not at the target",
            "the cube passes the target without being stopped correctly",
        )

    # ---------------------------------------------------------------------
    # Pick / grasp / lift predicates.
    # ---------------------------------------------------------------------
    is_pick = (
        "pick up" in label
        or "pickup" in label
        or "pick " in label
        or "grasp" in label
        or "lift" in label
    )

    if is_pick:
        target_obj = obj

        if "highlight" in label:
            target_obj = "highlighted cube"
            add("precondition",
                "the highlighted cube is resting on the table",
                "the cube that was highlighted is not yet held by the robot",
                "the previously highlighted cube is visible before being picked")
            add("contact_or_transition",
                "the robot gripper is grasping the highlighted cube",
                "the highlighted cube is being lifted by the gripper")
            add("postcondition_success",
                "the highlighted cube is held by the robot gripper",
                "the highlighted cube is lifted off the table")
            add("failed_attempt",
                "the robot grasps a cube that was not highlighted",
                "the robot touches the highlighted cube but does not lift it")

        elif "same" in label and ("previously picked" in label or "picked before" in label):
            target_obj = "same previously picked cube"
            add("precondition",
                "the same cube from the demonstration is resting on the table",
                "the previously picked cube is not yet held by the robot")
            add("contact_or_transition",
                "the robot gripper is grasping the same cube from the demonstration",
                "the same previously picked cube is being lifted")
            add("postcondition_success",
                "the same previously picked cube is held by the robot gripper",
                "the same cube from the demonstration is lifted off the table")
            add("failed_attempt",
                "the robot picks up a different cube than the demonstrated one",
                "the robot touches the demonstrated cube but does not lift it")

        elif "container" in label and "hiding" in label:
            target_obj = "correct container"
            add("precondition",
                "the correct container is resting on the table",
                "the container hiding the specified cube is not yet held",
                "the container is covering a cube on the table")
            add("contact_or_transition",
                "the robot gripper is grasping the correct container",
                "the container hiding the specified cube is being lifted")
            add("postcondition_success",
                "the correct container is held by the robot gripper",
                "the container hiding the specified cube is lifted off the table",
                "the hidden cube is revealed after the correct container is lifted")
            add("failed_attempt",
                "the robot picks up a container hiding the wrong cube",
                "the robot touches the correct container but does not lift it",
                "the robot lifts an incorrect container")

        elif "peg" in label:
            target_obj = "peg"
            add("precondition",
                "the peg is resting on the table",
                "the specified end of the peg is not yet grasped")
            add("contact_or_transition",
                "the robot gripper is grasping the specified end of the peg",
                "the peg is being lifted by the gripper")
            add("postcondition_success",
                "the peg is held by the robot gripper at the specified end",
                "the peg is lifted and controlled by the gripper")
            add("failed_attempt",
                "the robot grasps the wrong end of the peg",
                "the robot touches the peg but does not lift it")

        else:
            add("precondition",
                f"the {target_obj} is resting on the table",
                f"the {target_obj} is not held by the robot gripper",
                f"the robot gripper is open near the {target_obj}")
            add("contact_or_transition",
                f"the robot gripper is closing around the {target_obj}",
                f"the robot gripper is grasping the {target_obj}",
                f"the {target_obj} is being lifted by the gripper")
            add("postcondition_success",
                f"the {target_obj} is held by the robot gripper",
                f"the {target_obj} is lifted off the table",
                f"the {target_obj} is no longer resting on the surface")
            add("failed_attempt",
                f"the robot gripper touches the {target_obj} but does not lift it",
                f"the {target_obj} slips or remains on the table",
                f"the robot grasps the wrong object instead of the {target_obj}")

    # ---------------------------------------------------------------------
    # Place / put / drop predicates.
    # ---------------------------------------------------------------------
    is_place = (
        "place" in label
        or "put " in label
        or "put down" in label
        or "drop" in label
        or "move" in label and "target" in label
    )

    if is_place:
        placed_obj = obj

        if "bin" in label:
            add("precondition",
                f"the {placed_obj} is outside the bin",
                f"the robot is holding or approaching the {placed_obj} before putting it in the bin")
            add("contact_or_transition",
                f"the robot is moving the {placed_obj} over the bin",
                f"the {placed_obj} is being lowered into the bin",
                f"the {placed_obj} is entering the bin")
            add("postcondition_success",
                f"the {placed_obj} is inside the bin",
                f"the {placed_obj} has been released into the bin",
                f"the robot gripper is empty and the {placed_obj} is in the bin")
            add("failed_attempt",
                f"the {placed_obj} misses the bin",
                f"the {placed_obj} is dropped outside the bin",
                f"the robot releases the {placed_obj} before it is inside the bin")

        elif "target" in label:
            if "before the button" in label or "after the button" in label or "previously placed" in label:
                target_desc = "demonstrated target"
            elif "right-side" in label or "right side" in label:
                target_desc = "right-side target"
            elif "left-side" in label or "left side" in label:
                target_desc = "left-side target"
            else:
                target_desc = "target"

            add("precondition",
                f"the {placed_obj} is not yet on the {target_desc}",
                f"the robot is holding or approaching the {placed_obj} before placing it on the {target_desc}")
            add("contact_or_transition",
                f"the robot is moving the {placed_obj} toward the {target_desc}",
                f"the {placed_obj} is being lowered onto the {target_desc}",
                f"the {placed_obj} is above the {target_desc}")
            add("postcondition_success",
                f"the {placed_obj} is resting on the {target_desc}",
                f"the {placed_obj} has been released on the {target_desc}",
                f"the robot gripper is empty and the {placed_obj} is on the {target_desc}")
            add("failed_attempt",
                f"the {placed_obj} is placed on the wrong target",
                f"the {placed_obj} misses the {target_desc}",
                f"the robot releases the {placed_obj} away from the {target_desc}")

        else:
            add("precondition",
                f"the robot is holding the {placed_obj}",
                f"the {placed_obj} is above the table before being released")
            add("contact_or_transition",
                f"the robot is lowering the {placed_obj}",
                f"the robot gripper is releasing the {placed_obj}",
                f"the {placed_obj} is moving downward from the gripper")
            add("postcondition_success",
                f"the {placed_obj} is resting on the table",
                f"the {placed_obj} has been released by the gripper",
                f"the robot gripper is empty after releasing the {placed_obj}")
            add("failed_attempt",
                f"the {placed_obj} is dropped incorrectly",
                f"the {placed_obj} slips or falls away from the intended location")

    # ---------------------------------------------------------------------
    # Swing / move-to-target predicates.
    # ---------------------------------------------------------------------
    is_swing = "swing" in label or "right-to-left" in label or (
        "right-side target" in label and "left-side target" in label
    )
    is_move_cube = "move" in label and ("cube" in label or "block" in label)

    if is_swing:
        add("precondition",
            "the cube is held before moving between the two targets",
            "the cube has not yet reached the next target")
        add("contact_or_transition",
            "the robot is carrying the cube between the left-side and right-side targets",
            "the cube is moving from one target toward the other target")
        add("postcondition_success",
            "the cube reaches the specified side target while held by the gripper",
            "the cube is positioned above the correct side target")
        add("failed_attempt",
            "the cube moves toward the wrong side target",
            "the cube fails to reach the specified side target")

    elif is_move_cube:
        add("precondition",
            "the cube is at its starting location before the demonstrated movement",
            "the cube has not yet been moved in the demonstrated manner")
        add("contact_or_transition",
            "the robot is moving the cube along the demonstrated path",
            "the cube is sliding or being carried toward the target")
        add("postcondition_success",
            "the cube reaches the target in the demonstrated manner",
            "the cube is at the final target location")
        add("failed_attempt",
            "the cube moves along a different path than the demonstration",
            "the cube does not reach the target")

    # ---------------------------------------------------------------------
    # Peg insertion predicates.
    # ---------------------------------------------------------------------
    if "insert" in label and "peg" in label:
        add("precondition",
            "the peg is outside the box before insertion",
            "the hole or side of the box is empty before the peg is inserted",
            "the robot is holding the peg before aligning it with the box")
        add("contact_or_transition",
            "the peg is aligned with the side of the box",
            "the robot is pushing the peg into the box",
            "the tip of the peg is entering the hole in the box")
        add("postcondition_success",
            "the peg is inserted into the side of the box",
            "part of the peg remains inside the box after insertion",
            "the peg is seated in the box opening")
        add("failed_attempt",
            "the peg touches the box but does not enter the hole",
            "the peg is inserted into the wrong side of the box",
            "the peg is misaligned with the box opening")

    # ---------------------------------------------------------------------
    # PatternLock / RouteStick predicates.
    # ---------------------------------------------------------------------
    is_pattern = "pattern" in label or "retrace" in label or "trace" in label
    is_route = "route" in label or "navigate around" in label or (
        "stick" in label and "path" in label
    )

    if is_pattern:
        add("precondition",
            "the stick tip is at the start of the pattern",
            "the robot has not yet begun tracing the pattern",
            "the pattern is visible on the table before tracing")
        add("contact_or_transition",
            "the stick tip is touching the pattern on the table",
            "the robot is moving the stick along the pattern",
            "the stick is tracing a line on the pattern")
        add("postcondition_success",
            "the stick reaches the end of the pattern",
            "the full pattern has been retraced by the stick",
            "the stick is at the final point of the demonstrated pattern")
        add("failed_attempt",
            "the stick tip moves away from the pattern",
            "the robot traces the wrong pattern",
            "the stick deviates from the demonstrated path")

    if is_route:
        add("precondition",
            "the stick is at the start of the route",
            "the robot has not yet navigated around the obstacles",
            "the obstacle sticks are visible on the table")
        add("contact_or_transition",
            "the robot is moving the stick around the obstacle sticks",
            "the stick is following the demonstrated route",
            "the stick is navigating between obstacles on the table")
        add("postcondition_success",
            "the stick reaches the end of the route",
            "the robot completes the demonstrated path around the sticks",
            "the stick has followed the same route as in the video")
        add("failed_attempt",
            "the stick collides with an obstacle stick",
            "the stick follows the wrong route",
            "the robot deviates from the demonstrated path")

    # ---------------------------------------------------------------------
    # Explicit video-reference predicates.
    # ---------------------------------------------------------------------
    if "watch the video" in label or "demonstration" in label or "previously" in label:
        add("precondition",
            "the object referred to by the video is visible before the robot acts",
            "the demonstrated target or demonstrated object is visible before execution")
        add("contact_or_transition",
            "the robot is interacting with the same object referred to by the video",
            "the robot is moving toward the target specified by the video")
        add("postcondition_success",
            "the robot has manipulated the same object referred to by the video",
            "the object is placed at the target specified by the video")
        add("failed_attempt",
            "the robot manipulates a different object than the one referred to by the video",
            "the robot uses the wrong target compared with the video")

    return dedupe()


def _parse_action_object(label: str) -> tuple[str, str]:
    """Extract the main verb and primary object from a subtask label."""
    label = label.strip().lower()

    def strip_leading_article(s: str) -> str:
        return re.sub(r"^(the|a|an)\s+", "", s.strip())

    multi_word_verbs = [
        "pick up", "put down", "push down", "pull out", "pull up",
        "push in", "turn on", "turn off", "flip over",
    ]

    verb = None
    rest = ""

    for mv in sorted(multi_word_verbs, key=len, reverse=True):
        if re.match(rf"^{re.escape(mv)}(\s+|$)", label):
            verb = mv
            rest = label[len(mv):].strip()
            break

    if verb is None:
        parts = label.split(maxsplit=1)
        verb = parts[0] if parts else label
        rest = parts[1].strip() if len(parts) > 1 else ""

    rest = strip_leading_article(rest)

    relation_patterns = [
        r"\s+in\s+", r"\s+into\s+", r"\s+on\s+", r"\s+onto\s+",
        r"\s+to\s+", r"\s+inside\s+", r"\s+under\s+", r"\s+over\s+",
        r"\s+next\s+to\s+",
    ]

    obj = rest
    if verb in {"place", "put", "move", "drop"}:
        for pat in relation_patterns:
            obj = re.split(pat, rest, maxsplit=1)[0].strip()
            if obj != rest:
                break

    obj = strip_leading_article(obj)
    if not obj:
        obj = "object"

    return verb, obj


@dataclass
class CLIPPhaseScores:
    """Per-frame CLIP similarity scores.

    Only precondition, contact_or_transition, and postcondition_success are
    computed by CLIP.  failed_attempt is always zeros — it is derived
    temporally by label_fusion, not by CLIP.
    """

    frame_indices: np.ndarray          # (N,)
    precondition: np.ndarray           # (N,)
    contact_or_transition: np.ndarray  # (N,)
    postcondition_success: np.ndarray  # (N,)
    failed_attempt: np.ndarray         # (N,) — always zeros from CLIP

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "frame_indices": self.frame_indices,
            "precondition": self.precondition,
            "contact_or_transition": self.contact_or_transition,
            "postcondition_success": self.postcondition_success,
            "failed_attempt": self.failed_attempt,
        }

    def smoothed(self, kernel_size: int = 5) -> "CLIPPhaseScores":
        """Return a copy with scores smoothed using a uniform kernel."""
        if kernel_size <= 1:
            return self
        kernel = np.ones(kernel_size) / kernel_size
        return CLIPPhaseScores(
            frame_indices=self.frame_indices,
            precondition=np.convolve(self.precondition, kernel, mode="same"),
            contact_or_transition=np.convolve(
                self.contact_or_transition, kernel, mode="same"
            ),
            postcondition_success=np.convolve(
                self.postcondition_success, kernel, mode="same"
            ),
            failed_attempt=self.failed_attempt.copy(),  # zeros, no smoothing
        )


class CLIPScorer:
    """Scores frames against state-predicate text prompts using CLIP.

    Only scores 3 phases: precondition, contact_or_transition,
    postcondition_success.  failed_attempt is zero-filled (temporal-only).
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size

        print(f"Loading CLIP model: {model_name} on {device}")
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print("CLIP model loaded.")

    @torch.no_grad()
    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode text prompts -> (N, D) normalized embeddings."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()
                  if k in ("input_ids", "attention_mask")}
        emb = self.model.get_text_features(**inputs)
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def _encode_images(self, images: list[np.ndarray]) -> torch.Tensor:
        """Encode RGB images (H,W,3 uint8) -> (N, D) normalized embeddings."""
        pil_images = [Image.fromarray(img) for img in images]
        all_embs = []
        for i in range(0, len(pil_images), self.batch_size):
            batch = pil_images[i : i + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()
                      if k == "pixel_values"}
            emb = self.model.get_image_features(**inputs)
            all_embs.append(F.normalize(emb, dim=-1).cpu())
        return torch.cat(all_embs, dim=0)

    def score_segment(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
        subtask_label: str,
    ) -> CLIPPhaseScores:
        """Score all frames against 3 phase prompts (no failed_attempt).

        Args:
            frames: List of front-camera RGB images (H,W,3 uint8).
            frame_indices: Corresponding global timestep indices.
            subtask_label: The atomic subtask label.

        Returns:
            CLIPPhaseScores with precondition/contact/postcondition scored
            and failed_attempt zeroed.
        """
        prompts = generate_phase_prompts(subtask_label)

        # Only encode the 3 CLIP-scorable phases
        phase_text_embs: dict[str, torch.Tensor] = {}
        for phase_name in CLIP_PHASE_NAMES:
            phase_text_embs[phase_name] = self._encode_texts(
                prompts[phase_name]
            ).cpu()

        # Encode all frames
        image_embs = self._encode_images(frames)  # (N, D)

        # Compute per-phase scores: max similarity across prompts per frame
        scores: dict[str, np.ndarray] = {}
        for phase_name in CLIP_PHASE_NAMES:
            sim = image_embs @ phase_text_embs[phase_name].T  # (N, P)
            scores[phase_name] = sim.max(dim=-1).values.numpy() #FLAG: Is max the best way to aggregate multiple prompts per phase? Should we try mean or something else instead?

        n = len(frame_indices)
        return CLIPPhaseScores(
            frame_indices=np.array(frame_indices),
            precondition=scores["precondition"],
            contact_or_transition=scores["contact_or_transition"],
            postcondition_success=scores["postcondition_success"],
            failed_attempt=np.zeros(n, dtype=np.float32),
        )
