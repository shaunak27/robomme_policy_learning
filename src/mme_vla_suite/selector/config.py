"""Configuration for the RL-based frame selector."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class SelectorConfig:
    # ---- Architecture ----
    query_dim: int = 256
    scorer_hidden_dim: int = 256
    num_frames_to_select: int = 32
    max_candidates: int = 128
    use_episode_progress: bool = True
    neighbor_suppression_radius: int = 0  # 0 = disabled

    # Candidate features
    candidate_emb_dim: int = 2048  # SigLIP embedding dim (mean-pooled from cached 8x8 grid)

    # Query features
    front_view_emb_dim: int = 2048
    instruction_emb_dim: int = 256
    proprio_dim: int = 8

    # ---- PPO ----
    gamma: float = 0.0
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 16

    # ---- Reward ----
    redundancy_penalty_coef: float = 0.01
    redundancy_time_threshold: int = 2  # frames closer than this are "near-duplicate"

    # ---- Training ----
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    batch_size: int = 64
    num_train_steps: int = 50_000
    log_interval: int = 10
    save_interval: int = 1000
    eval_interval: int = 2000

    # ---- Stage 2 fine-tuning ----
    stage2_unfreeze_modulation: bool = False
    stage2_unfreeze_action_expert_lora: bool = False

    # ---- Paths ----
    vla_checkpoint_path: str = ""
    dataset_path: str = ""
    selector_checkpoint_dir: str = "runs/ckpts/rl_selector"
    history_config: str = "perceptual-rlselector-modul.yaml"

    # ---- Memory layout (must match the base VLA checkpoint) ----
    token_budget: int = 512
    token_per_image: int = 16  # 4x4 spatial grid used for VLA memory tokens
    num_views: int = 1

    # ---- Wandb ----
    wandb_enabled: bool = True
    wandb_project: str = "rl-frame-selector"
    exp_name: str = "rl_selector_v1"
