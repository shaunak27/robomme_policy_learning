"""Configuration for the Query-conditioned KeyFrame Selector (QKFS)."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class QKFSConfig:
    # ---- Architecture ----
    # Embedding dimensions (must match SigLIP outputs)
    frame_emb_dim: int = 2048       # SigLIP global embedding dim
    pos_emb_dim: int = 768          # positional embedding dim
    proprio_dim: int = 8            # proprioceptive state dim
    instruction_emb_dim: int = 256  # projected instruction embedding dim

    # Transformer dimensions
    hidden_dim: int = 256           # internal transformer dimension
    num_heads: int = 4              # attention heads
    num_query_layers: int = 2       # QueryEncoder transformer layers
    num_selector_layers: int = 2    # Selector cross-attention layers
    mlp_dim: int = 512              # feedforward hidden dim in transformer
    dropout_rate: float = 0.1

    # Query context
    num_recent_frames: int = 4      # R: number of recent frames in query

    # ---- Frame selection ----
    num_frames_to_select: int = 32  # K: total memory budget in frames
    max_candidates: int = 512       # max past frames to score
    reserve_recent: int = 4         # frames reserved for most recent obs
    max_segments: int = 32          # fixed pad size for segment arrays (avoids JIT retrace)

    # ---- Target distribution ----
    sigma: float = 2.0              # Gaussian width for soft keyframe targets
    epsilon: float = 1e-6           # numerical stability for KL

    # ---- Loss ----
    lambda_frame: float = 1.0       # weight for within-subtask frame loss

    # ---- Inference diversity ----
    alpha: float = 0.3              # visual similarity penalty weight
    beta: float = 0.3               # temporal closeness penalty weight
    tau: float = 8.0                # temporal closeness decay scale

    # ---- Temporal encoding ----
    max_episode_length: int = 1024  # max frames for learned temporal embedding
    use_relative_time: bool = True  # use relative age vs absolute position

    # ---- Training ----
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    batch_size: int = 64
    num_train_steps: int = 50_000
    warmup_steps: int = 1000
    log_interval: int = 50
    save_interval: int = 5000

    # ---- Paths ----
    dataset_path: str = ""
    topreward_dir: str = ""         # path to topreward_full labels
    checkpoint_dir: str = "runs/ckpts/qkfs"

    # ---- Memory layout (must match VLA checkpoint) ----
    token_budget: int = 512
    token_per_image: int = 16       # 4x4 spatial grid
    num_views: int = 1

    # ---- Wandb ----
    wandb_enabled: bool = True
    wandb_project: str = "qkfs-selector"
    exp_name: str = "qkfs_v1"
