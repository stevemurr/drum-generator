import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # Audio
    sample_rate: int = 44100
    duration: float = 1.5  # seconds — covers all one-shots
    n_samples: int = 66150  # sample_rate * duration

    # DAC latent shape (Descript Audio Codec)
    # encoder compresses 512x → latent dim 64, ~130 frames for 1.5s
    dac_latent_dim: int = 64
    dac_time_frames: int = 130  # ceil(n_samples / 512)

    # VAE (compresses DAC latents further for DiT to work in)
    vae_latent_dim: int = 16  # per time frame
    vae_hidden: int = 256

    # CLAP text embedding dim (laion/larger_clap_general)
    clap_dim: int = 512

    # DiT
    dit_dim: int = 256  # transformer hidden dim
    dit_heads: int = 8
    dit_layers: int = 6
    dit_patch_size: int = 4  # merge 4 time frames per token → ~32 tokens

    # Flow matching
    fm_steps_train: int = 1  # CFM samples one t per step
    fm_steps_infer: int = 8  # Euler ODE steps at inference

    # Training
    batch_size: int = 16
    lr: float = 1e-4
    vae_epochs: int = 100
    dit_epochs: int = 500
    cfg_dropout: float = 0.1  # prob of dropping text cond (CFG training)
    cfg_scale: float = 4.0  # guidance scale at inference
    ref_dropout: float = 0.5  # prob of dropping audio reference (high → text-only works well)
    ref_cfg_scale: float = 2.0  # guidance scale for reference at inference

    # Paths
    freesound_token: str = field(default_factory=lambda: os.environ.get("FREESOUND_TOKEN", ""))
    data_dir: str = "data"
    ckpt_dir: str = "checkpoints"

    # Dataset sources
    disk_dirs: list[str] = field(default_factory=list)
    disk_label_file: str | None = None
    synthetic_size: int = 0  # 0 = disabled
    synthetic_seed: int = 42

    # Augmentation
    augment: bool = True
    augment_transforms: list[str] = field(
        default_factory=lambda: ["pitch_shift", "gain", "noise", "reverb", "filter", "polarity"]
    )
    augment_p: float = 0.5  # per-transform probability
    augment_multiplier: int = 1  # effective dataset size multiplier
    clap_cache_dir: str | None = None


CFG = Config()
