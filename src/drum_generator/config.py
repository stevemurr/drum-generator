import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # Audio
    sample_rate: int = 44100
    duration: float = 1.5  # seconds — covers all one-shots
    n_samples: int = 66150  # sample_rate * duration

    # DAC latent shape (Descript Audio Codec 44kHz)
    # encoder outputs 1024-dim continuous latent, ~129 frames for 1.5s
    dac_latent_dim: int = 1024
    dac_time_frames: int = 129  # floor(n_samples / 512)

    # VAE (compresses DAC latents further for DiT to work in)
    vae_latent_dim: int = 16  # per time frame
    vae_hidden: int = 512  # base hidden dim for gradual compression

    # CLAP text embedding dim (laion/larger_clap_general)
    clap_dim: int = 512

    # DiT
    dit_dim: int = 256  # transformer hidden dim
    dit_heads: int = 8
    dit_layers: int = 6
    dit_patch_size: int = 3  # merge 3 time frames per token → 43 tokens

    # Flow matching
    fm_steps_train: int = 1  # CFM samples one t per step
    fm_steps_infer: int = 8  # Euler ODE steps at inference

    # Training
    batch_size: int = 16
    lr: float = 1e-4
    vae_epochs: int = 100
    vae_eta_min: float = 0.0  # CosineAnnealingLR floor (0 = schedule ends at 0 LR)
    vae_kl_weight: float = 1e-2  # KL regularization strength (higher = smoother latent space)
    vae_kl_warmup: int = 20  # epochs to ramp KL weight from 0 → vae_kl_weight
    vae_stft_weight: float = 0.0  # linear multi-res STFT loss weight (0 disables)
    vae_stft_phase_weight: float = 0.0  # phase term (w_phs) inside the linear MRSTFT
    vae_stft_mel_weight: float = 0.0  # separate mel-scaled MRSTFT loss weight (0 disables)
    vae_stft_weighted: bool = False  # use per-bin weighted linear MRSTFT (drum weight curve)
    vae_lowpass_weight: float = 0.0  # L1 on lowpassed waveform, targets sub-cutoff content (0 disables)
    vae_lowpass_cutoff: float = 500.0  # lowpass cutoff in Hz for the L1 term
    dit_epochs: int = 500
    dit_eta_min: float = 0.0  # CosineAnnealingLR floor (0 = schedule ends at 0 LR)
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

    # DAC latent caching (live path: encode on first access, store as .pt files)
    cache: bool = False  # pre-encode all samples through DAC
    dac_cache_dir: str = "cache"

    # Precomputed memmap path (fast path: skips wav decode / DAC encode / CLAP
    # entirely). When set, build_dataset() returns a MemmapDACDataset from
    # this directory — must contain dac_latents.npy, embeddings_text.npy,
    # index.json. Produced by the companion dataset-caption pipeline. No
    # augmentation in memmap mode — use the live path if you need pitch shift
    # / reverb / etc.
    memmap_dir: str | None = None


CFG = Config()
