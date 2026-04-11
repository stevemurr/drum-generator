"""
synthetic.py
------------
SyntheticDrumDataset: procedural drum sound generation via basic DSP.

Generates kick, snare, hi-hat, clap, tom, rimshot, and cymbal sounds
using sine sweeps, filtered noise, and amplitude envelopes. All synthesis
is pure torch/torchaudio math — no external DSP libraries required.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from drum_generator.config import CFG
from drum_generator.dataset.caption import CHAR_TAGS, ClapEmbedder, build_synthetic_caption

DRUM_TYPES = [
    "kick",
    "snare",
    "hihat_closed",
    "hihat_open",
    "clap",
    "tom",
    "rimshot",
    "cymbal",
]


def _biquad_lowpass(waveform: torch.Tensor, sr: int, cutoff: float) -> torch.Tensor:
    import torchaudio.functional as AF

    return AF.lowpass_biquad(waveform.unsqueeze(0), sr, cutoff).squeeze(0)


def _biquad_highpass(waveform: torch.Tensor, sr: int, cutoff: float) -> torch.Tensor:
    import torchaudio.functional as AF

    return AF.highpass_biquad(waveform.unsqueeze(0), sr, cutoff).squeeze(0)


def _biquad_bandpass(
    waveform: torch.Tensor, sr: int, low: float, high: float
) -> torch.Tensor:
    w = _biquad_highpass(waveform, sr, low)
    return _biquad_lowpass(w, sr, high)


def _normalize(waveform: torch.Tensor) -> torch.Tensor:
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    return waveform


# ---------------------------------------------------------------------------
# Synthesis functions — one per drum type
# ---------------------------------------------------------------------------


def _synth_kick(t: torch.Tensor, sr: int, rng: np.random.RandomState) -> torch.Tensor:
    """Pitch-swept sine + optional sub layer + transient click."""
    f0 = rng.uniform(150, 300)
    f1 = rng.uniform(30, 60)
    pitch_decay = rng.uniform(3, 15)
    amp_decay = rng.uniform(5, 20)

    freq_sweep = f1 + (f0 - f1) * torch.exp(-pitch_decay * t)
    phase = 2 * torch.pi * torch.cumsum(freq_sweep / sr, dim=0)
    body = torch.sin(phase) * torch.exp(-amp_decay * t)

    # Transient click
    click_len = int(0.002 * sr)
    click = torch.randn(click_len) * torch.linspace(1, 0, click_len)
    click_gain = rng.uniform(0.1, 0.5)
    body[:click_len] = body[:click_len] + click * click_gain

    # Optional sub layer
    if rng.random() < 0.4:
        sub_freq = rng.uniform(30, 50)
        sub_decay = rng.uniform(3, 8)
        sub_gain = rng.uniform(0.3, 0.7)
        sub = torch.sin(2 * torch.pi * sub_freq * t) * torch.exp(-sub_decay * t) * sub_gain
        body = body + sub

    return body


def _synth_snare(t: torch.Tensor, sr: int, rng: np.random.RandomState) -> torch.Tensor:
    """Damped sine body + bandpass-filtered noise burst."""
    tone_freq = rng.uniform(150, 250)
    tone_decay = rng.uniform(15, 30)
    tone = torch.sin(2 * torch.pi * tone_freq * t) * torch.exp(-tone_decay * t)

    noise = torch.randn_like(t)
    noise_low = rng.uniform(1000, 3000)
    noise_high = rng.uniform(6000, 10000)
    noise = _biquad_bandpass(noise, sr, noise_low, noise_high)
    noise_decay = rng.uniform(8, 20)
    noise = noise * torch.exp(-noise_decay * t)

    mix = rng.uniform(0.3, 0.7)
    return mix * tone + (1 - mix) * noise


def _synth_hihat(
    t: torch.Tensor, sr: int, rng: np.random.RandomState, open_hat: bool = False
) -> torch.Tensor:
    """Highpass-filtered noise with optional metallic ring modulation."""
    noise = torch.randn_like(t)
    cutoff = rng.uniform(5000, 12000)
    noise = _biquad_highpass(noise, sr, cutoff)

    if open_hat:
        decay = rng.uniform(5, 15)
    else:
        decay = rng.uniform(30, 80)

    env = torch.exp(-decay * t)

    # Optional metallic ring modulation
    if rng.random() < 0.5:
        metal_freq = rng.uniform(4000, 8000)
        noise = noise * (1 + 0.5 * torch.sin(2 * torch.pi * metal_freq * t))

    return noise * env


def _synth_clap(t: torch.Tensor, sr: int, rng: np.random.RandomState) -> torch.Tensor:
    """Multiple short noise bursts with slight delays (flamming)."""
    n_bursts = rng.randint(2, 5)
    burst_len = int(rng.uniform(0.005, 0.015) * sr)
    spacing = int(rng.uniform(0.01, 0.03) * sr)

    result = torch.zeros_like(t)
    for i in range(n_bursts):
        offset = i * spacing
        if offset + burst_len >= len(t):
            break
        burst = torch.randn(burst_len) * torch.linspace(1, 0, burst_len)
        result[offset : offset + burst_len] = result[offset : offset + burst_len] + burst

    # Bandpass filter
    bp_low = rng.uniform(500, 1500)
    bp_high = rng.uniform(3000, 8000)
    result = _biquad_bandpass(result, sr, bp_low, bp_high)

    # Tail decay
    tail_decay = rng.uniform(10, 25)
    result = result * torch.exp(-tail_decay * t)

    return result


def _synth_tom(t: torch.Tensor, sr: int, rng: np.random.RandomState) -> torch.Tensor:
    """Slower sine sweep than kick, moderate decay."""
    f0 = rng.uniform(100, 250)
    f1 = rng.uniform(60, 120)
    pitch_decay = rng.uniform(2, 8)
    amp_decay = rng.uniform(4, 12)

    freq_sweep = f1 + (f0 - f1) * torch.exp(-pitch_decay * t)
    phase = 2 * torch.pi * torch.cumsum(freq_sweep / sr, dim=0)
    body = torch.sin(phase) * torch.exp(-amp_decay * t)

    return body


def _synth_rimshot(t: torch.Tensor, sr: int, rng: np.random.RandomState) -> torch.Tensor:
    """Short high-pitched click + metallic ring."""
    # Click
    click_freq = rng.uniform(2000, 5000)
    click_decay = rng.uniform(40, 80)
    click = torch.sin(2 * torch.pi * click_freq * t) * torch.exp(-click_decay * t)

    # Metallic ring (inharmonic)
    ring_freq = rng.uniform(3000, 7000)
    ring_decay = rng.uniform(15, 30)
    ring = torch.sin(2 * torch.pi * ring_freq * t) * torch.exp(-ring_decay * t)
    ring_gain = rng.uniform(0.3, 0.6)

    return click + ring * ring_gain


def _synth_cymbal(t: torch.Tensor, sr: int, rng: np.random.RandomState) -> torch.Tensor:
    """Inharmonic sine cluster with slow decay."""
    n_partials = rng.randint(5, 12)
    result = torch.zeros_like(t)

    for _ in range(n_partials):
        freq = rng.uniform(2000, 14000)
        decay = rng.uniform(2, 8)
        amp = rng.uniform(0.3, 1.0)
        result = result + amp * torch.sin(2 * torch.pi * freq * t) * torch.exp(-decay * t)

    # Highpass to remove mud
    result = _biquad_highpass(result, sr, rng.uniform(3000, 6000))
    return result


_SYNTH_FUNCS = {
    "kick": _synth_kick,
    "snare": _synth_snare,
    "hihat_closed": lambda t, sr, rng: _synth_hihat(t, sr, rng, open_hat=False),
    "hihat_open": lambda t, sr, rng: _synth_hihat(t, sr, rng, open_hat=True),
    "clap": _synth_clap,
    "tom": _synth_tom,
    "rimshot": _synth_rimshot,
    "cymbal": _synth_cymbal,
}

# Characteristic tags that make sense per drum type
_TYPE_CHARS: dict[str, list[str]] = {
    "kick": ["punchy", "sub", "boomy", "tight", "hard", "soft", "warm", "dark", "clean", "distorted", "electronic", "acoustic", "trap", "house"],
    "snare": ["snappy", "crisp", "tight", "bright", "warm", "dry", "wet", "compressed", "distorted", "electronic", "acoustic", "vintage"],
    "hihat_closed": ["crisp", "tight", "bright", "dark", "electronic", "acoustic", "clean", "distorted"],
    "hihat_open": ["crisp", "bright", "dark", "electronic", "acoustic", "clean"],
    "clap": ["tight", "roomy", "dry", "wet", "electronic", "vintage", "compressed", "house", "trap"],
    "tom": ["warm", "boomy", "punchy", "tight", "acoustic", "electronic", "dark", "bright"],
    "rimshot": ["crisp", "snappy", "bright", "tight", "hard", "clean", "acoustic"],
    "cymbal": ["bright", "dark", "crisp", "warm", "clean", "distorted"],
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SyntheticDrumDataset(Dataset):
    """Generates synthetic drum sounds programmatically.

    Each __getitem__ call deterministically generates a sound based on
    the index (via seeded RNG), ensuring reproducibility.

    Returns (waveform [N_SAMPLES], clap_embed [CLAP_DIM]).
    """

    def __init__(
        self,
        size: int = 2000,
        sample_rate: int = CFG.sample_rate,
        n_samples: int = CFG.n_samples,
        clap_embedder: ClapEmbedder | None = None,
        seed: int = 42,
    ):
        self.size = size
        self.sr = sample_rate
        self.n_samples = n_samples
        self.seed = seed

        # Pre-generate parameter table for reproducibility
        rng = np.random.RandomState(seed)
        self.params: list[dict] = []
        for _ in range(size):
            drum_type = DRUM_TYPES[rng.randint(len(DRUM_TYPES))]
            available_chars = _TYPE_CHARS.get(drum_type, [])
            n_chars = min(rng.randint(1, 4), len(available_chars))
            chosen = list(rng.choice(available_chars, n_chars, replace=False))
            self.params.append({"drum_type": drum_type, "characteristics": chosen})

        # Pre-compute CLAP embeddings (captions are known upfront)
        clap = clap_embedder or ClapEmbedder.get()
        print(f"[synthetic] pre-computing CLAP embeddings for {size} synthetic sounds...")
        self.clap_cache = torch.zeros(size, CFG.clap_dim)
        for i, p in enumerate(self.params):
            caption = build_synthetic_caption(p["drum_type"], p["characteristics"])
            self.clap_cache[i] = clap.embed(caption)
        print("[synthetic] CLAP embeddings ready")

    def _synthesize(self, idx: int) -> torch.Tensor:
        """Generate waveform for the given index."""
        params = self.params[idx]
        rng = np.random.RandomState(self.seed + idx)

        t = torch.linspace(0, self.n_samples / self.sr, self.n_samples)
        synth_fn = _SYNTH_FUNCS[params["drum_type"]]
        waveform = synth_fn(t, self.sr, rng)

        waveform = _normalize(waveform)

        # Ensure exact length
        if waveform.shape[0] < self.n_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.n_samples - waveform.shape[0]))
        elif waveform.shape[0] > self.n_samples:
            waveform = waveform[: self.n_samples]

        return waveform

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        waveform = self._synthesize(idx)
        embed = self.clap_cache[idx]
        return waveform, embed
