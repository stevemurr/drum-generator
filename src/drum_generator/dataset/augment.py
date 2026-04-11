"""
augment.py
----------
Audio augmentation transforms and AugmentedDataset wrapper.

Wraps any (waveform, clap_embed) dataset with on-the-fly random transforms.
CLAP embeddings are passed through unchanged — augmented audio retains
the same semantic meaning.
"""

import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from drum_generator.config import CFG

# ---------------------------------------------------------------------------
# Transform classes
# ---------------------------------------------------------------------------


class PitchShift:
    """Shift pitch by resampling trick (resample to shifted rate, then back)."""

    def __init__(self, semitone_range: tuple[float, float] = (-2.0, 2.0)):
        self.semitone_range = semitone_range

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        import torchaudio.functional as AF

        semitones = random.uniform(*self.semitone_range)
        ratio = 2.0 ** (semitones / 12.0)
        intermediate_sr = int(CFG.sample_rate * ratio)
        if intermediate_sr == CFG.sample_rate:
            return waveform

        # Resample up/down then back to original rate
        shifted = AF.resample(waveform.unsqueeze(0), CFG.sample_rate, intermediate_sr)
        shifted = AF.resample(shifted, intermediate_sr, CFG.sample_rate)
        return shifted.squeeze(0)


class GainVariation:
    """Random gain adjustment in dB."""

    def __init__(self, db_range: tuple[float, float] = (-6.0, 6.0)):
        self.db_range = db_range

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        db = random.uniform(*self.db_range)
        return waveform * (10.0 ** (db / 20.0))


class NoiseInjection:
    """Add Gaussian noise at a random SNR."""

    def __init__(self, snr_range: tuple[float, float] = (20.0, 40.0)):
        self.snr_range = snr_range

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        snr_db = random.uniform(*self.snr_range)
        signal_power = waveform.pow(2).mean()
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = torch.randn_like(waveform) * noise_power.sqrt()
        return waveform + noise


class SimpleReverb:
    """Convolve with a synthetic exponential-decay noise impulse response."""

    def __init__(
        self,
        decay_range: tuple[float, float] = (0.1, 0.5),
        mix_range: tuple[float, float] = (0.05, 0.3),
    ):
        self.decay_range = decay_range
        self.mix_range = mix_range

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        decay_time = random.uniform(*self.decay_range)
        wet_mix = random.uniform(*self.mix_range)

        ir_len = int(decay_time * CFG.sample_rate)
        if ir_len < 2:
            return waveform

        t = torch.linspace(0, decay_time, ir_len)
        ir = torch.randn(ir_len) * torch.exp(-5.0 * t / decay_time)
        ir = ir / ir.abs().sum()  # normalize IR energy

        # Convolve via FFT
        pad_len = ir_len - 1
        wet = torch.nn.functional.conv1d(
            waveform.view(1, 1, -1),
            ir.flip(0).view(1, 1, -1),
            padding=pad_len,
        ).squeeze()
        wet = wet[: waveform.shape[0]]  # trim to original length

        return (1 - wet_mix) * waveform + wet_mix * wet


class BandFilter:
    """Random lowpass or highpass filter."""

    def __init__(
        self,
        filter_types: tuple[str, ...] = ("lowpass", "highpass"),
        freq_range: tuple[int, int] = (200, 8000),
    ):
        self.filter_types = filter_types
        self.freq_range = freq_range

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        import torchaudio.functional as AF

        ftype = random.choice(self.filter_types)
        freq = random.uniform(*self.freq_range)
        w = waveform.unsqueeze(0)

        if ftype == "lowpass":
            w = AF.lowpass_biquad(w, CFG.sample_rate, freq)
        else:
            w = AF.highpass_biquad(w, CFG.sample_rate, freq)

        return w.squeeze(0)


class Polarity:
    """Randomly invert waveform polarity (50% chance)."""

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            return -waveform
        return waveform


class RandomStartOffset:
    """Shift waveform start by a random number of samples (circular or zero-pad)."""

    def __init__(self, max_offset_samples: int = 2205):  # ~50ms at 44100Hz
        self.max_offset = max_offset_samples

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        offset = random.randint(0, self.max_offset)
        if offset == 0:
            return waveform
        # Zero-pad shift: prepend silence, trim end
        return F.pad(waveform[:-offset], (offset, 0))


# ---------------------------------------------------------------------------
# Transform registry
# ---------------------------------------------------------------------------

TRANSFORM_REGISTRY: dict[str, type] = {
    "pitch_shift": PitchShift,
    "gain": GainVariation,
    "noise": NoiseInjection,
    "reverb": SimpleReverb,
    "filter": BandFilter,
    "polarity": Polarity,
    "offset": RandomStartOffset,
}


def build_transforms(names: list[str] | None = None) -> list:
    """Build transform instances from a list of names."""
    if names is None:
        names = list(TRANSFORM_REGISTRY.keys())
    return [TRANSFORM_REGISTRY[n]() for n in names if n in TRANSFORM_REGISTRY]


# ---------------------------------------------------------------------------
# AugmentedDataset wrapper
# ---------------------------------------------------------------------------


class AugmentedDataset(Dataset):
    """Wraps any (waveform, clap_embed) dataset with on-the-fly augmentations.

    Each __getitem__ applies a random subset of transforms to the waveform.
    The CLAP embedding is passed through unchanged.

    The `multiplier` parameter scales the effective dataset size — each
    original sample appears `multiplier` times with different random
    augmentations per epoch.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        transforms: list | None = None,
        p_each: float = 0.5,
        multiplier: int = 1,
    ):
        self.base = base_dataset
        self.transforms = transforms or build_transforms()
        self.p_each = p_each
        self.multiplier = multiplier

    def __len__(self) -> int:
        return len(self.base) * self.multiplier

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_idx = idx % len(self.base)
        waveform, embed = self.base[real_idx]

        # Apply each transform independently with probability p_each
        for transform in self.transforms:
            if random.random() < self.p_each:
                waveform = transform(waveform)

        # Re-normalize to prevent clipping accumulation
        peak = waveform.abs().max()
        if peak > 1.0:
            waveform = waveform / peak

        # Ensure exact length
        n = waveform.shape[0]
        if n < CFG.n_samples:
            waveform = F.pad(waveform, (0, CFG.n_samples - n))
        elif n > CFG.n_samples:
            waveform = waveform[: CFG.n_samples]

        return waveform, embed
