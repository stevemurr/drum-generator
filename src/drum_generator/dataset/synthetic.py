"""
synthetic.py
------------
SyntheticDrumDataset: procedural drum sound generation via FM synthesis.

All drum types use frequency modulation as the core sound engine.
FM produces rich, evolving spectra from simple math — the same principle
behind classic drum machines (808, DX7). Spectral content is controlled
by modulation index and its envelope, not by post-filtering.

No external DSP libraries required — pure torch math.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from drum_generator.config import CFG
from drum_generator.dataset.caption import ClapEmbedder, build_synthetic_caption

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


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _normalize(waveform: torch.Tensor) -> torch.Tensor:
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    return waveform


def _waveshape(waveform: torch.Tensor, drive: float) -> torch.Tensor:
    """Soft-clip waveshaping via tanh."""
    if drive <= 1.0:
        return waveform
    return torch.tanh(waveform * drive) / torch.tanh(torch.tensor(drive))


def _adsr_envelope(
    n_samples: int, sr: int,
    attack_ms: float, hold_ms: float, decay_ms: float, sustain: float = 0.0,
) -> torch.Tensor:
    """Multi-stage envelope: attack → hold → decay → sustain level."""
    attack = int(attack_ms * sr / 1000)
    hold = int(hold_ms * sr / 1000)
    decay = int(decay_ms * sr / 1000)

    env = torch.zeros(n_samples)
    a_end = min(attack, n_samples)
    if a_end > 0:
        env[:a_end] = torch.linspace(0, 1, a_end)

    h_end = min(attack + hold, n_samples)
    env[a_end:h_end] = 1.0

    d_end = min(attack + hold + decay, n_samples)
    d_len = d_end - h_end
    if d_len > 0:
        t_d = torch.linspace(0, 1, d_len)
        env[h_end:d_end] = sustain + (1.0 - sustain) * torch.exp(-5.0 * t_d)

    env[d_end:] = sustain
    return env


# ---------------------------------------------------------------------------
# FM operators
# ---------------------------------------------------------------------------


def _fm_carrier(
    t: torch.Tensor, sr: int,
    freq: torch.Tensor | float,
    mod_signal: torch.Tensor | None = None,
    feedback: float = 0.0,
) -> torch.Tensor:
    """FM carrier with optional modulation input and self-feedback.

    Args:
        t: time vector (n_samples,)
        sr: sample rate
        freq: carrier frequency — scalar or (n_samples,) for pitch sweep
        mod_signal: modulator output to add to phase (n_samples,)
        feedback: self-modulation amount (0 = none)
    """
    n = len(t)
    if isinstance(freq, (int, float)):
        freq = torch.full((n,), freq)

    # Integrate frequency to get phase
    phase = 2 * torch.pi * torch.cumsum(freq / sr, dim=0)

    if mod_signal is not None:
        phase = phase + mod_signal

    if feedback > 0:
        # Simple one-sample feedback approximation
        out = torch.zeros(n)
        prev = 0.0
        for i in range(n):
            out[i] = torch.sin(phase[i] + feedback * prev)
            prev = out[i].item()
        return out

    return torch.sin(phase)


def _fm_pair(
    t: torch.Tensor, sr: int,
    carrier_freq: torch.Tensor | float,
    mod_freq: float,
    mod_depth_env: torch.Tensor,
    amp_env: torch.Tensor,
    feedback: float = 0.0,
) -> torch.Tensor:
    """Two-operator FM: modulator → carrier, with enveloped mod depth.

    The mod_depth_env controls spectral evolution over time.
    """
    n = len(t)
    # Modulator
    mod_phase = 2 * torch.pi * mod_freq * t
    modulator = mod_depth_env * torch.sin(mod_phase)

    # Carrier
    carrier = _fm_carrier(t, sr, carrier_freq, mod_signal=modulator, feedback=feedback)

    return carrier * amp_env


def _fm_noise(
    t: torch.Tensor, sr: int,
    base_freq: float,
    mod_ratio: float,
    mod_depth: float,
    amp_env: torch.Tensor,
) -> torch.Tensor:
    """FM-generated noise-like signal. High mod_depth + high mod_freq = noise."""
    mod_freq = base_freq * mod_ratio
    mod_phase = 2 * torch.pi * mod_freq * t
    carrier_phase = 2 * torch.pi * base_freq * t + mod_depth * torch.sin(mod_phase)
    return torch.sin(carrier_phase) * amp_env


def _pitch_sweep(
    n: int, sr: int, f_start: float, f_end: float, decay_rate: float
) -> torch.Tensor:
    """Exponential pitch sweep from f_start to f_end."""
    t = torch.linspace(0, n / sr, n)
    return f_end + (f_start - f_end) * torch.exp(-decay_rate * t)


# ---------------------------------------------------------------------------
# Characteristic-aware parameter builders
# ---------------------------------------------------------------------------


def _apply_kick_chars(chars: list[str], rng: np.random.RandomState) -> dict:
    p = {
        "carrier_start": rng.uniform(180, 300),
        "carrier_end": rng.uniform(35, 60),
        "pitch_decay": rng.uniform(8, 15),
        "mod_ratio": rng.uniform(1.5, 3.0),
        "mod_depth_peak": rng.uniform(4.0, 10.0),
        "mod_decay_ms": rng.uniform(10, 40),
        "feedback": rng.uniform(0.0, 0.3),
        "amp_attack_ms": rng.uniform(0.1, 1.0),
        "amp_hold_ms": rng.uniform(2, 8),
        "amp_decay_ms": rng.uniform(150, 350),
        "drive": rng.uniform(1.0, 1.5),
        "sub_mix": rng.uniform(0.0, 0.2),
        "sub_freq": rng.uniform(35, 55),
    }
    for c in chars:
        if c == "punchy":
            p["mod_depth_peak"] = rng.uniform(10.0, 18.0)
            p["mod_decay_ms"] = rng.uniform(5, 15)
            p["pitch_decay"] = rng.uniform(12, 20)
        elif c == "sub":
            p["carrier_end"] = rng.uniform(25, 40)
            p["sub_mix"] = rng.uniform(0.3, 0.6)
            p["amp_decay_ms"] = rng.uniform(300, 500)
            p["mod_depth_peak"] = rng.uniform(2.0, 5.0)
        elif c == "boomy":
            p["amp_decay_ms"] = rng.uniform(350, 550)
            p["carrier_end"] = rng.uniform(30, 50)
            p["feedback"] = rng.uniform(0.2, 0.5)
        elif c == "tight":
            p["amp_decay_ms"] = rng.uniform(60, 130)
            p["pitch_decay"] = rng.uniform(15, 25)
        elif c == "hard":
            p["mod_depth_peak"] = rng.uniform(12.0, 22.0)
            p["drive"] = rng.uniform(2.0, 4.0)
            p["mod_decay_ms"] = rng.uniform(3, 10)
        elif c == "soft":
            p["mod_depth_peak"] = rng.uniform(1.0, 3.0)
            p["drive"] = rng.uniform(1.0, 1.2)
            p["amp_attack_ms"] = rng.uniform(2.0, 5.0)
        elif c == "warm":
            p["feedback"] = rng.uniform(0.3, 0.6)
            p["carrier_start"] = rng.uniform(130, 200)
            p["mod_depth_peak"] = rng.uniform(3.0, 7.0)
        elif c == "dark":
            p["carrier_start"] = rng.uniform(100, 160)
            p["mod_depth_peak"] = rng.uniform(2.0, 5.0)
            p["mod_decay_ms"] = rng.uniform(5, 15)
        elif c == "distorted":
            p["drive"] = rng.uniform(3.0, 6.0)
            p["feedback"] = rng.uniform(0.4, 0.8)
        elif c == "clean":
            p["drive"] = 1.0
            p["feedback"] = rng.uniform(0.0, 0.1)
        elif c == "electronic":
            p["mod_ratio"] = rng.choice([2.0, 3.0, 4.0])
            p["drive"] = rng.uniform(1.5, 3.0)
        elif c == "acoustic":
            p["mod_ratio"] = rng.uniform(1.2, 2.5)
            p["feedback"] = rng.uniform(0.1, 0.3)
            p["mod_depth_peak"] = rng.uniform(5.0, 12.0)
        elif c == "trap":
            p["carrier_end"] = rng.uniform(25, 40)
            p["sub_mix"] = rng.uniform(0.3, 0.5)
            p["amp_decay_ms"] = rng.uniform(400, 600)
            p["drive"] = rng.uniform(1.5, 2.5)
        elif c == "house":
            p["carrier_start"] = rng.uniform(160, 230)
            p["amp_decay_ms"] = rng.uniform(120, 250)
            p["drive"] = rng.uniform(1.5, 2.5)
    return p


def _apply_snare_chars(chars: list[str], rng: np.random.RandomState) -> dict:
    p = {
        "body_freq": rng.uniform(150, 250),
        "body_mod_ratio": rng.uniform(1.3, 2.5),
        "body_mod_depth": rng.uniform(3.0, 8.0),
        "body_decay_ms": rng.uniform(40, 100),
        "noise_freq": rng.uniform(800, 2000),
        "noise_mod_ratio": rng.uniform(7.0, 15.0),
        "noise_mod_depth": rng.uniform(10.0, 25.0),
        "noise_decay_ms": rng.uniform(60, 150),
        "body_mix": rng.uniform(0.35, 0.6),
        "attack_ms": rng.uniform(0.3, 1.5),
        "drive": rng.uniform(1.0, 1.5),
    }
    for c in chars:
        if c == "snappy":
            p["noise_mod_depth"] = rng.uniform(20.0, 35.0)
            p["noise_decay_ms"] = rng.uniform(100, 200)
        elif c == "crisp":
            p["attack_ms"] = rng.uniform(0.1, 0.5)
            p["noise_mod_ratio"] = rng.uniform(12.0, 20.0)
        elif c == "tight":
            p["body_decay_ms"] = rng.uniform(20, 50)
            p["noise_decay_ms"] = rng.uniform(30, 70)
        elif c == "bright":
            p["noise_mod_ratio"] = rng.uniform(12.0, 20.0)
            p["noise_freq"] = rng.uniform(1500, 3000)
        elif c == "warm":
            p["noise_mod_ratio"] = rng.uniform(5.0, 9.0)
            p["body_mod_depth"] = rng.uniform(2.0, 5.0)
            p["drive"] = rng.uniform(1.5, 2.5)
        elif c == "dry":
            p["noise_decay_ms"] = rng.uniform(30, 60)
        elif c == "wet":
            p["noise_decay_ms"] = rng.uniform(150, 300)
        elif c == "compressed":
            p["drive"] = rng.uniform(2.0, 3.5)
        elif c == "distorted":
            p["drive"] = rng.uniform(3.0, 5.0)
            p["body_mod_depth"] = rng.uniform(8.0, 15.0)
        elif c == "electronic":
            p["body_mod_ratio"] = rng.choice([2.0, 3.0])
            p["drive"] = rng.uniform(1.5, 3.0)
        elif c == "acoustic":
            p["body_mod_ratio"] = rng.uniform(1.1, 1.8)
            p["noise_mod_depth"] = rng.uniform(15.0, 30.0)
            p["body_mix"] = rng.uniform(0.4, 0.6)
        elif c == "vintage":
            p["noise_mod_ratio"] = rng.uniform(6.0, 10.0)
            p["drive"] = rng.uniform(1.5, 2.5)
    return p


def _apply_hihat_chars(chars: list[str], rng: np.random.RandomState, open_hat: bool) -> dict:
    p = {
        "n_carriers": rng.randint(4, 8),
        "base_freq": rng.uniform(250, 500),
        "mod_depth": rng.uniform(8.0, 18.0),
        "decay_ms": rng.uniform(150, 600) if open_hat else rng.uniform(15, 70),
        "attack_ms": rng.uniform(0.1, 0.8),
        "drive": rng.uniform(1.0, 1.5),
    }
    for c in chars:
        if c == "crisp":
            p["mod_depth"] = rng.uniform(15.0, 25.0)
            p["attack_ms"] = rng.uniform(0.05, 0.3)
        elif c == "bright":
            p["base_freq"] = rng.uniform(400, 700)
            p["mod_depth"] = rng.uniform(12.0, 22.0)
        elif c == "dark":
            p["base_freq"] = rng.uniform(150, 300)
            p["mod_depth"] = rng.uniform(5.0, 10.0)
        elif c == "electronic":
            p["mod_depth"] = rng.uniform(15.0, 30.0)
            p["drive"] = rng.uniform(1.5, 2.5)
        elif c == "acoustic":
            p["n_carriers"] = rng.randint(6, 10)
            p["mod_depth"] = rng.uniform(6.0, 12.0)
        elif c == "distorted":
            p["drive"] = rng.uniform(2.5, 4.0)
            p["mod_depth"] = rng.uniform(20.0, 35.0)
        elif c == "clean":
            p["drive"] = 1.0
            p["mod_depth"] = rng.uniform(6.0, 12.0)
        elif c == "tight":
            if open_hat:
                p["decay_ms"] = rng.uniform(80, 180)
            else:
                p["decay_ms"] = rng.uniform(8, 30)
    return p


def _apply_clap_chars(chars: list[str], rng: np.random.RandomState) -> dict:
    p = {
        "n_bursts": rng.randint(3, 6),
        "burst_len_ms": rng.uniform(5, 15),
        "spacing_ms": rng.uniform(10, 30),
        "fm_freq": rng.uniform(600, 1500),
        "fm_mod_ratio": rng.uniform(5.0, 12.0),
        "fm_mod_depth": rng.uniform(10.0, 25.0),
        "tail_decay_ms": rng.uniform(60, 160),
        "drive": rng.uniform(1.0, 1.5),
    }
    for c in chars:
        if c == "tight":
            p["tail_decay_ms"] = rng.uniform(30, 70)
        elif c == "roomy":
            p["tail_decay_ms"] = rng.uniform(200, 400)
        elif c == "dry":
            p["tail_decay_ms"] = rng.uniform(30, 60)
        elif c == "wet":
            p["tail_decay_ms"] = rng.uniform(200, 400)
        elif c == "electronic":
            p["fm_mod_depth"] = rng.uniform(18.0, 35.0)
            p["drive"] = rng.uniform(1.5, 2.5)
        elif c == "vintage":
            p["fm_mod_ratio"] = rng.uniform(4.0, 8.0)
            p["fm_mod_depth"] = rng.uniform(8.0, 15.0)
            p["drive"] = rng.uniform(1.3, 2.0)
        elif c == "compressed":
            p["drive"] = rng.uniform(2.0, 3.5)
        elif c == "house":
            p["tail_decay_ms"] = rng.uniform(120, 250)
            p["n_bursts"] = rng.randint(4, 7)
        elif c == "trap":
            p["tail_decay_ms"] = rng.uniform(40, 80)
    return p


def _apply_tom_chars(chars: list[str], rng: np.random.RandomState) -> dict:
    p = {
        "carrier_start": rng.uniform(150, 320),
        "carrier_end": rng.uniform(60, 150),
        "pitch_decay": rng.uniform(3, 10),
        "mod_ratio": rng.uniform(1.3, 2.8),
        "mod_depth_peak": rng.uniform(3.0, 10.0),
        "mod_decay_ms": rng.uniform(20, 60),
        "feedback": rng.uniform(0.0, 0.3),
        "amp_attack_ms": rng.uniform(0.3, 2.0),
        "amp_decay_ms": rng.uniform(150, 400),
        "drive": rng.uniform(1.0, 1.5),
        "stick_mix": rng.uniform(0.0, 0.15),
    }
    for c in chars:
        if c == "warm":
            p["feedback"] = rng.uniform(0.3, 0.5)
            p["carrier_start"] = rng.uniform(100, 200)
            p["mod_depth_peak"] = rng.uniform(2.0, 5.0)
        elif c == "boomy":
            p["amp_decay_ms"] = rng.uniform(350, 550)
            p["carrier_end"] = rng.uniform(50, 80)
        elif c == "punchy":
            p["mod_depth_peak"] = rng.uniform(8.0, 16.0)
            p["mod_decay_ms"] = rng.uniform(8, 20)
            p["pitch_decay"] = rng.uniform(8, 15)
        elif c == "tight":
            p["amp_decay_ms"] = rng.uniform(80, 160)
        elif c == "acoustic":
            p["mod_ratio"] = rng.uniform(1.1, 1.8)
            p["stick_mix"] = rng.uniform(0.1, 0.25)
        elif c == "electronic":
            p["mod_ratio"] = rng.choice([2.0, 3.0])
            p["drive"] = rng.uniform(1.5, 2.5)
        elif c == "dark":
            p["carrier_start"] = rng.uniform(80, 150)
            p["mod_depth_peak"] = rng.uniform(2.0, 5.0)
        elif c == "bright":
            p["carrier_start"] = rng.uniform(250, 400)
            p["mod_depth_peak"] = rng.uniform(8.0, 14.0)
    return p


def _apply_rimshot_chars(chars: list[str], rng: np.random.RandomState) -> dict:
    p = {
        "click_freq": rng.uniform(1500, 4000),
        "click_mod_ratio": rng.uniform(3.0, 7.0),
        "click_mod_depth": rng.uniform(12.0, 25.0),
        "click_decay_ms": rng.uniform(5, 20),
        "ring_freq": rng.uniform(2000, 5000),
        "ring_mod_ratio": rng.uniform(1.5, 4.0),
        "ring_mod_depth": rng.uniform(4.0, 10.0),
        "ring_decay_ms": rng.uniform(30, 80),
        "ring_gain": rng.uniform(0.3, 0.6),
        "drive": rng.uniform(1.0, 1.5),
    }
    for c in chars:
        if c == "crisp":
            p["click_freq"] = rng.uniform(3000, 6000)
            p["click_mod_depth"] = rng.uniform(18.0, 30.0)
        elif c == "snappy":
            p["click_decay_ms"] = rng.uniform(2, 8)
            p["click_mod_depth"] = rng.uniform(15.0, 28.0)
        elif c == "bright":
            p["ring_freq"] = rng.uniform(4000, 8000)
            p["click_freq"] = rng.uniform(3000, 6000)
        elif c == "tight":
            p["ring_decay_ms"] = rng.uniform(15, 35)
        elif c == "hard":
            p["drive"] = rng.uniform(2.0, 3.5)
            p["click_mod_depth"] = rng.uniform(20.0, 35.0)
        elif c == "clean":
            p["drive"] = 1.0
        elif c == "acoustic":
            p["click_mod_ratio"] = rng.uniform(2.0, 4.0)
            p["ring_mod_ratio"] = rng.uniform(1.2, 2.5)
    return p


def _apply_cymbal_chars(chars: list[str], rng: np.random.RandomState) -> dict:
    p = {
        "n_carriers": rng.randint(6, 12),
        "base_freq": rng.uniform(200, 500),
        "mod_depth": rng.uniform(8.0, 18.0),
        "decay_ms": rng.uniform(400, 1000),
        "attack_ms": rng.uniform(0.5, 2.0),
        "drive": rng.uniform(1.0, 1.3),
    }
    for c in chars:
        if c == "bright":
            p["base_freq"] = rng.uniform(400, 700)
            p["mod_depth"] = rng.uniform(14.0, 22.0)
        elif c == "dark":
            p["base_freq"] = rng.uniform(150, 300)
            p["mod_depth"] = rng.uniform(5.0, 10.0)
        elif c == "crisp":
            p["mod_depth"] = rng.uniform(16.0, 25.0)
            p["attack_ms"] = rng.uniform(0.1, 0.5)
        elif c == "warm":
            p["mod_depth"] = rng.uniform(5.0, 10.0)
            p["drive"] = rng.uniform(1.3, 2.0)
        elif c == "clean":
            p["drive"] = 1.0
            p["mod_depth"] = rng.uniform(6.0, 12.0)
        elif c == "distorted":
            p["drive"] = rng.uniform(2.5, 4.0)
            p["mod_depth"] = rng.uniform(18.0, 30.0)
    return p


# Inharmonic frequency ratios for metallic FM percussion
_METAL_RATIOS = [1.0, 1.483, 1.932, 2.546, 2.732, 3.155, 3.671,
                 4.107, 4.581, 5.202, 5.834, 6.408]


# ---------------------------------------------------------------------------
# FM synthesis functions — one per drum type
# ---------------------------------------------------------------------------


def _synth_kick(t: torch.Tensor, sr: int, rng: np.random.RandomState, chars: list[str]) -> torch.Tensor:
    """FM kick: pitch-swept carrier with decaying mod index + feedback."""
    p = _apply_kick_chars(chars, rng)
    n = len(t)

    # Pitch sweep
    carrier_freq = _pitch_sweep(n, sr, p["carrier_start"], p["carrier_end"], p["pitch_decay"])

    # Modulator with decaying depth → bright attack, pure sub tail
    mod_freq = p["carrier_end"] * p["mod_ratio"]
    mod_env = _adsr_envelope(n, sr, 0.1, 0.5, p["mod_decay_ms"]) * p["mod_depth_peak"]
    amp_env = _adsr_envelope(n, sr, p["amp_attack_ms"], p["amp_hold_ms"], p["amp_decay_ms"])

    body = _fm_pair(t, sr, carrier_freq, mod_freq, mod_env, amp_env, feedback=p["feedback"])
    body = _waveshape(body, p["drive"])

    # Sub layer (pure sine, no FM)
    if p["sub_mix"] > 0.05:
        sub_env = _adsr_envelope(n, sr, 2.0, 5.0, p["amp_decay_ms"] * 1.3)
        sub = torch.sin(2 * torch.pi * p["sub_freq"] * t) * sub_env * p["sub_mix"]
        body = body + sub

    return body


def _synth_snare(t: torch.Tensor, sr: int, rng: np.random.RandomState, chars: list[str]) -> torch.Tensor:
    """FM snare: tonal FM body + FM-generated noise (high mod index = noise)."""
    p = _apply_snare_chars(chars, rng)
    n = len(t)

    # Tonal body: 2-op FM
    body_mod_env = _adsr_envelope(n, sr, 0.3, 1.0, p["body_decay_ms"]) * p["body_mod_depth"]
    body_amp_env = _adsr_envelope(n, sr, p["attack_ms"], 1.0, p["body_decay_ms"])
    body_mod_freq = p["body_freq"] * p["body_mod_ratio"]
    body = _fm_pair(t, sr, p["body_freq"], body_mod_freq, body_mod_env, body_amp_env)

    # Noise component: FM with very high mod index → noise-like spectrum
    noise_amp_env = _adsr_envelope(n, sr, p["attack_ms"], 0.5, p["noise_decay_ms"])
    noise = _fm_noise(t, sr, p["noise_freq"], p["noise_mod_ratio"], p["noise_mod_depth"], noise_amp_env)

    result = p["body_mix"] * body + (1 - p["body_mix"]) * noise
    result = _waveshape(result, p["drive"])
    return result


def _synth_hihat(
    t: torch.Tensor, sr: int, rng: np.random.RandomState, chars: list[str], open_hat: bool = False,
) -> torch.Tensor:
    """FM hi-hat: multiple carriers at inharmonic ratios, each FM-modulated."""
    p = _apply_hihat_chars(chars, rng, open_hat)
    n = len(t)

    result = torch.zeros(n)
    n_carriers = min(p["n_carriers"], len(_METAL_RATIOS))

    for i in range(n_carriers):
        freq = p["base_freq"] * _METAL_RATIOS[i]
        if freq > sr / 2:
            continue
        # Each carrier gets its own mod ratio (inharmonic)
        mod_ratio = _METAL_RATIOS[(i * 3 + 1) % len(_METAL_RATIOS)]
        mod_freq = freq * mod_ratio

        # Per-carrier variation in depth and decay
        depth = p["mod_depth"] * rng.uniform(0.6, 1.4)
        carrier_decay = p["decay_ms"] * rng.uniform(0.7, 1.3)
        amp = rng.uniform(0.3, 1.0) / (1 + i * 0.25)

        mod_env = _adsr_envelope(n, sr, p["attack_ms"], 0.2, carrier_decay) * depth
        amp_env = _adsr_envelope(n, sr, p["attack_ms"], 0.2, carrier_decay)

        carrier = _fm_pair(t, sr, freq, mod_freq, mod_env, amp_env)
        result = result + amp * carrier

    result = _waveshape(result, p["drive"])
    return result


def _synth_clap(t: torch.Tensor, sr: int, rng: np.random.RandomState, chars: list[str]) -> torch.Tensor:
    """FM clap: staggered FM noise bursts."""
    p = _apply_clap_chars(chars, rng)
    n = len(t)

    burst_samples = int(p["burst_len_ms"] * sr / 1000)
    spacing_samples = int(p["spacing_ms"] * sr / 1000)

    result = torch.zeros(n)
    for i in range(p["n_bursts"]):
        offset = i * spacing_samples
        if offset + burst_samples >= n:
            break
        burst_t = torch.linspace(0, burst_samples / sr, burst_samples)
        burst_env = _adsr_envelope(burst_samples, sr, 0.1, 0.3, p["burst_len_ms"] * 0.7)
        burst = _fm_noise(burst_t, sr, p["fm_freq"], p["fm_mod_ratio"], p["fm_mod_depth"], burst_env)
        result[offset:offset + burst_samples] = result[offset:offset + burst_samples] + burst

    # Overall tail envelope
    tail_env = _adsr_envelope(n, sr, 0.5, 1.0, p["tail_decay_ms"])
    result = result * tail_env
    result = _waveshape(result, p["drive"])
    return result


def _synth_tom(t: torch.Tensor, sr: int, rng: np.random.RandomState, chars: list[str]) -> torch.Tensor:
    """FM tom: pitch-swept FM like kick, higher frequencies, longer sustain."""
    p = _apply_tom_chars(chars, rng)
    n = len(t)

    carrier_freq = _pitch_sweep(n, sr, p["carrier_start"], p["carrier_end"], p["pitch_decay"])
    mod_freq = p["carrier_end"] * p["mod_ratio"]

    mod_env = _adsr_envelope(n, sr, 0.3, 1.0, p["mod_decay_ms"]) * p["mod_depth_peak"]
    amp_env = _adsr_envelope(n, sr, p["amp_attack_ms"], 3.0, p["amp_decay_ms"])

    body = _fm_pair(t, sr, carrier_freq, mod_freq, mod_env, amp_env, feedback=p["feedback"])
    body = _waveshape(body, p["drive"])

    # Stick transient: short FM burst
    if p["stick_mix"] > 0.02:
        stick_env = _adsr_envelope(n, sr, 0.1, 0.2, 8.0)
        stick = _fm_noise(t, sr, 3000, 5.0, 15.0, stick_env) * p["stick_mix"]
        body = body + stick

    return body


def _synth_rimshot(t: torch.Tensor, sr: int, rng: np.random.RandomState, chars: list[str]) -> torch.Tensor:
    """FM rimshot: sharp FM click + resonant FM ring."""
    p = _apply_rimshot_chars(chars, rng)
    n = len(t)

    # Click: high mod depth FM burst
    click_mod_env = _adsr_envelope(n, sr, 0.1, 0.2, p["click_decay_ms"]) * p["click_mod_depth"]
    click_amp_env = _adsr_envelope(n, sr, 0.1, 0.2, p["click_decay_ms"])
    click_mod_freq = p["click_freq"] * p["click_mod_ratio"]
    click = _fm_pair(t, sr, p["click_freq"], click_mod_freq, click_mod_env, click_amp_env)

    # Ring: moderate mod depth, longer decay
    ring_mod_env = _adsr_envelope(n, sr, 0.2, 0.5, p["ring_decay_ms"]) * p["ring_mod_depth"]
    ring_amp_env = _adsr_envelope(n, sr, 0.2, 0.5, p["ring_decay_ms"])
    ring_mod_freq = p["ring_freq"] * p["ring_mod_ratio"]
    ring = _fm_pair(t, sr, p["ring_freq"], ring_mod_freq, ring_mod_env, ring_amp_env)

    result = click + p["ring_gain"] * ring
    result = _waveshape(result, p["drive"])
    return result


def _synth_cymbal(t: torch.Tensor, sr: int, rng: np.random.RandomState, chars: list[str]) -> torch.Tensor:
    """FM cymbal: dense inharmonic FM cluster with slow decay."""
    p = _apply_cymbal_chars(chars, rng)
    n = len(t)

    result = torch.zeros(n)
    n_carriers = min(p["n_carriers"], len(_METAL_RATIOS))

    for i in range(n_carriers):
        freq = p["base_freq"] * _METAL_RATIOS[i]
        if freq > sr / 2:
            continue
        mod_ratio = _METAL_RATIOS[(i * 5 + 2) % len(_METAL_RATIOS)]
        mod_freq = freq * mod_ratio

        depth = p["mod_depth"] * rng.uniform(0.5, 1.5)
        carrier_decay = p["decay_ms"] * rng.uniform(0.6, 1.4)
        amp = rng.uniform(0.3, 1.0) / (1 + i * 0.15)

        mod_env = _adsr_envelope(n, sr, p["attack_ms"], 0.5, carrier_decay) * depth
        amp_env = _adsr_envelope(n, sr, p["attack_ms"], 0.5, carrier_decay)

        carrier = _fm_pair(t, sr, freq, mod_freq, mod_env, amp_env)
        result = result + amp * carrier

    result = _waveshape(result, p["drive"])
    return result


_SYNTH_FUNCS = {
    "kick": _synth_kick,
    "snare": _synth_snare,
    "hihat_closed": lambda t, sr, rng, chars: _synth_hihat(t, sr, rng, chars, open_hat=False),
    "hihat_open": lambda t, sr, rng, chars: _synth_hihat(t, sr, rng, chars, open_hat=True),
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
    """Generates synthetic drum sounds programmatically via FM synthesis.

    Each __getitem__ call deterministically generates a sound based on
    the index (via seeded RNG), ensuring reproducibility. Drum types are
    assigned round-robin for even coverage.

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
        # Round-robin across drum types for even coverage
        rng = np.random.RandomState(seed)
        self.params: list[dict] = []
        for i in range(size):
            drum_type = DRUM_TYPES[i % len(DRUM_TYPES)]
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
        waveform = synth_fn(t, self.sr, rng, params["characteristics"])

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
