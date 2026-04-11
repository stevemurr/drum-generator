"""
synthetic.py
------------
SyntheticDrumDataset: procedural drum sound generation via basic DSP.

Generates kick, snare, hi-hat, clap, tom, rimshot, and cymbal sounds
using sine sweeps, filtered noise, waveshaping, multi-stage envelopes,
and inharmonic partial clusters. All synthesis is pure torch/torchaudio
math — no external DSP libraries required.
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
# DSP helpers
# ---------------------------------------------------------------------------


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


def _biquad_peak(
    waveform: torch.Tensor, sr: int, freq: float, gain_db: float, q: float = 1.0
) -> torch.Tensor:
    """Peaking EQ filter."""
    import torchaudio.functional as AF

    return AF.equalizer_biquad(waveform.unsqueeze(0), sr, freq, gain_db, q).squeeze(0)


def _normalize(waveform: torch.Tensor) -> torch.Tensor:
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    return waveform


def _adsr_envelope(
    n_samples: int, sr: int,
    attack_ms: float, hold_ms: float, decay_ms: float, sustain: float = 0.0,
) -> torch.Tensor:
    """Multi-stage envelope: attack → hold → decay → sustain level."""
    attack = int(attack_ms * sr / 1000)
    hold = int(hold_ms * sr / 1000)
    decay = int(decay_ms * sr / 1000)

    env = torch.zeros(n_samples)

    # Attack: 0 → 1
    a_end = min(attack, n_samples)
    if a_end > 0:
        env[:a_end] = torch.linspace(0, 1, a_end)

    # Hold: 1
    h_end = min(attack + hold, n_samples)
    env[a_end:h_end] = 1.0

    # Decay: 1 → sustain (exponential)
    d_end = min(attack + hold + decay, n_samples)
    d_len = d_end - h_end
    if d_len > 0:
        t_d = torch.linspace(0, 1, d_len)
        env[h_end:d_end] = sustain + (1.0 - sustain) * torch.exp(-5.0 * t_d)

    # Sustain
    env[d_end:] = sustain

    return env


def _waveshape(waveform: torch.Tensor, drive: float) -> torch.Tensor:
    """Soft-clip waveshaping via tanh. drive > 1 adds harmonics."""
    if drive <= 1.0:
        return waveform
    return torch.tanh(waveform * drive) / torch.tanh(torch.tensor(drive))


def _comb_filter(waveform: torch.Tensor, delay_samples: int, feedback: float) -> torch.Tensor:
    """Simple feedforward comb filter for metallic/resonant effects."""
    out = waveform.clone()
    if delay_samples >= len(waveform):
        return out
    out[delay_samples:] = out[delay_samples:] + feedback * waveform[:-delay_samples]
    return out


def _simple_reverb_tail(n_samples: int, sr: int, decay_time: float, mix: float) -> torch.Tensor:
    """Generate a synthetic reverb impulse response for convolution."""
    ir_len = int(decay_time * sr)
    if ir_len < 2:
        return torch.zeros(n_samples)
    t = torch.linspace(0, decay_time, ir_len)
    ir = torch.randn(ir_len) * torch.exp(-5.0 * t / decay_time)
    ir = ir / ir.abs().sum()
    return ir


def _convolve(signal: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
    """Convolve signal with impulse response, keeping original length."""
    pad_len = len(ir) - 1
    out = torch.nn.functional.conv1d(
        signal.view(1, 1, -1),
        ir.flip(0).view(1, 1, -1),
        padding=pad_len,
    ).squeeze()
    return out[: len(signal)]


# ---------------------------------------------------------------------------
# Characteristic-aware parameter modifiers
# ---------------------------------------------------------------------------


def _apply_kick_chars(chars: list[str], rng: np.random.RandomState) -> dict:
    """Map characteristic tags to synthesis parameters for kicks."""
    p = {
        "f0": rng.uniform(150, 300),
        "f1": rng.uniform(35, 60),
        "pitch_decay": rng.uniform(5, 12),
        "attack_ms": rng.uniform(0.5, 3.0),
        "hold_ms": rng.uniform(2, 10),
        "body_decay_ms": rng.uniform(80, 250),
        "drive": rng.uniform(1.0, 1.5),
        "click_gain": rng.uniform(0.1, 0.3),
        "sub_mix": rng.uniform(0.0, 0.3),
        "sub_freq": rng.uniform(35, 55),
    }

    for c in chars:
        if c == "punchy":
            p["attack_ms"] = rng.uniform(0.1, 1.0)
            p["click_gain"] = rng.uniform(0.3, 0.6)
            p["pitch_decay"] = rng.uniform(8, 15)
        elif c == "sub":
            p["f1"] = rng.uniform(25, 40)
            p["sub_mix"] = rng.uniform(0.4, 0.7)
            p["body_decay_ms"] = rng.uniform(200, 400)
        elif c == "boomy":
            p["f1"] = rng.uniform(30, 50)
            p["body_decay_ms"] = rng.uniform(250, 400)
            p["drive"] = rng.uniform(1.0, 1.3)
        elif c == "tight":
            p["body_decay_ms"] = rng.uniform(40, 100)
            p["pitch_decay"] = rng.uniform(10, 20)
        elif c == "hard":
            p["click_gain"] = rng.uniform(0.4, 0.7)
            p["drive"] = rng.uniform(2.0, 4.0)
            p["attack_ms"] = rng.uniform(0.1, 0.5)
        elif c == "soft":
            p["click_gain"] = rng.uniform(0.0, 0.1)
            p["drive"] = rng.uniform(1.0, 1.2)
            p["attack_ms"] = rng.uniform(2.0, 5.0)
        elif c == "warm":
            p["drive"] = rng.uniform(1.5, 2.5)
            p["f0"] = rng.uniform(120, 200)
        elif c == "dark":
            p["f0"] = rng.uniform(100, 180)
            p["f1"] = rng.uniform(25, 45)
        elif c == "distorted":
            p["drive"] = rng.uniform(3.0, 6.0)
        elif c == "clean":
            p["drive"] = 1.0
        elif c == "electronic":
            p["drive"] = rng.uniform(1.5, 3.0)
            p["click_gain"] = rng.uniform(0.05, 0.15)
        elif c == "acoustic":
            p["drive"] = rng.uniform(1.0, 1.3)
            p["click_gain"] = rng.uniform(0.2, 0.5)
            p["f0"] = rng.uniform(120, 200)
        elif c == "trap":
            p["f1"] = rng.uniform(25, 40)
            p["sub_mix"] = rng.uniform(0.4, 0.6)
            p["body_decay_ms"] = rng.uniform(300, 500)
            p["drive"] = rng.uniform(1.5, 2.5)
        elif c == "house":
            p["f0"] = rng.uniform(150, 220)
            p["body_decay_ms"] = rng.uniform(100, 200)
            p["drive"] = rng.uniform(1.5, 2.5)
    return p


def _apply_snare_chars(chars: list[str], rng: np.random.RandomState) -> dict:
    p = {
        "tone_freq": rng.uniform(150, 250),
        "tone_decay_ms": rng.uniform(30, 80),
        "noise_low": rng.uniform(1500, 3000),
        "noise_high": rng.uniform(6000, 10000),
        "noise_decay_ms": rng.uniform(50, 150),
        "mix": rng.uniform(0.3, 0.6),  # tone vs noise
        "attack_ms": rng.uniform(0.5, 2.0),
        "drive": rng.uniform(1.0, 1.5),
        "wire_resonances": 3,
    }

    for c in chars:
        if c == "snappy":
            p["noise_decay_ms"] = rng.uniform(80, 180)
            p["noise_high"] = rng.uniform(10000, 14000)
            p["wire_resonances"] = rng.randint(4, 7)
        elif c == "crisp":
            p["attack_ms"] = rng.uniform(0.1, 0.5)
            p["noise_high"] = rng.uniform(12000, 16000)
        elif c == "tight":
            p["tone_decay_ms"] = rng.uniform(15, 40)
            p["noise_decay_ms"] = rng.uniform(30, 70)
        elif c == "bright":
            p["noise_low"] = rng.uniform(3000, 5000)
            p["noise_high"] = rng.uniform(12000, 16000)
        elif c == "warm":
            p["noise_high"] = rng.uniform(4000, 7000)
            p["drive"] = rng.uniform(1.5, 2.5)
        elif c == "dry":
            p["noise_decay_ms"] = rng.uniform(30, 60)
        elif c == "wet":
            p["noise_decay_ms"] = rng.uniform(120, 250)
        elif c == "compressed":
            p["drive"] = rng.uniform(2.0, 3.5)
        elif c == "distorted":
            p["drive"] = rng.uniform(3.0, 5.0)
        elif c == "electronic":
            p["tone_freq"] = rng.uniform(180, 280)
            p["drive"] = rng.uniform(1.5, 3.0)
            p["wire_resonances"] = rng.randint(1, 3)
        elif c == "acoustic":
            p["drive"] = rng.uniform(1.0, 1.3)
            p["wire_resonances"] = rng.randint(4, 7)
            p["mix"] = rng.uniform(0.4, 0.6)
        elif c == "vintage":
            p["noise_high"] = rng.uniform(5000, 8000)
            p["drive"] = rng.uniform(1.5, 2.5)
    return p


def _apply_hihat_chars(chars: list[str], rng: np.random.RandomState, open_hat: bool) -> dict:
    p = {
        "n_partials": rng.randint(6, 10),
        "base_freq": rng.uniform(300, 600),
        "noise_cutoff": rng.uniform(6000, 12000),
        "noise_mix": rng.uniform(0.3, 0.5),
        "decay_ms": rng.uniform(100, 500) if open_hat else rng.uniform(15, 60),
        "attack_ms": rng.uniform(0.1, 1.0),
        "drive": rng.uniform(1.0, 1.3),
    }

    for c in chars:
        if c == "crisp":
            p["noise_cutoff"] = rng.uniform(10000, 15000)
            p["attack_ms"] = rng.uniform(0.05, 0.3)
        elif c == "bright":
            p["base_freq"] = rng.uniform(500, 800)
            p["noise_cutoff"] = rng.uniform(10000, 16000)
        elif c == "dark":
            p["base_freq"] = rng.uniform(200, 400)
            p["noise_cutoff"] = rng.uniform(4000, 7000)
        elif c == "electronic":
            p["noise_mix"] = rng.uniform(0.5, 0.7)
            p["drive"] = rng.uniform(1.5, 2.5)
        elif c == "acoustic":
            p["n_partials"] = rng.randint(8, 14)
            p["noise_mix"] = rng.uniform(0.2, 0.4)
        elif c == "distorted":
            p["drive"] = rng.uniform(2.5, 4.0)
        elif c == "clean":
            p["drive"] = 1.0
            p["noise_mix"] = rng.uniform(0.15, 0.3)
        elif c == "tight":
            if open_hat:
                p["decay_ms"] = rng.uniform(60, 150)
            else:
                p["decay_ms"] = rng.uniform(8, 25)
    return p


def _apply_clap_chars(chars: list[str], rng: np.random.RandomState) -> dict:
    p = {
        "n_bursts": rng.randint(3, 6),
        "burst_len_ms": rng.uniform(5, 15),
        "spacing_ms": rng.uniform(10, 30),
        "bp_low": rng.uniform(500, 1500),
        "bp_high": rng.uniform(3000, 8000),
        "tail_decay_ms": rng.uniform(60, 150),
        "reverb_mix": rng.uniform(0.1, 0.3),
        "reverb_time": rng.uniform(0.1, 0.3),
        "drive": rng.uniform(1.0, 1.5),
    }

    for c in chars:
        if c == "tight":
            p["tail_decay_ms"] = rng.uniform(30, 70)
            p["reverb_mix"] = rng.uniform(0.0, 0.1)
        elif c == "roomy":
            p["reverb_mix"] = rng.uniform(0.3, 0.5)
            p["reverb_time"] = rng.uniform(0.3, 0.6)
        elif c == "dry":
            p["reverb_mix"] = 0.0
            p["tail_decay_ms"] = rng.uniform(30, 60)
        elif c == "wet":
            p["reverb_mix"] = rng.uniform(0.3, 0.5)
            p["reverb_time"] = rng.uniform(0.2, 0.5)
        elif c == "electronic":
            p["bp_low"] = rng.uniform(800, 2000)
            p["drive"] = rng.uniform(1.5, 2.5)
        elif c == "vintage":
            p["bp_high"] = rng.uniform(4000, 6000)
            p["drive"] = rng.uniform(1.3, 2.0)
        elif c == "compressed":
            p["drive"] = rng.uniform(2.0, 3.5)
        elif c == "house":
            p["reverb_mix"] = rng.uniform(0.2, 0.4)
            p["n_bursts"] = rng.randint(4, 7)
        elif c == "trap":
            p["tail_decay_ms"] = rng.uniform(40, 80)
            p["reverb_mix"] = rng.uniform(0.05, 0.15)
    return p


def _apply_tom_chars(chars: list[str], rng: np.random.RandomState) -> dict:
    p = {
        "f0": rng.uniform(120, 280),
        "f1": rng.uniform(60, 140),
        "pitch_decay": rng.uniform(3, 8),
        "attack_ms": rng.uniform(0.5, 3.0),
        "body_decay_ms": rng.uniform(100, 300),
        "drive": rng.uniform(1.0, 1.5),
        "click_gain": rng.uniform(0.05, 0.2),
        "noise_mix": rng.uniform(0.0, 0.15),
    }

    for c in chars:
        if c == "warm":
            p["f0"] = rng.uniform(100, 180)
            p["drive"] = rng.uniform(1.5, 2.5)
        elif c == "boomy":
            p["body_decay_ms"] = rng.uniform(250, 450)
            p["f1"] = rng.uniform(50, 80)
        elif c == "punchy":
            p["attack_ms"] = rng.uniform(0.1, 1.0)
            p["click_gain"] = rng.uniform(0.2, 0.4)
            p["pitch_decay"] = rng.uniform(6, 12)
        elif c == "tight":
            p["body_decay_ms"] = rng.uniform(50, 120)
        elif c == "acoustic":
            p["noise_mix"] = rng.uniform(0.1, 0.25)
            p["drive"] = rng.uniform(1.0, 1.3)
        elif c == "electronic":
            p["drive"] = rng.uniform(1.5, 2.5)
            p["noise_mix"] = rng.uniform(0.0, 0.05)
        elif c == "dark":
            p["f0"] = rng.uniform(80, 150)
            p["f1"] = rng.uniform(40, 70)
        elif c == "bright":
            p["f0"] = rng.uniform(200, 350)
            p["click_gain"] = rng.uniform(0.15, 0.3)
    return p


def _apply_rimshot_chars(chars: list[str], rng: np.random.RandomState) -> dict:
    p = {
        "click_freq": rng.uniform(2000, 5000),
        "click_decay_ms": rng.uniform(5, 20),
        "ring_freq": rng.uniform(3000, 7000),
        "ring_decay_ms": rng.uniform(30, 80),
        "ring_gain": rng.uniform(0.3, 0.6),
        "drive": rng.uniform(1.0, 1.5),
        "n_harmonics": 2,
    }

    for c in chars:
        if c == "crisp":
            p["click_freq"] = rng.uniform(4000, 7000)
            p["click_decay_ms"] = rng.uniform(3, 10)
        elif c == "snappy":
            p["click_decay_ms"] = rng.uniform(2, 8)
            p["ring_decay_ms"] = rng.uniform(20, 50)
        elif c == "bright":
            p["click_freq"] = rng.uniform(4000, 8000)
            p["ring_freq"] = rng.uniform(5000, 10000)
        elif c == "tight":
            p["ring_decay_ms"] = rng.uniform(15, 35)
        elif c == "hard":
            p["drive"] = rng.uniform(2.0, 3.5)
        elif c == "clean":
            p["drive"] = 1.0
        elif c == "acoustic":
            p["n_harmonics"] = rng.randint(3, 5)
    return p


def _apply_cymbal_chars(chars: list[str], rng: np.random.RandomState) -> dict:
    p = {
        "n_partials": rng.randint(8, 16),
        "base_freq": rng.uniform(300, 600),
        "decay_ms": rng.uniform(300, 800),
        "noise_mix": rng.uniform(0.2, 0.4),
        "hp_cutoff": rng.uniform(3000, 6000),
        "drive": rng.uniform(1.0, 1.3),
    }

    for c in chars:
        if c == "bright":
            p["base_freq"] = rng.uniform(500, 900)
            p["hp_cutoff"] = rng.uniform(5000, 8000)
        elif c == "dark":
            p["base_freq"] = rng.uniform(200, 400)
            p["hp_cutoff"] = rng.uniform(1500, 3500)
        elif c == "crisp":
            p["hp_cutoff"] = rng.uniform(6000, 10000)
        elif c == "warm":
            p["hp_cutoff"] = rng.uniform(2000, 4000)
            p["drive"] = rng.uniform(1.3, 2.0)
        elif c == "clean":
            p["drive"] = 1.0
            p["noise_mix"] = rng.uniform(0.1, 0.2)
        elif c == "distorted":
            p["drive"] = rng.uniform(2.5, 4.0)
    return p


# ---------------------------------------------------------------------------
# Inharmonic partial ratios for metallic percussion
# Based on physical cymbal/bell models
# ---------------------------------------------------------------------------

_CYMBAL_RATIOS = [1.0, 1.483, 1.932, 2.328, 2.732, 3.155, 3.583, 3.890,
                  4.414, 4.890, 5.321, 5.862, 6.305, 6.842, 7.384]


# ---------------------------------------------------------------------------
# Synthesis functions — one per drum type
# ---------------------------------------------------------------------------


def _synth_kick(t: torch.Tensor, sr: int, rng: np.random.RandomState, chars: list[str]) -> torch.Tensor:
    """Pitch-swept sine with ADSR envelope, waveshaping, and optional sub layer."""
    p = _apply_kick_chars(chars, rng)
    n = len(t)

    # Pitch sweep
    freq_sweep = p["f1"] + (p["f0"] - p["f1"]) * torch.exp(-p["pitch_decay"] * t)
    phase = 2 * torch.pi * torch.cumsum(freq_sweep / sr, dim=0)
    body = torch.sin(phase)

    # ADSR envelope
    env = _adsr_envelope(n, sr, p["attack_ms"], p["hold_ms"], p["body_decay_ms"])
    body = body * env

    # Waveshaping for warmth/harmonics
    body = _waveshape(body, p["drive"])

    # Transient: shaped impulse (not random noise)
    click_len = int(0.003 * sr)
    click_t = torch.linspace(0, 1, click_len)
    click = torch.sin(2 * torch.pi * 3000 * click_t / sr * click_len) * (1 - click_t) ** 2
    click = _waveshape(click, max(p["drive"], 2.0))
    body[:click_len] = body[:click_len] + click * p["click_gain"]

    # Sub layer
    if p["sub_mix"] > 0.05:
        sub_env = _adsr_envelope(n, sr, 2.0, 5.0, p["body_decay_ms"] * 1.5)
        sub = torch.sin(2 * torch.pi * p["sub_freq"] * t) * sub_env * p["sub_mix"]
        body = body + sub

    return body


def _synth_snare(t: torch.Tensor, sr: int, rng: np.random.RandomState, chars: list[str]) -> torch.Tensor:
    """Damped sine body + resonant snare wire noise + waveshaping."""
    p = _apply_snare_chars(chars, rng)
    n = len(t)

    # Tonal body with ADSR
    tone_env = _adsr_envelope(n, sr, p["attack_ms"], 1.0, p["tone_decay_ms"])
    tone = torch.sin(2 * torch.pi * p["tone_freq"] * t) * tone_env
    tone = _waveshape(tone, p["drive"])

    # Noise body
    noise = torch.randn(n)

    # Snare wire resonances: multiple comb filters at inharmonic intervals
    wire_noise = noise.clone()
    for i in range(p["wire_resonances"]):
        delay = int(sr / rng.uniform(1500, 6000))
        fb = rng.uniform(0.2, 0.5)
        wire_noise = _comb_filter(wire_noise, delay, fb)

    wire_noise = _biquad_bandpass(wire_noise, sr, p["noise_low"], p["noise_high"])

    noise_env = _adsr_envelope(n, sr, 0.5, 0.5, p["noise_decay_ms"])
    wire_noise = wire_noise * noise_env

    return p["mix"] * tone + (1 - p["mix"]) * wire_noise


def _synth_hihat(
    t: torch.Tensor, sr: int, rng: np.random.RandomState, chars: list[str], open_hat: bool = False
) -> torch.Tensor:
    """Inharmonic metallic partials + filtered noise, with ADSR."""
    p = _apply_hihat_chars(chars, rng, open_hat)
    n = len(t)

    # Metallic partials from physical cymbal ratios
    partials = torch.zeros(n)
    n_partials = min(p["n_partials"], len(_CYMBAL_RATIOS))
    for i in range(n_partials):
        freq = p["base_freq"] * _CYMBAL_RATIOS[i]
        if freq > sr / 2:
            continue
        # Each partial has slightly different decay
        partial_decay = p["decay_ms"] * rng.uniform(0.6, 1.4)
        partial_env = _adsr_envelope(n, sr, p["attack_ms"], 0.2, partial_decay)
        amp = rng.uniform(0.3, 1.0) / (1 + i * 0.3)  # higher partials quieter
        partials = partials + amp * torch.sin(2 * torch.pi * freq * t) * partial_env

    # Filtered noise component
    noise = torch.randn(n)
    noise = _biquad_highpass(noise, sr, p["noise_cutoff"])
    noise_env = _adsr_envelope(n, sr, p["attack_ms"], 0.1, p["decay_ms"] * 0.7)
    noise = noise * noise_env

    # Mix partials and noise
    result = (1 - p["noise_mix"]) * partials + p["noise_mix"] * noise
    result = _waveshape(result, p["drive"])

    return result


def _synth_clap(t: torch.Tensor, sr: int, rng: np.random.RandomState, chars: list[str]) -> torch.Tensor:
    """Multiple noise bursts with flamming, reverb tail, and waveshaping."""
    p = _apply_clap_chars(chars, rng)
    n = len(t)

    burst_len = int(p["burst_len_ms"] * sr / 1000)
    spacing = int(p["spacing_ms"] * sr / 1000)

    result = torch.zeros(n)
    for i in range(p["n_bursts"]):
        offset = i * spacing
        if offset + burst_len >= n:
            break
        # Shaped burst with fast attack, fast decay
        burst_env = _adsr_envelope(burst_len, sr, 0.1, 0.5, p["burst_len_ms"] * 0.6)
        burst = torch.randn(burst_len) * burst_env
        result[offset : offset + burst_len] = result[offset : offset + burst_len] + burst

    # Bandpass filter
    result = _biquad_bandpass(result, sr, p["bp_low"], p["bp_high"])

    # Overall decay envelope
    tail_env = _adsr_envelope(n, sr, 0.5, 1.0, p["tail_decay_ms"])
    result = result * tail_env

    # Waveshaping
    result = _waveshape(result, p["drive"])

    # Reverb tail
    if p["reverb_mix"] > 0.01:
        ir = _simple_reverb_tail(n, sr, p["reverb_time"], p["reverb_mix"])
        wet = _convolve(result, ir)
        result = (1 - p["reverb_mix"]) * result + p["reverb_mix"] * wet

    return result


def _synth_tom(t: torch.Tensor, sr: int, rng: np.random.RandomState, chars: list[str]) -> torch.Tensor:
    """Pitch-swept sine with ADSR, waveshaping, and optional noise layer."""
    p = _apply_tom_chars(chars, rng)
    n = len(t)

    # Pitch sweep
    freq_sweep = p["f1"] + (p["f0"] - p["f1"]) * torch.exp(-p["pitch_decay"] * t)
    phase = 2 * torch.pi * torch.cumsum(freq_sweep / sr, dim=0)
    body = torch.sin(phase)

    # ADSR envelope
    env = _adsr_envelope(n, sr, p["attack_ms"], 3.0, p["body_decay_ms"])
    body = body * env
    body = _waveshape(body, p["drive"])

    # Transient click
    click_len = int(0.003 * sr)
    if click_len > 0 and p["click_gain"] > 0.01:
        click_env = torch.linspace(1, 0, click_len) ** 2
        click = torch.randn(click_len) * click_env
        body[:click_len] = body[:click_len] + click * p["click_gain"]

    # Optional stick noise
    if p["noise_mix"] > 0.02:
        noise = torch.randn(n)
        noise = _biquad_bandpass(noise, sr, 2000, 8000)
        noise_env = _adsr_envelope(n, sr, 0.3, 0.5, 30)
        body = body + noise * noise_env * p["noise_mix"]

    return body


def _synth_rimshot(t: torch.Tensor, sr: int, rng: np.random.RandomState, chars: list[str]) -> torch.Tensor:
    """Sharp click + metallic harmonics with ADSR."""
    p = _apply_rimshot_chars(chars, rng)
    n = len(t)

    # Click with ADSR
    click_env = _adsr_envelope(n, sr, 0.1, 0.2, p["click_decay_ms"])
    click = torch.sin(2 * torch.pi * p["click_freq"] * t) * click_env

    # Metallic ring with multiple harmonics
    ring = torch.zeros(n)
    ring_env = _adsr_envelope(n, sr, 0.2, 0.5, p["ring_decay_ms"])
    for i in range(p["n_harmonics"]):
        harm_freq = p["ring_freq"] * (1 + i * rng.uniform(0.4, 0.9))
        if harm_freq > sr / 2:
            break
        amp = p["ring_gain"] / (1 + i * 0.4)
        ring = ring + amp * torch.sin(2 * torch.pi * harm_freq * t) * ring_env

    result = click + ring
    result = _waveshape(result, p["drive"])

    return result


def _synth_cymbal(t: torch.Tensor, sr: int, rng: np.random.RandomState, chars: list[str]) -> torch.Tensor:
    """Dense inharmonic partial cluster + noise, with ADSR and waveshaping."""
    p = _apply_cymbal_chars(chars, rng)
    n = len(t)

    # Inharmonic partials from cymbal ratios
    partials = torch.zeros(n)
    n_use = min(p["n_partials"], len(_CYMBAL_RATIOS))
    for i in range(n_use):
        freq = p["base_freq"] * _CYMBAL_RATIOS[i]
        if freq > sr / 2:
            continue
        partial_decay = p["decay_ms"] * rng.uniform(0.5, 1.5)
        partial_env = _adsr_envelope(n, sr, rng.uniform(0.1, 1.0), 0.5, partial_decay)
        amp = rng.uniform(0.3, 1.0) / (1 + i * 0.15)
        partials = partials + amp * torch.sin(2 * torch.pi * freq * t) * partial_env

    # Noise layer
    noise = torch.randn(n)
    noise = _biquad_highpass(noise, sr, p["hp_cutoff"])
    noise_env = _adsr_envelope(n, sr, 0.5, 1.0, p["decay_ms"] * 0.8)
    noise = noise * noise_env

    result = (1 - p["noise_mix"]) * partials + p["noise_mix"] * noise
    result = _waveshape(result, p["drive"])

    # Highpass to clean up
    result = _biquad_highpass(result, sr, p["hp_cutoff"] * 0.5)

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
