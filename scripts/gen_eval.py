"""
gen_eval.py
-----------
DiT generation quality diagnostic. For each drum class:

  1. Generate N samples from a class-specific text prompt.
  2. Pull N real samples of the same class from the memmap.
  3. Compute acoustic metrics on both sets.
  4. Report distribution-level deltas (median, IQR).

Unlike recon_eval.py, this doesn't have per-sample ground truth (the DiT
samples from noise conditioned on text, no paired reference). Comparisons
are therefore distributional: "are generated kicks, as a population, in
the same acoustic neighborhood as real kicks, as a population?"

Metrics:
  - spectral centroid   — 'how bright is the overall sound'
  - spectral bandwidth  — 'how spread-out is the energy around the centroid'
                          (tight body = narrow, blob = wide)
  - spectral flatness   — 'tonal vs noisy' (real kicks are tonal, real
                          snares are mixed, real hats are flat)
  - low-band ratio      — 'how much sub-200Hz energy'
  - crest factor        — 'peak / RMS' — transient sharpness
  - attack time (ms)    — 'onset to peak amplitude'
  - decay time (ms)     — 'peak to -20dB below peak'

Usage:
  python scripts/gen_eval.py \\
      --vae-ckpt /path/to/vae_best.pt \\
      --dit-ckpt /path/to/dit_best.pt \\
      --memmap-dir /path/to/memmap \\
      --n-per-class 40
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch

from drum_generator.codec import decode_from_dac_latent
from drum_generator.config import CFG
from drum_generator.dit import DrumDiT, generate as fm_generate
from drum_generator.vae import DrumVAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 44100
N_FFT = 4096
LOW_BAND_CUTOFF = 200.0


# ---------------------------------------------------------------------------
# Checkpoint loading (mirrors recon_eval.py)
# ---------------------------------------------------------------------------


def _unwrap(ckpt):
    return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt


def load_vae(path: str) -> DrumVAE:
    state = _unwrap(torch.load(path, map_location=DEVICE, weights_only=False))
    mu_head = state.get("encoder.mu_head.weight")
    if mu_head is not None:
        CFG.vae_latent_dim = mu_head.shape[0]
    vae = DrumVAE().to(DEVICE).eval()
    vae.load_state_dict(state)
    return vae


def load_dit(path: str) -> DrumDiT:
    state = _unwrap(torch.load(path, map_location=DEVICE, weights_only=False))
    dit = DrumDiT().to(DEVICE).eval()
    dit.load_state_dict(state, strict=False)
    return dit


def load_memmap(memmap_dir: str):
    d = Path(memmap_dir)
    wav = np.load(d / "waveforms.npy", mmap_mode="r")
    return wav


def load_tag_indices(memmap_dir: str, keywords: list[str]) -> np.ndarray:
    d = Path(memmap_dir)
    with open(d / "index.json") as f:
        sha_order = json.load(f)
    tags_by_sha: dict[str, str] = {}
    with open(d / "captions_structured.jsonl") as f:
        for line in f:
            row = json.loads(line)
            tags_by_sha[row["sha16"]] = row.get("tags", "")
    kw = [k.lower() for k in keywords]
    return np.array(
        [
            i
            for i, sha in enumerate(sha_order)
            if any(k in tags_by_sha.get(sha, "").lower() for k in kw)
        ],
        dtype=np.int64,
    )


# ---------------------------------------------------------------------------
# CLAP text encoding (loaded lazily, cached)
# ---------------------------------------------------------------------------


_clap_cache: dict = {}


def _get_clap():
    if "model" not in _clap_cache:
        from transformers import ClapModel, ClapProcessor

        _clap_cache["proc"] = ClapProcessor.from_pretrained("laion/larger_clap_general")
        _clap_cache["model"] = (
            ClapModel.from_pretrained("laion/larger_clap_general").to(DEVICE).eval()
        )
    return _clap_cache["proc"], _clap_cache["model"]


def encode_prompt(prompt: str) -> torch.Tensor:
    proc, model = _get_clap()
    inputs = proc(text=prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    return out.pooler_output if hasattr(out, "pooler_output") else out  # (1, 512)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_batch(
    vae: DrumVAE,
    dit: DrumDiT,
    prompt: str,
    n: int,
    batch_size: int = 16,
    steps: int = 8,
    cfg_scale: float = 4.0,
) -> np.ndarray:
    """Generate n samples for a prompt. Returns (n, ~66k) waveforms."""
    clap_one = encode_prompt(prompt)  # (1, 512)
    out_chunks = []
    remaining = n
    while remaining > 0:
        b = min(batch_size, remaining)
        clap_b = clap_one.expand(b, -1).contiguous()
        z = fm_generate(
            dit,
            clap_embed=clap_b,
            ref_z=None,
            steps=steps,
            cfg_scale=cfg_scale,
            device=DEVICE,
        )
        dac_z_hat = vae.decode(z)
        wav = decode_from_dac_latent(dac_z_hat, DEVICE)
        out_chunks.append(wav.cpu().numpy())
        remaining -= b
    return np.concatenate(out_chunks, axis=0)  # (n, N_SAMPLES)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class Metrics:
    centroid: float
    bandwidth: float
    flatness: float
    low_band_ratio: float
    crest_factor: float
    attack_ms: float
    decay_ms: float


def _stft_power(wav: np.ndarray) -> np.ndarray:
    return np.abs(librosa.stft(wav, n_fft=N_FFT)) ** 2


def _low_band_ratio(S: np.ndarray) -> float:
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    low = freqs < LOW_BAND_CUTOFF
    total = S.sum()
    return float(S[low].sum() / total) if total > 1e-12 else 0.0


def _attack_decay_ms(wav: np.ndarray) -> tuple[float, float]:
    """Attack = onset → peak amplitude; decay = peak → -20 dB below peak.

    Uses an envelope approximation: squared signal smoothed with a short
    moving average. Robust enough for short drum one-shots.
    """
    env = wav.astype(np.float32) ** 2
    win = max(1, int(SR * 0.002))  # 2ms smoothing
    env = np.convolve(env, np.ones(win) / win, mode="same")
    peak_idx = int(np.argmax(env))
    peak = env[peak_idx]
    if peak < 1e-10:
        return 0.0, 0.0

    # Attack: first sample where env > 10% of peak, to peak
    threshold = peak * 0.1
    onset_candidates = np.where(env[:peak_idx] > threshold)[0]
    onset_idx = int(onset_candidates[0]) if len(onset_candidates) > 0 else 0
    attack_samples = max(peak_idx - onset_idx, 0)

    # Decay: peak to first sample below peak * 10^(-2) (= -20 dB) after peak
    decay_threshold = peak * (10 ** (-20 / 10))  # -20 dB
    after_peak = env[peak_idx:]
    below = np.where(after_peak < decay_threshold)[0]
    decay_samples = int(below[0]) if len(below) > 0 else len(after_peak)

    return attack_samples * 1000.0 / SR, decay_samples * 1000.0 / SR


def compute_metrics(wav: np.ndarray) -> Metrics:
    wav = wav.astype(np.float32)
    # DC removal — stabilizes spectral metrics on drum oneshots
    wav = wav - wav.mean()

    S = _stft_power(wav)
    centroid = float(librosa.feature.spectral_centroid(y=wav, sr=SR, n_fft=N_FFT).mean())
    bandwidth = float(librosa.feature.spectral_bandwidth(y=wav, sr=SR, n_fft=N_FFT).mean())
    flatness = float(librosa.feature.spectral_flatness(y=wav, n_fft=N_FFT).mean())
    lbr = _low_band_ratio(S)

    rms = np.sqrt(np.mean(wav ** 2))
    peak = float(np.abs(wav).max())
    crest = float(peak / rms) if rms > 1e-12 else 0.0

    attack_ms, decay_ms = _attack_decay_ms(wav)

    return Metrics(
        centroid=centroid,
        bandwidth=bandwidth,
        flatness=flatness,
        low_band_ratio=lbr,
        crest_factor=crest,
        attack_ms=attack_ms,
        decay_ms=decay_ms,
    )


def compute_metrics_batch(wavs: np.ndarray) -> list[Metrics]:
    return [compute_metrics(wavs[i]) for i in range(len(wavs))]


# ---------------------------------------------------------------------------
# Summary / comparison
# ---------------------------------------------------------------------------


def summarize(metrics: list[Metrics]) -> dict[str, dict[str, float]]:
    """Per-metric median / p25 / p75."""
    fields = ["centroid", "bandwidth", "flatness", "low_band_ratio",
              "crest_factor", "attack_ms", "decay_ms"]
    out = {}
    for f in fields:
        vals = np.array([getattr(m, f) for m in metrics])
        out[f] = {
            "median": float(np.median(vals)),
            "p25": float(np.percentile(vals, 25)),
            "p75": float(np.percentile(vals, 75)),
        }
    return out


def print_comparison(class_name: str, gen_summary: dict, real_summary: dict) -> None:
    print(f"\n=== {class_name} ===")
    header = f"  {'metric':<18} {'real median':>14}  {'gen median':>14}  {'delta':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    fmt_rules = {
        "centroid":       ("{:>12.0f} Hz", "{:+10.0f}"),
        "bandwidth":      ("{:>12.0f} Hz", "{:+10.0f}"),
        "flatness":       ("{:>14.4f}",    "{:+10.4f}"),
        "low_band_ratio": ("{:>14.3f}",    "{:+10.3f}"),
        "crest_factor":   ("{:>14.2f}",    "{:+10.2f}"),
        "attack_ms":      ("{:>12.1f} ms", "{:+10.1f}"),
        "decay_ms":       ("{:>12.1f} ms", "{:+10.1f}"),
    }
    for field, (val_fmt, delta_fmt) in fmt_rules.items():
        rm = real_summary[field]["median"]
        gm = gen_summary[field]["median"]
        delta = gm - rm
        print(
            f"  {field:<18} "
            f"{val_fmt.format(rm)}  {val_fmt.format(gm)}  {delta_fmt.format(delta)}"
        )


# ---------------------------------------------------------------------------
# Class prompts — one per class, keyword-match to the tag vocabulary
# ---------------------------------------------------------------------------


CLASS_PROMPTS: dict[str, str] = {
    "kick":   "punchy deep kick drum, sub-heavy, tight",
    "snare":  "tight acoustic snare drum, crisp",
    "hihat":  "closed hi-hat, bright, short",
    "tom":    "deep warm tom drum hit",
    "clap":   "clean clap, tight, punchy",
    "crash":  "bright crash cymbal",
    "perc":   "percussive hit, short",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-ckpt", required=True)
    ap.add_argument("--dit-ckpt", required=True)
    ap.add_argument("--memmap-dir", required=True)
    ap.add_argument("--n-per-class", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8, help="Euler ODE steps")
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument(
        "--classes",
        default=None,
        help="Comma-separated subset of classes to eval (default: all). "
             f"Choices: {sorted(CLASS_PROMPTS)}",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[gen_eval] vae: {args.vae_ckpt}")
    print(f"[gen_eval] dit: {args.dit_ckpt}")
    print(f"[gen_eval] memmap: {args.memmap_dir}")
    print(f"[gen_eval] n per class: {args.n_per_class}")

    vae = load_vae(args.vae_ckpt)
    dit = load_dit(args.dit_ckpt)
    wav_memmap = load_memmap(args.memmap_dir)

    rng = np.random.default_rng(args.seed)

    classes = (
        [c.strip() for c in args.classes.split(",") if c.strip()]
        if args.classes
        else sorted(CLASS_PROMPTS)
    )
    for c in classes:
        if c not in CLASS_PROMPTS:
            print(f"[gen_eval] skipping unknown class '{c}'")
            continue

    deltas_summary: dict[str, dict[str, float]] = {}

    for class_name in classes:
        if class_name not in CLASS_PROMPTS:
            continue
        prompt = CLASS_PROMPTS[class_name]

        # Generate
        print(f"\n[gen_eval] generating {args.n_per_class} '{class_name}' samples...")
        gen_wavs = generate_batch(
            vae, dit, prompt, args.n_per_class,
            batch_size=args.batch_size,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
        )
        gen_metrics = compute_metrics_batch(gen_wavs)
        gen_summary = summarize(gen_metrics)

        # Real comparison set
        real_idx_pool = load_tag_indices(args.memmap_dir, [class_name])
        if len(real_idx_pool) == 0:
            print(f"[gen_eval] no real samples for '{class_name}' — skipping compare")
            continue
        n_real = min(args.n_per_class, len(real_idx_pool))
        real_idx = rng.choice(real_idx_pool, size=n_real, replace=False)
        real_idx.sort()
        real_wavs = np.stack([wav_memmap[i].astype(np.float32) for i in real_idx])
        real_metrics = compute_metrics_batch(real_wavs)
        real_summary = summarize(real_metrics)

        print_comparison(class_name, gen_summary, real_summary)

        deltas_summary[class_name] = {
            f: gen_summary[f]["median"] - real_summary[f]["median"]
            for f in gen_summary
        }

    # Compact summary for scraping
    print("\nsummary_json: " + json.dumps({"deltas": deltas_summary}))


if __name__ == "__main__":
    main()
