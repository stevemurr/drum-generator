"""
recon_eval.py
-------------
Quantitative VAE reconstruction quality metrics on a memmap dataset.

Takes a VAE checkpoint + memmap directory, samples N examples, runs each
through the VAE (deterministic mu-only, no reparameterization) and the
DAC decoder, then reports:

  - spectral centroid shift (true vs recon, Hz)
  - low-band (<200 Hz) energy ratio
  - per-mel-band log-magnitude L1 error, with worst bands highlighted

Use for numerical A/B comparison across VAE training configs.

Usage:
  python scripts/recon_eval.py \\
      --ckpt /path/to/vae_best.pt \\
      --memmap-dir /path/to/memmap \\
      --n 200
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch

from drum_generator.codec import decode_from_dac_latent
from drum_generator.config import CFG
from drum_generator.vae import DrumVAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 44100
N_FFT = 4096
N_MELS = 64
LOW_BAND_CUTOFF = 200.0


def load_vae(path: str) -> DrumVAE:
    """Load a VAE from either checkpoint format, auto-adjusting CFG to match.

    The VAE's `vae_latent_dim` is a construction-time config, so if the
    checkpoint was trained with a non-default latent dim we have to set
    CFG before constructing the model.
    """
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Infer vae_latent_dim from mu_head shape: (latent_dim, 128, 1)
    mu_head = state.get("encoder.mu_head.weight")
    if mu_head is not None:
        latent_dim = mu_head.shape[0]
        if latent_dim != CFG.vae_latent_dim:
            print(
                f"[recon_eval] ckpt vae_latent_dim={latent_dim} "
                f"(CFG default={CFG.vae_latent_dim}); adjusting CFG"
            )
            CFG.vae_latent_dim = latent_dim

    vae = DrumVAE().to(DEVICE).eval()
    vae.load_state_dict(state)
    return vae


def load_memmap(memmap_dir: str):
    d = Path(memmap_dir)
    dac = np.load(d / "dac_latents.npy", mmap_mode="r")
    wav = np.load(d / "waveforms.npy", mmap_mode="r")
    return dac, wav


def load_tag_filter(memmap_dir: str, keywords: list[str]) -> np.ndarray:
    """Return memmap-row indices whose tags contain any of `keywords`.

    Reads the index.json row order and captions_structured.jsonl (keyed
    by sha16) from the memmap directory. Matching is case-insensitive
    substring against the compact `tags` field — e.g. "kick" matches
    "kick rock clean...".
    """
    d = Path(memmap_dir)
    with open(d / "index.json") as f:
        sha_order = json.load(f)

    tags_by_sha: dict[str, str] = {}
    with open(d / "captions_structured.jsonl") as f:
        for line in f:
            row = json.loads(line)
            tags_by_sha[row["sha16"]] = row.get("tags", "")

    kw = [k.lower() for k in keywords]
    kept = []
    for i, sha in enumerate(sha_order):
        tags = tags_by_sha.get(sha, "").lower()
        if any(k in tags for k in kw):
            kept.append(i)
    return np.array(kept, dtype=np.int64)


def spectral_centroid(wav: np.ndarray) -> float:
    return float(librosa.feature.spectral_centroid(y=wav, sr=SR, n_fft=N_FFT).mean())


def low_band_ratio(wav: np.ndarray) -> float:
    S = np.abs(librosa.stft(wav, n_fft=N_FFT)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    low = freqs < LOW_BAND_CUTOFF
    total = S.sum()
    if total < 1e-12:
        return 0.0
    return float(S[low].sum() / total)


def log_mel(wav: np.ndarray) -> np.ndarray:
    M = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=N_FFT, n_mels=N_MELS)
    return np.log(M + 1e-7)


def crest_factor(wav: np.ndarray) -> float:
    """peak / RMS — high value = sharp transient, low value = smeared energy."""
    wav = wav.astype(np.float32)
    rms = float(np.sqrt(np.mean(wav ** 2)))
    if rms < 1e-12:
        return 0.0
    return float(np.abs(wav).max() / rms)


def attack_ms(wav: np.ndarray) -> float:
    """Time from 10%-of-peak onset to peak amplitude, in milliseconds.

    Uses a squared-signal envelope with 2ms smoothing. Robust for short
    drum one-shots; doesn't depend on pitch tracking.
    """
    wav = wav.astype(np.float32)
    env = wav ** 2
    win = max(1, int(SR * 0.002))  # 2ms smoothing
    env = np.convolve(env, np.ones(win) / win, mode="same")
    peak_idx = int(np.argmax(env))
    peak = env[peak_idx]
    if peak < 1e-10:
        return 0.0
    threshold = peak * 0.1
    onset_candidates = np.where(env[:peak_idx] > threshold)[0]
    onset_idx = int(onset_candidates[0]) if len(onset_candidates) > 0 else 0
    return max(peak_idx - onset_idx, 0) * 1000.0 / SR


@torch.no_grad()
def recon_batch(vae: DrumVAE, dac_batch: np.ndarray) -> np.ndarray:
    """(B, 1024, 129) numpy → (B, 66048) waveform numpy via mu-only decode."""
    z = torch.from_numpy(dac_batch.copy()).to(DEVICE)
    mu, _ = vae.encode(z)
    z_hat = vae.decode(mu)
    wav_hat = decode_from_dac_latent(z_hat, DEVICE)
    return wav_hat.cpu().numpy()


def pct(a: np.ndarray, p: float) -> float:
    return float(np.percentile(a, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to vae_*.pt")
    ap.add_argument("--memmap-dir", required=True, help="memmap dataset directory")
    ap.add_argument("--n", type=int, default=200, help="number of samples to evaluate")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--tag-filter",
        default=None,
        help="Comma-separated keywords; only eval samples whose captions "
             "tags field contains any of them (case-insensitive substring "
             "match). Example: --tag-filter kick,bass-drum",
    )
    args = ap.parse_args()

    print(f"[recon_eval] ckpt:       {args.ckpt}")
    print(f"[recon_eval] memmap:     {args.memmap_dir}")
    print(f"[recon_eval] device:     {DEVICE}")

    vae = load_vae(args.ckpt)
    dac, wav = load_memmap(args.memmap_dir)
    print(f"[recon_eval] dac shape:  {dac.shape}  waveform shape: {wav.shape}")

    rng = np.random.default_rng(args.seed)
    if args.tag_filter:
        keywords = [k.strip() for k in args.tag_filter.split(",") if k.strip()]
        candidates = load_tag_filter(args.memmap_dir, keywords)
        print(
            f"[recon_eval] tag filter {keywords!r}: "
            f"{len(candidates)}/{len(dac)} samples matched"
        )
        if len(candidates) == 0:
            raise SystemExit("no samples matched the tag filter")
    else:
        candidates = np.arange(len(dac))

    n = min(args.n, len(candidates))
    idx = rng.choice(candidates, size=n, replace=False)
    idx.sort()  # memmap friendly

    centroid_true = np.zeros(n, dtype=np.float32)
    centroid_hat = np.zeros(n, dtype=np.float32)
    lbr_true = np.zeros(n, dtype=np.float32)
    lbr_hat = np.zeros(n, dtype=np.float32)
    crest_true = np.zeros(n, dtype=np.float32)
    crest_hat = np.zeros(n, dtype=np.float32)
    attack_true = np.zeros(n, dtype=np.float32)
    attack_hat = np.zeros(n, dtype=np.float32)
    mel_errors = np.zeros((n, N_MELS), dtype=np.float32)

    B = args.batch_size
    cursor = 0
    while cursor < n:
        chunk = idx[cursor : cursor + B]
        dac_chunk = np.stack([dac[i] for i in chunk])  # (b, 1024, 129)
        wav_hat_chunk = recon_batch(vae, dac_chunk)  # (b, 66048) np
        wav_true_chunk = np.stack([wav[i].astype(np.float32) for i in chunk])

        for j in range(len(chunk)):
            wt = wav_true_chunk[j]
            wh = wav_hat_chunk[j]
            centroid_true[cursor + j] = spectral_centroid(wt)
            centroid_hat[cursor + j] = spectral_centroid(wh)
            lbr_true[cursor + j] = low_band_ratio(wt)
            lbr_hat[cursor + j] = low_band_ratio(wh)
            crest_true[cursor + j] = crest_factor(wt)
            crest_hat[cursor + j] = crest_factor(wh)
            attack_true[cursor + j] = attack_ms(wt)
            attack_hat[cursor + j] = attack_ms(wh)
            mel_errors[cursor + j] = np.abs(log_mel(wh) - log_mel(wt)).mean(axis=1)

        cursor += len(chunk)
        print(f"[recon_eval] {cursor}/{n}", end="\r", flush=True)
    print()

    centroid_delta = centroid_hat - centroid_true
    lbr_delta = lbr_hat - lbr_true
    crest_delta = crest_hat - crest_true
    attack_delta = attack_hat - attack_true

    print(f"\n=== recon eval: {n} samples ===\n")

    print("spectral centroid (Hz) — 'how bright is it overall'")
    print(f"  true:   median={np.median(centroid_true):7.0f}  mean={centroid_true.mean():7.0f}")
    print(f"  recon:  median={np.median(centroid_hat):7.0f}  mean={centroid_hat.mean():7.0f}")
    print(
        f"  shift:  median={np.median(centroid_delta):+7.0f}  "
        f"mean={centroid_delta.mean():+7.0f}  "
        f"p25={pct(centroid_delta,25):+7.0f}  p75={pct(centroid_delta,75):+7.0f}"
    )
    shift_up = int((centroid_delta > 0).sum())
    print(f"  samples brighter than truth: {shift_up}/{n} ({100*shift_up/n:.1f}%)")

    print(f"\nlow-band (<{LOW_BAND_CUTOFF:.0f} Hz) energy ratio — 'how much sub-bass'")
    print(f"  true:   median={np.median(lbr_true):.3f}  mean={lbr_true.mean():.3f}")
    print(f"  recon:  median={np.median(lbr_hat):.3f}  mean={lbr_hat.mean():.3f}")
    print(
        f"  delta:  median={np.median(lbr_delta):+.3f}  "
        f"mean={lbr_delta.mean():+.3f}  "
        f"p25={pct(lbr_delta,25):+.3f}  p75={pct(lbr_delta,75):+.3f}"
    )
    thinner = int((lbr_delta < 0).sum())
    print(f"  samples with less sub-bass than truth: {thinner}/{n} ({100*thinner/n:.1f}%)")

    print(f"\ncrest factor (peak/RMS) — 'transient sharpness'")
    print(f"  true:   median={np.median(crest_true):.2f}  mean={crest_true.mean():.2f}")
    print(f"  recon:  median={np.median(crest_hat):.2f}  mean={crest_hat.mean():.2f}")
    print(
        f"  delta:  median={np.median(crest_delta):+.2f}  "
        f"mean={crest_delta.mean():+.2f}  "
        f"p25={pct(crest_delta,25):+.2f}  p75={pct(crest_delta,75):+.2f}"
    )
    softer = int((crest_delta < 0).sum())
    print(f"  samples with lower crest factor (softer transient): "
          f"{softer}/{n} ({100*softer/n:.1f}%)")

    print(f"\nattack time (ms) — 'onset to peak amplitude'")
    print(f"  true:   median={np.median(attack_true):.1f}  mean={attack_true.mean():.1f}")
    print(f"  recon:  median={np.median(attack_hat):.1f}  mean={attack_hat.mean():.1f}")
    print(
        f"  delta:  median={np.median(attack_delta):+.1f}  "
        f"mean={attack_delta.mean():+.1f}  "
        f"p25={pct(attack_delta,25):+.1f}  p75={pct(attack_delta,75):+.1f}"
    )
    slower = int((attack_delta > 0).sum())
    print(f"  samples with slower attack than truth: "
          f"{slower}/{n} ({100*slower/n:.1f}%)")

    print(f"\nper-mel-band log-magnitude L1 error (mean across time, median across samples)")
    median_per_band = np.median(mel_errors, axis=0)
    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmax=SR / 2)
    worst = np.argsort(median_per_band)[::-1][:10]
    print(f"  overall median error:  {median_per_band.mean():.3f}")
    print(f"  worst 10 mel bands:")
    for rank, bin_i in enumerate(worst):
        lo = mel_freqs[max(bin_i - 1, 0)]
        hi = mel_freqs[min(bin_i + 1, N_MELS - 1)]
        print(
            f"    #{rank+1:2d}  bin {bin_i:2d}  ~{mel_freqs[bin_i]:6.0f} Hz  "
            f"(range {lo:6.0f}-{hi:6.0f})  err={median_per_band[bin_i]:.3f}"
        )

    # Compact summary line for logging across experiments
    summary = {
        "ckpt": args.ckpt,
        "n": n,
        "centroid_shift_median_hz": float(np.median(centroid_delta)),
        "low_band_ratio_delta_median": float(np.median(lbr_delta)),
        "crest_delta_median": float(np.median(crest_delta)),
        "attack_delta_median_ms": float(np.median(attack_delta)),
        "mel_error_mean": float(median_per_band.mean()),
    }
    print(f"\nsummary: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
