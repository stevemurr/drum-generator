"""
train.py
--------
Phase 1: Train VAE  (waveform → DAC latents → VAE latents → DAC latents)
Phase 2: Train DiT  (flow matching in VAE latent space, CLAP + ref conditioned)

Run:
    python train.py --phase vae
    python train.py --phase dit
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from drum_generator.codec import decode_from_dac_latent, encode_to_dac_latent
from drum_generator.config import CFG
from drum_generator.dataset import build_dataset
from drum_generator.dit import DrumDiT, flow_matching_loss
from drum_generator.vae import DrumVAE, vae_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Phase 1: Train VAE
# ---------------------------------------------------------------------------


def _build_stft_losses():
    """Build (linear_mrstft, mel_mrstft) pair based on CFG weights.

    FFT sizes tuned for drums at 44.1 kHz: 2048 catches low-frequency body
    (down to ~22 Hz bin), 1024 mid detail, 512 catches transients (~11 ms).

    - linear MRSTFT: magnitude + phase term (w_phs = CFG.vae_stft_phase_weight).
      The phase term directly penalizes phase-incoherence artifacts (ringing,
      tonal smearing in high frequencies) that magnitude-only loss misses.
    - mel MRSTFT: mel-scaled magnitude loss that compresses narrow
      high-frequency spikes into perceptually wider bins, reducing the
      optimizer's incentive to "pay off" magnitude targets with ringy tones.

    Either component may be None if its weight is 0.
    """
    from auraloss.freq import MultiResolutionSTFTLoss

    fft = [2048, 1024, 512]
    hop = [512, 256, 128]
    win = [2048, 1024, 512]

    linear = None
    if CFG.vae_stft_weight > 0:
        linear = MultiResolutionSTFTLoss(
            fft_sizes=fft,
            hop_sizes=hop,
            win_lengths=win,
            w_sc=1.0,
            w_log_mag=1.0,
            w_lin_mag=0.0,
            w_phs=CFG.vae_stft_phase_weight,
        ).to(DEVICE)

    mel = None
    if CFG.vae_stft_mel_weight > 0:
        # n_bins=64 — at fft_size=512 (smallest resolution), 128 mel bins
        # produces empty filters at the low end (too narrow vs FFT bin width)
        # and yields NaN. 64 bins is safe across [2048, 1024, 512].
        mel = MultiResolutionSTFTLoss(
            fft_sizes=fft,
            hop_sizes=hop,
            win_lengths=win,
            w_sc=1.0,
            w_log_mag=1.0,
            w_lin_mag=0.0,
            scale="mel",
            sample_rate=CFG.sample_rate,
            n_bins=64,
        ).to(DEVICE)

    return linear, mel


def train_vae(train_loader, val_loader):
    vae = DrumVAE().to(DEVICE)
    opt = torch.optim.AdamW(vae.parameters(), lr=CFG.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG.vae_epochs)

    stft_linear, stft_mel = _build_stft_losses()
    stft_enabled = stft_linear is not None or stft_mel is not None
    if stft_enabled:
        parts = []
        if stft_linear is not None:
            part = f"linear(w={CFG.vae_stft_weight}"
            if CFG.vae_stft_phase_weight > 0:
                part += f", w_phs={CFG.vae_stft_phase_weight}"
            part += ")"
            parts.append(part)
        if stft_mel is not None:
            parts.append(f"mel(w={CFG.vae_stft_mel_weight})")
        print(f"[vae] STFT losses enabled: {' + '.join(parts)}")

    def apply_stft(loss, dac_z, recon_z, no_grad_recon: bool):
        """Adds the STFT loss terms (linear + mel, as configured) to `loss`.

        When `no_grad_recon` is True, the recon-path DAC decode skips
        gradient tracking (used during validation). Otherwise gradients
        flow through DAC back into the VAE (used during training).
        """
        if not stft_enabled:
            return loss
        wav_true = decode_from_dac_latent(dac_z, DEVICE, no_grad=True)
        wav_hat = decode_from_dac_latent(recon_z, DEVICE, no_grad=no_grad_recon)
        wh = wav_hat.unsqueeze(1)
        wt = wav_true.unsqueeze(1)
        if stft_linear is not None:
            loss = loss + CFG.vae_stft_weight * stft_linear(wh, wt)
        if stft_mel is not None:
            loss = loss + CFG.vae_stft_mel_weight * stft_mel(wh, wt)
        return loss

    best_val = float("inf")
    for epoch in range(1, CFG.vae_epochs + 1):
        # --- train ---
        vae.train()
        train_loss = 0.0
        for audio, _ in train_loader:  # ignore CLAP for VAE phase
            audio = audio.to(DEVICE)
            # Detect cached DAC latents (3D) vs raw waveforms (2D)
            dac_z = audio if audio.dim() == 3 else encode_to_dac_latent(audio, DEVICE)

            recon_z, mu, logvar, _ = vae(dac_z)

            # KL warmup: ramp weight from 0 → target over warmup period
            kl_w = min(CFG.vae_kl_weight, CFG.vae_kl_weight * epoch / CFG.vae_kl_warmup)
            loss, recon, kl = vae_loss(recon_z, dac_z, mu, logvar, kl_w)

            loss = apply_stft(loss, dac_z, recon_z, no_grad_recon=False)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()

        sch.step()

        # --- val ---
        vae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for audio, _ in val_loader:
                audio = audio.to(DEVICE)
                dac_z = audio if audio.dim() == 3 else encode_to_dac_latent(audio, DEVICE)
                recon_z, mu, logvar, _ = vae(dac_z)
                loss, _, _ = vae_loss(recon_z, dac_z, mu, logvar, CFG.vae_kl_weight)
                loss = apply_stft(loss, dac_z, recon_z, no_grad_recon=True)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(
            f"VAE epoch {epoch:3d}/{CFG.vae_epochs} | "
            f"train {train_loss:.5f} | val {val_loss:.5f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(vae.state_dict(), f"{CFG.ckpt_dir}/vae_best.pt")
            print(f"  ✓ saved vae_best.pt")

    return vae


# ---------------------------------------------------------------------------
# Derangement helper (permutation with no fixed points)
# ---------------------------------------------------------------------------


def _derangement(n: int, device: str = "cpu") -> torch.Tensor:
    """Random permutation where perm[i] != i for all i."""
    perm = torch.randperm(n, device=device)
    for i in range(n):
        if perm[i] == i:
            swap = (i + 1) % n
            perm[i], perm[swap] = perm[swap].item(), perm[i].item()
    return perm


# ---------------------------------------------------------------------------
# Phase 2: Train DiT (flow matching)
# ---------------------------------------------------------------------------


def train_dit(train_loader, val_loader, vae: DrumVAE):
    dit = DrumDiT().to(DEVICE)
    opt = torch.optim.AdamW(dit.parameters(), lr=CFG.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG.dit_epochs)

    vae.eval()
    best_val = float("inf")

    for epoch in range(1, CFG.dit_epochs + 1):
        # --- train ---
        dit.train()
        train_loss = 0.0
        for audio, clap_embeds in train_loader:
            audio = audio.to(DEVICE)
            clap_embeds = clap_embeds.to(DEVICE)

            # Encode to VAE latent (no grad — VAE is frozen)
            with torch.no_grad():
                dac_z = audio if audio.dim() == 3 else encode_to_dac_latent(audio, DEVICE)
                mu, logvar = vae.encode(dac_z)
                x1 = vae.reparameterize(mu, logvar)  # (B, 16, T)

            # Create reference by shuffling batch (no self-reference)
            B = x1.shape[0]
            perm = _derangement(B, device=x1.device)
            ref_z = x1[perm]  # (B, 16, T) — reuse already-computed latents

            loss = flow_matching_loss(
                dit, x1, clap_embeds,
                ref_z=ref_z,
                cfg_dropout=CFG.cfg_dropout,
                ref_dropout=CFG.ref_dropout,
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()

        sch.step()

        # --- val ---
        dit.eval()
        val_loss = 0.0
        with torch.no_grad():
            for audio, clap_embeds in val_loader:
                audio = audio.to(DEVICE)
                clap_embeds = clap_embeds.to(DEVICE)
                dac_z = audio if audio.dim() == 3 else encode_to_dac_latent(audio, DEVICE)
                mu, logvar = vae.encode(dac_z)
                x1 = vae.reparameterize(mu, logvar)

                B = x1.shape[0]
                perm = _derangement(B, device=x1.device)
                ref_z = x1[perm]

                loss = flow_matching_loss(
                    dit, x1, clap_embeds,
                    ref_z=ref_z,
                    cfg_dropout=0.0,
                    ref_dropout=0.0,
                )
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(
            f"DiT epoch {epoch:3d}/{CFG.dit_epochs} | "
            f"train {train_loss:.5f} | val {val_loss:.5f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(dit.state_dict(), f"{CFG.ckpt_dir}/dit_best.pt")
            print(f"  ✓ saved dit_best.pt")

    return dit


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["vae", "dit", "both"], default="both")
    parser.add_argument(
        "--memmap-dir",
        default=None,
        help="Directory containing precomputed dac_latents.npy + "
             "embeddings_text.npy + index.json (from dataset-caption pipeline). "
             "Skips wav decode / DAC encode / CLAP encode at training time. "
             "Disables waveform augmentation.",
    )
    parser.add_argument(
        "--ckpt-dir",
        default=None,
        help="Directory to save vae_best.pt / dit_best.pt. "
             f"Default: CFG.ckpt_dir ({CFG.ckpt_dir!r}, relative to cwd).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Batch size for the DataLoader (applies to both VAE and DiT "
             f"phases). Default: CFG.batch_size ({CFG.batch_size}).",
    )
    parser.add_argument(
        "--vae-epochs",
        type=int,
        default=None,
        help=f"Number of VAE training epochs. "
             f"Default: CFG.vae_epochs ({CFG.vae_epochs}).",
    )
    parser.add_argument(
        "--dit-epochs",
        type=int,
        default=None,
        help=f"Number of DiT training epochs. "
             f"Default: CFG.dit_epochs ({CFG.dit_epochs}).",
    )
    parser.add_argument(
        "--vae-kl-weight",
        type=float,
        default=None,
        help=f"VAE KL regularization weight (higher = smoother latent space, "
             f"looser reconstruction). Default: CFG.vae_kl_weight ({CFG.vae_kl_weight}).",
    )
    parser.add_argument(
        "--vae-kl-warmup",
        type=int,
        default=None,
        help=f"Epochs to ramp KL weight from 0 to --vae-kl-weight. "
             f"Default: CFG.vae_kl_warmup ({CFG.vae_kl_warmup}).",
    )
    parser.add_argument(
        "--vae-latent-dim",
        type=int,
        default=None,
        help=f"VAE bottleneck channel count. NOTE: changing this invalidates "
             f"any existing vae_best.pt / dit_best.pt. "
             f"Default: CFG.vae_latent_dim ({CFG.vae_latent_dim}).",
    )
    parser.add_argument(
        "--vae-hidden",
        type=int,
        default=None,
        help=f"VAE encoder/decoder base hidden channels. Bigger = more capacity "
             f"and more memory. Default: CFG.vae_hidden ({CFG.vae_hidden}).",
    )
    parser.add_argument(
        "--vae-stft-weight",
        type=float,
        default=None,
        help=f"Weight for linear multi-res STFT magnitude loss (applied in "
             f"waveform space via DAC decode). 0 disables the linear STFT "
             f"term; typical values 0.5–4.0. Roughly doubles per-step training "
             f"cost when any STFT term is enabled. "
             f"Default: CFG.vae_stft_weight ({CFG.vae_stft_weight}).",
    )
    parser.add_argument(
        "--vae-stft-phase-weight",
        type=float,
        default=None,
        help=f"Phase term (w_phs) inside the linear MRSTFT constructor — "
             f"targets phase-incoherence artifacts (ringing, tonal smearing) "
             f"that magnitude-only STFT loss misses. 0 disables. Typical "
             f"values 0.1–0.5; higher can destabilize training. "
             f"Default: CFG.vae_stft_phase_weight ({CFG.vae_stft_phase_weight}).",
    )
    parser.add_argument(
        "--vae-stft-mel-weight",
        type=float,
        default=None,
        help=f"Weight for a separate mel-scaled multi-res STFT loss, added "
             f"on top of the linear STFT term. Mel compresses narrow "
             f"high-frequency spikes into perceptually wider bins, reducing "
             f"the optimizer's incentive to produce ringy tones to match "
             f"magnitude targets. 0 disables. Typical values 0.5–2.0. "
             f"Default: CFG.vae_stft_mel_weight ({CFG.vae_stft_mel_weight}).",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        choices=["auto", "on", "off"],
        default="auto",
        help="cuDNN algorithm benchmarking mode. 'on' (PyTorch default) times "
             "multiple conv algorithms on first call per shape and caches the "
             "fastest — ~15-30%% faster steady-state but causes a multi-minute "
             "first-step stall for conv-heavy networks. 'off' uses a heuristic "
             "instead — slightly slower steady-state but immediate start and "
             "lower workspace memory. 'auto' = off when --vae-stft-weight > 0 "
             "(DAC decoder has many shapes and GB10 benchmark payoff is low), "
             "otherwise on. Default: auto.",
    )
    args = parser.parse_args()

    if args.memmap_dir:
        CFG.memmap_dir = args.memmap_dir
        print(f"[train] memmap mode: {args.memmap_dir}")

    if args.ckpt_dir:
        CFG.ckpt_dir = args.ckpt_dir
        print(f"[train] ckpt dir: {args.ckpt_dir}")

    if args.batch_size is not None:
        CFG.batch_size = args.batch_size
        print(f"[train] batch_size: {args.batch_size}")

    if args.vae_epochs is not None:
        CFG.vae_epochs = args.vae_epochs
        print(f"[train] vae_epochs: {args.vae_epochs}")

    if args.dit_epochs is not None:
        CFG.dit_epochs = args.dit_epochs
        print(f"[train] dit_epochs: {args.dit_epochs}")

    if args.vae_kl_weight is not None:
        CFG.vae_kl_weight = args.vae_kl_weight
        print(f"[train] vae_kl_weight: {args.vae_kl_weight}")

    if args.vae_kl_warmup is not None:
        CFG.vae_kl_warmup = args.vae_kl_warmup
        print(f"[train] vae_kl_warmup: {args.vae_kl_warmup}")

    if args.vae_latent_dim is not None:
        CFG.vae_latent_dim = args.vae_latent_dim
        print(f"[train] vae_latent_dim: {args.vae_latent_dim}")

    if args.vae_hidden is not None:
        CFG.vae_hidden = args.vae_hidden
        print(f"[train] vae_hidden: {args.vae_hidden}")

    if args.vae_stft_weight is not None:
        CFG.vae_stft_weight = args.vae_stft_weight
        print(f"[train] vae_stft_weight: {args.vae_stft_weight}")

    if args.vae_stft_phase_weight is not None:
        CFG.vae_stft_phase_weight = args.vae_stft_phase_weight
        print(f"[train] vae_stft_phase_weight: {args.vae_stft_phase_weight}")

    if args.vae_stft_mel_weight is not None:
        CFG.vae_stft_mel_weight = args.vae_stft_mel_weight
        print(f"[train] vae_stft_mel_weight: {args.vae_stft_mel_weight}")

    # cuDNN benchmark mode. 'auto' = off when STFT loss is enabled (DAC decoder
    # has many conv shapes and optimal GB10 kernels often aren't available, so
    # benchmarking is wasted work and stalls first-step). 'on' | 'off' override.
    if args.cudnn_benchmark == "on":
        torch.backends.cudnn.benchmark = True
        print("[train] cudnn.benchmark: on")
    elif args.cudnn_benchmark == "off":
        torch.backends.cudnn.benchmark = False
        print("[train] cudnn.benchmark: off")
    else:  # auto
        auto_off = (
            (CFG.vae_stft_weight and CFG.vae_stft_weight > 0)
            or (CFG.vae_stft_mel_weight and CFG.vae_stft_mel_weight > 0)
        )
        torch.backends.cudnn.benchmark = not auto_off
        print(f"[train] cudnn.benchmark: auto -> {'off' if auto_off else 'on'}")

    print(f"Training on {DEVICE}")
    os.makedirs(CFG.ckpt_dir, exist_ok=True)

    # Dataset + splits
    ds = build_dataset()
    n_val = max(1, int(len(ds) * 0.1))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    if args.phase in ("vae", "both"):
        print("\n=== Phase 1: VAE training ===")
        vae = train_vae(train_loader, val_loader)

    if args.phase in ("dit", "both"):
        print("\n=== Phase 2: DiT training ===")
        vae = DrumVAE().to(DEVICE)
        vae.load_state_dict(
            torch.load(f"{CFG.ckpt_dir}/vae_best.pt", map_location=DEVICE)
        )
        train_dit(train_loader, val_loader, vae)


if __name__ == "__main__":
    main()
