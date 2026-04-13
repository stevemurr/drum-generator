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


def _build_stft_loss():
    """Multi-resolution STFT loss for perceptual waveform-space reconstruction.

    FFT sizes tuned for drums at 44.1 kHz: 2048 catches low-frequency body
    (down to ~22 Hz bin), 1024 mid detail, 512 catches transients (~11 ms).
    No mel scaling, no perceptual weighting — we want to preserve drum body,
    not de-emphasize it.
    """
    from auraloss.freq import MultiResolutionSTFTLoss
    return MultiResolutionSTFTLoss(
        fft_sizes=[2048, 1024, 512],
        hop_sizes=[512, 256, 128],
        win_lengths=[2048, 1024, 512],
        w_sc=1.0,
        w_log_mag=1.0,
        w_lin_mag=0.0,
    ).to(DEVICE)


def train_vae(train_loader, val_loader):
    vae = DrumVAE().to(DEVICE)
    opt = torch.optim.AdamW(vae.parameters(), lr=CFG.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG.vae_epochs)

    stft_loss_fn = None
    if CFG.vae_stft_weight and CFG.vae_stft_weight > 0:
        stft_loss_fn = _build_stft_loss()
        print(f"[vae] multi-res STFT loss enabled (weight={CFG.vae_stft_weight})")

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

            if stft_loss_fn is not None:
                # Compare waveforms post-DAC-decode. Target decode runs under
                # no_grad to save memory; prediction decode keeps gradients so
                # the STFT loss can backprop through DAC → VAE.
                wav_true = decode_from_dac_latent(dac_z, DEVICE, no_grad=True)
                wav_hat = decode_from_dac_latent(recon_z, DEVICE, no_grad=False)
                stft_loss = stft_loss_fn(wav_hat.unsqueeze(1), wav_true.unsqueeze(1))
                loss = loss + CFG.vae_stft_weight * stft_loss

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
                if stft_loss_fn is not None:
                    wav_true = decode_from_dac_latent(dac_z, DEVICE, no_grad=True)
                    wav_hat = decode_from_dac_latent(recon_z, DEVICE, no_grad=True)
                    stft_loss = stft_loss_fn(wav_hat.unsqueeze(1), wav_true.unsqueeze(1))
                    loss = loss + CFG.vae_stft_weight * stft_loss
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
        help=f"Weight for auraloss multi-res STFT loss (applied in waveform "
             f"space via DAC decode). 0 disables; typical values 0.1–1.0. "
             f"Roughly doubles per-step training cost when enabled. "
             f"Default: CFG.vae_stft_weight ({CFG.vae_stft_weight}).",
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
