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

from drum_generator.codec import encode_to_dac_latent
from drum_generator.config import CFG
from drum_generator.dataset import build_dataset
from drum_generator.dit import DrumDiT, flow_matching_loss
from drum_generator.vae import DrumVAE, vae_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Phase 1: Train VAE
# ---------------------------------------------------------------------------


def train_vae(train_loader, val_loader):
    vae = DrumVAE().to(DEVICE)
    opt = torch.optim.AdamW(vae.parameters(), lr=CFG.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG.vae_epochs)

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
    args = parser.parse_args()

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
