"""
train.py
--------
Phase 1: Train VAE  (waveform → DAC latents → VAE latents → DAC latents)
Phase 2: Train DiT  (flow matching in VAE latent space, CLAP conditioned)

Run:
    python train.py --phase vae
    python train.py --phase dit
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from drum_generator.config import CFG
from drum_generator.dataset import build_dataset
from drum_generator.dit import DrumDiT, flow_matching_loss
from drum_generator.vae import DrumVAE, vae_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# DAC helper (lazy-loaded to avoid import cost if not needed)
# ---------------------------------------------------------------------------

_dac_model = None


def get_dac():
    global _dac_model
    if _dac_model is None:
        import dac

        _dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
        _dac_model = _dac_model.to(DEVICE).eval()
    return _dac_model


def encode_to_dac_latent(waveform: torch.Tensor) -> torch.Tensor:
    """
    waveform: (B, N_SAMPLES) float32
    returns:  (B, dac_latent_dim=64, T=130) continuous latent (pre-quantization)
    """
    dac = get_dac()
    with torch.no_grad():
        wav = waveform.unsqueeze(1).to(DEVICE)  # (B, 1, N)
        z, _, _, _, _ = dac.encode(wav)  # continuous encoder output
    return z  # (B, 64, T)


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
        for wavs, _ in train_loader:  # ignore CLAP for VAE phase
            wavs = wavs.to(DEVICE)
            dac_z = encode_to_dac_latent(wavs)  # (B, 64, T)

            recon_z, mu, logvar, _ = vae(dac_z)

            # KL warmup: ramp weight from 0 → 1e-4 over first 20 epochs
            kl_w = min(1e-4, 1e-4 * epoch / 20)
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
            for wavs, _ in val_loader:
                dac_z = encode_to_dac_latent(wavs.to(DEVICE))
                recon_z, mu, logvar, _ = vae(dac_z)
                loss, _, _ = vae_loss(recon_z, dac_z, mu, logvar, 1e-4)
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
        for wavs, clap_embeds in train_loader:
            wavs = wavs.to(DEVICE)
            clap_embeds = clap_embeds.to(DEVICE)

            # Encode to VAE latent (no grad — VAE is frozen)
            with torch.no_grad():
                dac_z = encode_to_dac_latent(wavs)  # (B, 64, T)
                mu, logvar = vae.encode(dac_z)
                x1 = vae.reparameterize(mu, logvar)  # (B, 16, T)

            loss = flow_matching_loss(dit, x1, clap_embeds, CFG.cfg_dropout)

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
            for wavs, clap_embeds in val_loader:
                wavs = wavs.to(DEVICE)
                clap_embeds = clap_embeds.to(DEVICE)
                dac_z = encode_to_dac_latent(wavs)
                mu, logvar = vae.encode(dac_z)
                x1 = vae.reparameterize(mu, logvar)
                loss = flow_matching_loss(dit, x1, clap_embeds, cfg_dropout=0.0)
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
