"""
vae.py
------
1D-CNN VAE that compresses DAC encoder latents (64 ch × 130 frames)
into a smaller continuous latent (16 ch × 130 frames) for the DiT.

Flow:
  waveform → [DAC encoder] → dac_z (64×130) → [VAE encoder] → mu, logvar
  z = mu + eps*std  →  [VAE decoder] → dac_z_hat (64×130)
  dac_z_hat → [DAC decoder] → waveform_hat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from drum_generator.config import CFG


class ResBlock1d(nn.Module):
    """Simple residual block for 1D conv."""

    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, ch),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class VAEEncoder(nn.Module):
    """
    Input:  (B, dac_latent_dim=64, T=130)
    Output: mu (B, vae_latent_dim=16, T), logvar (B, 16, T)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(CFG.dac_latent_dim, CFG.vae_hidden, 3, padding=1),
            nn.SiLU(),
            ResBlock1d(CFG.vae_hidden),
            ResBlock1d(CFG.vae_hidden),
            nn.Conv1d(CFG.vae_hidden, CFG.vae_hidden // 2, 3, padding=1),
            nn.SiLU(),
            ResBlock1d(CFG.vae_hidden // 2),
        )
        mid = CFG.vae_hidden // 2
        self.mu_head = nn.Conv1d(mid, CFG.vae_latent_dim, 1)
        self.logvar_head = nn.Conv1d(mid, CFG.vae_latent_dim, 1)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-10, 2)
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    Input:  (B, vae_latent_dim=16, T=130)
    Output: (B, dac_latent_dim=64, T)
    """

    def __init__(self):
        super().__init__()
        mid = CFG.vae_hidden // 2
        self.net = nn.Sequential(
            nn.Conv1d(CFG.vae_latent_dim, mid, 3, padding=1),
            nn.SiLU(),
            ResBlock1d(mid),
            ResBlock1d(mid),
            nn.Conv1d(mid, CFG.vae_hidden, 3, padding=1),
            nn.SiLU(),
            ResBlock1d(CFG.vae_hidden),
            ResBlock1d(CFG.vae_hidden),
            nn.Conv1d(CFG.vae_hidden, CFG.dac_latent_dim, 1),
        )

    def forward(self, z):
        return self.net(z)


class DrumVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VAEEncoder()
        self.decoder = VAEDecoder()

    def encode(self, dac_z) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(dac_z)

    def decode(self, z) -> torch.Tensor:
        return self.decoder(z)

    def reparameterize(self, mu, logvar) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, dac_z):
        mu, logvar = self.encode(dac_z)
        z = self.reparameterize(mu, logvar)
        dac_z_hat = self.decode(z)
        return dac_z_hat, mu, logvar, z


def vae_loss(dac_z_hat, dac_z, mu, logvar, kl_weight: float = 1e-4):
    recon = F.mse_loss(dac_z_hat, dac_z)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    return recon + kl_weight * kl, recon, kl
