"""
vae.py
------
1D-CNN VAE that compresses DAC encoder latents (1024 ch × 129 frames)
into a smaller continuous latent (16 ch × 129 frames) for the DiT.

Gradual channel compression: 1024 → 512 → 256 → 128 → 16
Each stage uses 3 residual blocks for richer feature learning.

Flow:
  waveform → [DAC encoder] → dac_z (1024×129) → [VAE encoder] → mu, logvar
  z = mu + eps*std  →  [VAE decoder] → dac_z_hat (1024×129)
  dac_z_hat → [DAC decoder] → waveform_hat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from drum_generator.config import CFG

BLOCKS_PER_STAGE = 3


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


def _res_stack(ch: int, n: int = BLOCKS_PER_STAGE) -> nn.Sequential:
    """Stack of N residual blocks at the same channel width."""
    return nn.Sequential(*[ResBlock1d(ch) for _ in range(n)])


class VAEEncoder(nn.Module):
    """
    Gradual channel compression: 1024 → 512 → 256 → 128 → 16
    3 residual blocks per stage.
    Input:  (B, 1024, T=129)
    Output: mu (B, 16, T), logvar (B, 16, T)
    """

    def __init__(self):
        super().__init__()
        H = CFG.vae_hidden  # 512

        self.net = nn.Sequential(
            # 1024 → 512
            nn.Conv1d(CFG.dac_latent_dim, H, 3, padding=1),
            nn.SiLU(),
            _res_stack(H),
            # 512 → 256
            nn.Conv1d(H, H // 2, 3, padding=1),
            nn.SiLU(),
            _res_stack(H // 2),
            # 256 → 128
            nn.Conv1d(H // 2, H // 4, 3, padding=1),
            nn.SiLU(),
            _res_stack(H // 4),
        )
        mid = H // 4  # 128
        self.mu_head = nn.Conv1d(mid, CFG.vae_latent_dim, 1)
        self.logvar_head = nn.Conv1d(mid, CFG.vae_latent_dim, 1)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-10, 2)
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    Mirror of encoder: 16 → 128 → 256 → 512 → 1024
    3 residual blocks per stage.
    Input:  (B, 16, T=129)
    Output: (B, 1024, T)
    """

    def __init__(self):
        super().__init__()
        H = CFG.vae_hidden  # 512

        self.net = nn.Sequential(
            # 16 → 128
            nn.Conv1d(CFG.vae_latent_dim, H // 4, 3, padding=1),
            nn.SiLU(),
            _res_stack(H // 4),
            # 128 → 256
            nn.Conv1d(H // 4, H // 2, 3, padding=1),
            nn.SiLU(),
            _res_stack(H // 2),
            # 256 → 512
            nn.Conv1d(H // 2, H, 3, padding=1),
            nn.SiLU(),
            _res_stack(H),
            # 512 → 1024
            nn.Conv1d(H, CFG.dac_latent_dim, 1),
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
