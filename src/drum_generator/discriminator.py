"""
discriminator.py
----------------
Multi-Period Discriminator (MPD) and Multi-Scale Discriminator (MSD)
for adversarial training of the DiT. Adapted from HiFi-GAN / BigVGAN.

MPD reshapes the waveform into 2D at different periods and uses 2D convs
to detect periodic artifacts. MSD operates at multiple temporal scales
(1x, 2x-downsampled, 4x-downsampled) with 1D convs to detect
coarse-to-fine waveform artifacts.

Both discriminator families return a list of (output, [features]) pairs —
one per sub-discriminator — so the generator can compute both adversarial
loss (on the outputs) and feature-matching loss (on the intermediate
feature maps).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


# ---------------------------------------------------------------------------
# Period sub-discriminator (2D convs on folded waveform)
# ---------------------------------------------------------------------------


class PeriodSubDiscriminator(nn.Module):
    def __init__(self, period: int, channels: int = 32, max_channels: int = 512):
        super().__init__()
        self.period = period

        ch_in = 1
        layers = []
        for i in range(4):
            ch_out = min(channels * (4 ** i), max_channels)
            layers.append(
                spectral_norm(
                    nn.Conv2d(ch_in, ch_out, (5, 1), stride=(3, 1), padding=(2, 0))
                )
            )
            layers.append(nn.LeakyReLU(0.1))
            ch_in = ch_out

        layers.append(
            spectral_norm(nn.Conv2d(ch_in, ch_in, (5, 1), padding=(2, 0)))
        )
        layers.append(nn.LeakyReLU(0.1))
        layers.append(spectral_norm(nn.Conv2d(ch_in, 1, (3, 1), padding=(1, 0))))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        # x: (B, 1, T)
        B, C, T = x.shape
        # Pad to make T divisible by period, then fold
        pad = (self.period - (T % self.period)) % self.period
        x = F.pad(x, (0, pad), mode="reflect")
        x = x.view(B, C, -1, self.period)  # (B, 1, T//p, p)

        features = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                features.append(x)
        return x, features


# ---------------------------------------------------------------------------
# Scale sub-discriminator (1D convs on downsampled waveform)
# ---------------------------------------------------------------------------


class ScaleSubDiscriminator(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        layers = [
            spectral_norm(nn.Conv1d(1, channels, 15, padding=7)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Conv1d(channels, channels, 41, stride=4,
                                    padding=20, groups=4)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Conv1d(channels, channels * 2, 41, stride=4,
                                    padding=20, groups=16)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Conv1d(channels * 2, channels * 4, 41, stride=4,
                                    padding=20, groups=16)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Conv1d(channels * 4, channels * 4, 5, padding=2)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Conv1d(channels * 4, 1, 3, padding=1)),
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        # x: (B, 1, T)
        features = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                features.append(x)
        return x, features


# ---------------------------------------------------------------------------
# Full discriminators
# ---------------------------------------------------------------------------


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: list[int] | None = None):
        super().__init__()
        if periods is None:
            periods = [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList(
            [PeriodSubDiscriminator(p) for p in periods]
        )

    def forward(self, x: torch.Tensor):
        # x: (B, T) or (B, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        results = []
        for d in self.discriminators:
            out, feats = d(x)
            results.append((out, feats))
        return results


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, n_scales: int = 3):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [ScaleSubDiscriminator() for _ in range(n_scales)]
        )
        self.downsample = nn.AvgPool1d(4, stride=2, padding=2)

    def forward(self, x: torch.Tensor):
        # x: (B, T) or (B, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        results = []
        for i, d in enumerate(self.discriminators):
            out, feats = d(x)
            results.append((out, feats))
            if i < len(self.discriminators) - 1:
                x = self.downsample(x)
        return results


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def discriminator_loss(
    disc_real_outputs: list[tuple],
    disc_fake_outputs: list[tuple],
) -> torch.Tensor:
    """Hinge loss for the discriminator."""
    loss = torch.tensor(0.0, device=disc_real_outputs[0][0].device)
    for (real_out, _), (fake_out, _) in zip(disc_real_outputs, disc_fake_outputs):
        loss = loss + F.relu(1.0 - real_out).mean() + F.relu(1.0 + fake_out).mean()
    return loss


def generator_loss(disc_fake_outputs: list[tuple]) -> torch.Tensor:
    """Hinge loss for the generator (wants discriminator to say 'real')."""
    loss = torch.tensor(0.0, device=disc_fake_outputs[0][0].device)
    for fake_out, _ in disc_fake_outputs:
        loss = loss - fake_out.mean()
    return loss


def feature_matching_loss(
    disc_real_outputs: list[tuple],
    disc_fake_outputs: list[tuple],
) -> torch.Tensor:
    """L1 between discriminator intermediate features on real vs generated."""
    loss = torch.tensor(0.0, device=disc_real_outputs[0][0].device)
    n_features = 0
    for (_, real_feats), (_, fake_feats) in zip(disc_real_outputs, disc_fake_outputs):
        for rf, ff in zip(real_feats, fake_feats):
            loss = loss + F.l1_loss(ff, rf.detach())
            n_features += 1
    return loss / max(n_features, 1)
