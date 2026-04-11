"""
dit.py
------
Diffusion Transformer (DiT) trained with Conditional Flow Matching (CFM).

Architecture:
  - Input: noisy VAE latent z_t (B, vae_latent_dim=16, T=130)
  - Patchify: merge patch_size=4 frames → tokens (B, T/4=32, 16*4=64)
  - N transformer blocks with AdaLN conditioning on (t_embed + clap_embed)
  - Cross-attention to CLAP text embedding
  - Linear head → velocity v (B, T/4, patch_dim) → unpatchify → (B, 16, T)

Flow matching:
  Training:  t ~ U(0,1),  x_t = (1-t)*x0 + t*x1,  target_v = x1 - x0
  Loss:      MSE(v_theta(x_t, t, c), target_v)
  Inference: Euler ODE  x_{t+dt} = x_t + v_theta * dt  from t=0→1
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from drum_generator.config import CFG

# ---------------------------------------------------------------------------
# Timestep embedding (sinusoidal → MLP)
# ---------------------------------------------------------------------------


class TimestepEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) float in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None] * freqs[None] * 1000.0
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# AdaLN (Adaptive Layer Norm)
# Conditioning vector c predicts scale (gamma) and shift (beta) dynamically.
# ---------------------------------------------------------------------------


class AdaLN(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * dim),  # → gamma, beta
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (B, T, dim),  c: (B, cond_dim)
        gamma, beta = self.proj(c).chunk(2, dim=-1)  # each (B, dim)
        return self.norm(x) * (1 + gamma[:, None]) + beta[:, None]


# ---------------------------------------------------------------------------
# DiT Block: AdaLN → self-attention → AdaLN → cross-attention → FFN
# ---------------------------------------------------------------------------


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, cond_dim: int, clap_dim: int):
        super().__init__()
        self.adaLN1 = AdaLN(dim, cond_dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.adaLN2 = AdaLN(dim, cond_dim)
        # Cross-attention to CLAP text embedding
        self.cross_attn = nn.MultiheadAttention(
            dim, heads, batch_first=True, kdim=clap_dim, vdim=clap_dim
        )
        self.clap_proj = nn.Linear(clap_dim, dim)  # project keys/values

        self.adaLN3 = AdaLN(dim, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, cond, clap_kv):
        # x:       (B, T_tokens, dim)
        # cond:    (B, cond_dim)   — t_embed + clap_embed projected
        # clap_kv: (B, 1, clap_dim) — CLAP text embed as cross-attn key/value

        # Self-attention
        h = self.adaLN1(x, cond)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out

        # Cross-attention to CLAP
        h = self.adaLN2(x, cond)
        kv = self.clap_proj(clap_kv)  # (B, 1, dim)
        ca_out, _ = self.cross_attn(h, kv, kv)
        x = x + ca_out

        # FFN
        h = self.adaLN3(x, cond)
        x = x + self.ffn(h)
        return x


# ---------------------------------------------------------------------------
# Full DiT
# ---------------------------------------------------------------------------


class DrumDiT(nn.Module):
    def __init__(self):
        super().__init__()
        C = CFG
        self.patch_size = C.dit_patch_size
        self.patch_dim = C.vae_latent_dim * C.dit_patch_size  # 16*4 = 64
        self.n_tokens = C.dac_time_frames // C.dit_patch_size  # 130//4 = 32

        # Timestep embedding → cond_dim
        cond_dim = C.dit_dim
        self.t_embed = TimestepEmbed(cond_dim)
        # Project CLAP (512) → cond_dim for AdaLN conditioning signal
        self.clap_to_cond = nn.Linear(C.clap_dim, cond_dim)

        # Patchify: (B, patch_dim, n_tokens) → (B, n_tokens, dit_dim)
        self.patch_embed = nn.Linear(self.patch_dim, C.dit_dim)

        # Positional embedding (learned)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_tokens, C.dit_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                DiTBlock(C.dit_dim, C.dit_heads, cond_dim, C.clap_dim)
                for _ in range(C.dit_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(C.dit_dim)
        self.final_proj = nn.Linear(C.dit_dim, self.patch_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def patchify(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, vae_latent_dim, T)
        B, C, T = z.shape
        z = z.reshape(B, C, self.n_tokens, self.patch_size)  # (B,C,n_tok,p)
        z = z.permute(0, 2, 1, 3)  # (B,n_tok,C,p)
        z = z.reshape(B, self.n_tokens, -1)  # (B,n_tok,patch_dim)
        return z

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, n_tokens, patch_dim)
        B, N, _ = tokens.shape
        C = CFG.vae_latent_dim
        z = tokens.reshape(B, N, C, self.patch_size)  # (B,n_tok,C,p)
        z = z.permute(0, 2, 1, 3)  # (B,C,n_tok,p)
        z = z.reshape(B, C, N * self.patch_size)  # (B,C,T)
        return z

    def forward(
        self,
        z_t: torch.Tensor,  # (B, vae_latent_dim, T) — noisy latent
        t: torch.Tensor,  # (B,) — timestep in [0,1]
        clap_embed: torch.Tensor,  # (B, clap_dim) — text conditioning
    ) -> torch.Tensor:
        # Build conditioning vector: t_embed + clap contribution
        t_emb = self.t_embed(t)  # (B, cond_dim)
        c_emb = self.clap_to_cond(clap_embed)  # (B, cond_dim)
        cond = t_emb + c_emb  # (B, cond_dim)

        # CLAP as cross-attention keys/values
        clap_kv = clap_embed.unsqueeze(1)  # (B, 1, clap_dim)

        # Patchify + embed
        x = self.patchify(z_t)  # (B, n_tokens, patch_dim)
        x = self.patch_embed(x) + self.pos_embed  # (B, n_tokens, dit_dim)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, cond, clap_kv)

        x = self.final_norm(x)
        x = self.final_proj(x)  # (B, n_tokens, patch_dim)
        v = self.unpatchify(x)  # (B, vae_latent_dim, T)
        return v


# ---------------------------------------------------------------------------
# Flow matching loss + inference
# ---------------------------------------------------------------------------


def flow_matching_loss(
    dit: DrumDiT,
    x1: torch.Tensor,  # (B, 16, T) — clean VAE latent
    clap_embed: torch.Tensor,  # (B, 512)
    cfg_dropout: float = CFG.cfg_dropout,
) -> torch.Tensor:
    B = x1.shape[0]
    device = x1.device

    # Sample noise and timestep
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=device)

    # Interpolate: straight-line path
    t_b = t.view(B, 1, 1)
    x_t = (1 - t_b) * x0 + t_b * x1  # noisy latent at time t
    v_target = x1 - x0  # constant velocity (straight line)

    # Classifier-free guidance: randomly null-out CLAP conditioning
    if cfg_dropout > 0:
        mask = (torch.rand(B, device=device) < cfg_dropout).float()
        clap_embed = clap_embed * (1 - mask[:, None])

    v_pred = dit(x_t, t, clap_embed)
    return F.mse_loss(v_pred, v_target)


@torch.no_grad()
def generate(
    dit: DrumDiT,
    clap_embed: torch.Tensor,  # (1, 512)
    steps: int = CFG.fm_steps_infer,
    cfg_scale: float = CFG.cfg_scale,
    device: str = "cpu",
) -> torch.Tensor:
    """Euler ODE from pure noise to clean VAE latent, with CFG."""
    dit.eval()
    B = clap_embed.shape[0]
    shape = (B, CFG.vae_latent_dim, CFG.dac_time_frames)
    x = torch.randn(shape, device=device)

    null_embed = torch.zeros_like(clap_embed)  # unconditioned
    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full((B,), i / steps, device=device)

        v_cond = dit(x, t, clap_embed)
        v_uncond = dit(x, t, null_embed)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)  # CFG

        x = x + v * dt

    return x  # (B, vae_latent_dim, T) — clean VAE latent
