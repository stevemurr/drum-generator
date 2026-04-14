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
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from drum_generator.codec import decode_from_dac_latent, encode_to_dac_latent, set_dac_optim
from drum_generator.config import CFG
from drum_generator.dataset import build_dataset
from drum_generator.dit import DrumDiT, flow_matching_loss
from drum_generator.vae import DrumVAE, vae_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _save_ckpt(
    path: str,
    *,
    phase: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_val: float,
) -> None:
    """Atomic dict-format checkpoint write: tmp file + rename.

    Deliberately excludes RNG state — resume is "approximately continue",
    not bit-exact, and keeping the dict to torch/python primitives lets
    torch.load use its safer weights_only=True default.
    """
    ckpt = {
        "phase": phase,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def _unwrap_model_state(ckpt) -> dict:
    """Return a flat model state_dict from either format.

    New format: dict with a "model" key. Legacy format: flat state_dict.
    """
    if isinstance(ckpt, dict) and "model" in ckpt and "phase" in ckpt:
        return ckpt["model"]
    return ckpt


# ---------------------------------------------------------------------------
# Phase 1: Train VAE
# ---------------------------------------------------------------------------


def _unpack_batch(batch, device):
    """Normalize DataLoader batch into (audio, clap, wav_pre).

    MemmapDACDataset returns 3-tuples (dac_z, clap, wav_true_or_sentinel).
    CachedDACDataset / DiskAudioDataset / others return 2-tuples (dac_z, clap).
    This helper handles both, returning an empty-tensor sentinel for wav_pre
    when the dataset doesn't provide pre-decoded waveforms.
    """
    audio = batch[0].to(device)
    clap = batch[1].to(device)
    if len(batch) > 2 and batch[2].numel() > 0:
        wav_pre = batch[2].to(device)
    else:
        wav_pre = torch.empty(0, device=device)
    return audio, clap, wav_pre


class WeightedMRSTFTLoss(nn.Module):
    """Multi-resolution magnitude STFT loss with per-frequency-bin weighting.

    At each FFT resolution, computes log-magnitude L1 and spectral
    convergence, but multiplies each bin's error by a frequency-dependent
    weight before reducing. A weight of 0 effectively masks a bin out.

    This is a drop-in replacement for auraloss's MultiResolutionSTFTLoss
    on the linear-frequency path, exposing the bin-weight axis that
    auraloss doesn't. Mel remains handled by auraloss separately.
    """

    def __init__(
        self,
        fft_sizes: list[int],
        hop_sizes: list[int],
        win_lengths: list[int],
        sample_rate: int,
        weight_fn,  # callable: (n_bins, sample_rate) -> (n_bins,) tensor
    ):
        super().__init__()
        self.fft_sizes = list(fft_sizes)
        self.hop_sizes = list(hop_sizes)
        self.win_lengths = list(win_lengths)
        for i, (n_fft, win_len) in enumerate(zip(fft_sizes, win_lengths)):
            self.register_buffer(f"window_{i}", torch.hann_window(win_len))
            w = weight_fn(n_bins=n_fft // 2 + 1, sample_rate=sample_rate)
            self.register_buffer(f"bin_weights_{i}", w)

    def _mag(self, x: torch.Tensor, i: int) -> torch.Tensor:
        return torch.stft(
            x,
            n_fft=self.fft_sizes[i],
            hop_length=self.hop_sizes[i],
            win_length=self.win_lengths[i],
            window=getattr(self, f"window_{i}"),
            return_complex=True,
        ).abs()

    def forward(self, wav_hat: torch.Tensor, wav_true: torch.Tensor) -> torch.Tensor:
        # auraloss convention: (B, 1, T). Squeeze the channel dim for torch.stft.
        if wav_hat.dim() == 3:
            wav_hat = wav_hat.squeeze(1)
        if wav_true.dim() == 3:
            wav_true = wav_true.squeeze(1)

        losses = []
        for i in range(len(self.fft_sizes)):
            weights = getattr(self, f"bin_weights_{i}").view(1, -1, 1)
            mag_true = self._mag(wav_true, i)
            mag_hat = self._mag(wav_hat, i)

            log_true = torch.log(mag_true + 1e-7)
            log_hat = torch.log(mag_hat + 1e-7)
            log_mag = ((log_hat - log_true).abs() * weights).mean()

            diff = (mag_true - mag_hat) * weights
            ref = mag_true * weights
            sc = diff.norm() / ref.norm().clamp(min=1e-7)

            losses.append(log_mag + sc)
        return torch.stack(losses).mean()


def drum_weight_curve(n_bins: int, sample_rate: int) -> torch.Tensor:
    """Per-bin frequency weights tuned for drum one-shots.

    - 0x      below 20 Hz    (DC / sub-rumble — not drum content)
    - 3x      40-320 Hz      (kick fundamental + first harmonics)
    - 1x      320 Hz-15 kHz  (body, mid, attack, high detail)
    - 1.0→0.1 above 15 kHz   (smooth rolloff — near noise floor)
    """
    freqs = torch.linspace(0, sample_rate / 2, n_bins)
    weights = torch.ones(n_bins)

    weights[freqs < 20] = 0.0

    kick_band = (freqs >= 40) & (freqs <= 320)
    weights[kick_band] = 3.0

    rolloff = freqs > 15000
    roll_frac = (freqs[rolloff] - 15000) / (sample_rate / 2 - 15000)
    weights[rolloff] = 1.0 - 0.9 * roll_frac

    return weights


def _make_lowpass_fir(cutoff_hz: float, sample_rate: int, num_taps: int = 257) -> torch.Tensor:
    """Linear-phase windowed-sinc FIR lowpass kernel.

    Linear phase means a fixed group delay of (num_taps - 1) / 2 samples,
    which cancels identically between wav_hat and wav_true inside an L1
    comparison — so the filter is transparent to the loss value.
    """
    nyquist = sample_rate / 2
    cutoff_norm = cutoff_hz / nyquist

    t = torch.arange(num_taps, dtype=torch.float32) - (num_taps - 1) / 2
    pi_t = torch.pi * t
    sinc = torch.where(
        t == 0,
        torch.tensor(cutoff_norm),
        torch.sin(cutoff_norm * pi_t) / pi_t.clamp(min=1e-12),
    )
    kernel = sinc * torch.hann_window(num_taps)
    return kernel / kernel.sum()


class LowpassL1Loss(nn.Module):
    """L1 on a linear-phase-lowpassed version of both waveforms.

    Targets sub-cutoff frequency content (kick body, snare fundamentals)
    as a direct sample-space objective, without inheriting the
    phase-sensitivity of full-band waveform L1 — low-frequency content
    is smooth over many samples, so small transient phase drift doesn't
    produce large error.
    """

    def __init__(self, cutoff_hz: float = 500.0, sample_rate: int = 44100, num_taps: int = 257):
        super().__init__()
        kernel = _make_lowpass_fir(cutoff_hz, sample_rate, num_taps)
        self.register_buffer("kernel", kernel.view(1, 1, -1))
        self.pad = (num_taps - 1) // 2

    def forward(self, wav_hat: torch.Tensor, wav_true: torch.Tensor) -> torch.Tensor:
        # wav_* shape: (B, T) — add channel dim, reflect-pad, conv, compare.
        wh = F.pad(wav_hat.unsqueeze(1), (self.pad, self.pad), mode="reflect")
        wt = F.pad(wav_true.unsqueeze(1), (self.pad, self.pad), mode="reflect")
        wh_lp = F.conv1d(wh, self.kernel)
        wt_lp = F.conv1d(wt, self.kernel)
        return F.l1_loss(wh_lp, wt_lp)


def _build_stft_losses():
    """Build (linear_mrstft, mel_mrstft, lowpass_l1) triple based on CFG.

    FFT sizes tuned for drums at 44.1 kHz: 4096 resolves sub-bass (~10.7 Hz
    bin — kick fundamentals at 50–80 Hz land in bins 5–8 with clear
    separation), 2048 low-body (~21 Hz bin), 1024 mid detail, 512 catches
    transients (~11 ms window).

    - linear MRSTFT: magnitude + phase term (w_phs = CFG.vae_stft_phase_weight).
      When CFG.vae_stft_weighted is True, uses WeightedMRSTFTLoss with the
      drum_weight_curve instead of auraloss — same FFT sizes, but per-bin
      frequency weighting that boosts the kick band and masks DC / noise
      floor. Phase term is not available on the weighted path.
    - mel MRSTFT: mel-scaled magnitude loss that compresses narrow
      high-frequency spikes into perceptually wider bins, reducing the
      optimizer's incentive to "pay off" magnitude targets with ringy tones.
    - lowpass L1: linear-phase FIR lowpass of both waveforms, then L1.
      Direct sample-space target on sub-cutoff content (kick body).

    Any component may be None if its weight is 0.
    """
    from auraloss.freq import MultiResolutionSTFTLoss

    fft = [4096, 2048, 1024, 512]
    hop = [1024, 512, 256, 128]
    win = [4096, 2048, 1024, 512]

    linear = None
    if CFG.vae_stft_weight > 0:
        if CFG.vae_stft_weighted:
            linear = WeightedMRSTFTLoss(
                fft_sizes=fft,
                hop_sizes=hop,
                win_lengths=win,
                sample_rate=CFG.sample_rate,
                weight_fn=drum_weight_curve,
            ).to(DEVICE)
        else:
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

    lowpass = None
    if CFG.vae_lowpass_weight > 0:
        lowpass = LowpassL1Loss(
            cutoff_hz=CFG.vae_lowpass_cutoff,
            sample_rate=CFG.sample_rate,
        ).to(DEVICE)

    return linear, mel, lowpass


def train_vae(train_loader, val_loader, resume_state: dict | None = None):
    vae = DrumVAE().to(DEVICE)
    opt = torch.optim.AdamW(vae.parameters(), lr=CFG.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, CFG.vae_epochs, eta_min=CFG.vae_eta_min
    )

    start_epoch = 1
    best_val = float("inf")
    if resume_state is not None:
        vae.load_state_dict(resume_state["model"])
        opt.load_state_dict(resume_state["optimizer"])
        sch.load_state_dict(resume_state["scheduler"])
        start_epoch = resume_state["epoch"] + 1
        best_val = resume_state["best_val"]
        print(
            f"[vae] resumed from epoch {resume_state['epoch']} "
            f"(best_val={best_val:.5f}), continuing at epoch {start_epoch}"
        )

    stft_linear, stft_mel, lowpass = _build_stft_losses()
    waveform_losses_enabled = (
        stft_linear is not None or stft_mel is not None or lowpass is not None
    )
    if waveform_losses_enabled:
        parts = []
        if stft_linear is not None:
            kind = "linear-weighted" if CFG.vae_stft_weighted else "linear"
            part = f"{kind}(w={CFG.vae_stft_weight}"
            if not CFG.vae_stft_weighted and CFG.vae_stft_phase_weight > 0:
                part += f", w_phs={CFG.vae_stft_phase_weight}"
            part += ")"
            parts.append(part)
        if stft_mel is not None:
            parts.append(f"mel(w={CFG.vae_stft_mel_weight})")
        if lowpass is not None:
            parts.append(
                f"lowpass-L1(w={CFG.vae_lowpass_weight}, "
                f"cutoff={CFG.vae_lowpass_cutoff:.0f}Hz)"
            )
        print(f"[vae] waveform-space losses enabled: {' + '.join(parts)}")

    def apply_stft(loss, dac_z, recon_z, wav_true_pre, no_grad_recon: bool):
        """Adds waveform-space losses (linear STFT + mel STFT + lowpass L1,
        as configured) to `loss`.

        If `wav_true_pre` is a non-empty tensor (shape (B, T) > 0 samples),
        it's used directly as the target waveform and the target-path DAC
        decode is skipped — the pre-decoded path, ~40% faster per step.
        Otherwise the target waveform is produced live via DAC decode.

        When `no_grad_recon` is True, the recon-path DAC decode skips
        gradient tracking (used during validation). Otherwise gradients
        flow through DAC back into the VAE (used during training).
        """
        if not waveform_losses_enabled:
            return loss
        if wav_true_pre is not None and wav_true_pre.numel() > 0:
            wav_true = wav_true_pre
        else:
            wav_true = decode_from_dac_latent(dac_z, DEVICE, no_grad=True)
        wav_hat = decode_from_dac_latent(recon_z, DEVICE, no_grad=no_grad_recon)
        wh = wav_hat.unsqueeze(1)
        wt = wav_true.unsqueeze(1)
        if stft_linear is not None:
            loss = loss + CFG.vae_stft_weight * stft_linear(wh, wt)
        if stft_mel is not None:
            loss = loss + CFG.vae_stft_mel_weight * stft_mel(wh, wt)
        if lowpass is not None:
            loss = loss + CFG.vae_lowpass_weight * lowpass(wav_hat, wav_true)
        return loss

    for epoch in range(start_epoch, CFG.vae_epochs + 1):
        epoch_start = time.time()
        # --- train ---
        vae.train()
        train_loss = 0.0
        for batch in train_loader:  # ignore CLAP for VAE phase
            audio, _clap, wav_true_pre = _unpack_batch(batch, DEVICE)
            # Detect cached DAC latents (3D) vs raw waveforms (2D)
            dac_z = audio if audio.dim() == 3 else encode_to_dac_latent(audio, DEVICE)

            recon_z, mu, logvar, _ = vae(dac_z)

            # KL warmup: ramp weight from 0 → target over warmup period
            kl_w = min(CFG.vae_kl_weight, CFG.vae_kl_weight * epoch / CFG.vae_kl_warmup)
            loss, recon, kl = vae_loss(recon_z, dac_z, mu, logvar, kl_w)

            loss = apply_stft(loss, dac_z, recon_z, wav_true_pre, no_grad_recon=False)

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
            for batch in val_loader:
                audio, _clap, wav_true_pre = _unpack_batch(batch, DEVICE)
                dac_z = audio if audio.dim() == 3 else encode_to_dac_latent(audio, DEVICE)
                recon_z, mu, logvar, _ = vae(dac_z)
                loss, _, _ = vae_loss(recon_z, dac_z, mu, logvar, CFG.vae_kl_weight)
                loss = apply_stft(loss, dac_z, recon_z, wav_true_pre, no_grad_recon=True)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        epoch_time = time.time() - epoch_start
        print(
            f"VAE epoch {epoch:3d}/{CFG.vae_epochs} | "
            f"train {train_loss:.5f} | val {val_loss:.5f} | "
            f"time {epoch_time:.1f}s"
        )

        _save_ckpt(
            f"{CFG.ckpt_dir}/vae_last.pt",
            phase="vae",
            model=vae,
            optimizer=opt,
            scheduler=sch,
            epoch=epoch,
            best_val=best_val,
        )

        if val_loss < best_val:
            best_val = val_loss
            _save_ckpt(
                f"{CFG.ckpt_dir}/vae_best.pt",
                phase="vae",
                model=vae,
                optimizer=opt,
                scheduler=sch,
                epoch=epoch,
                best_val=best_val,
            )
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


def train_dit(train_loader, val_loader, vae: DrumVAE, resume_state: dict | None = None):
    dit = DrumDiT().to(DEVICE)
    opt = torch.optim.AdamW(dit.parameters(), lr=CFG.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG.dit_epochs)

    vae.eval()
    start_epoch = 1
    best_val = float("inf")
    if resume_state is not None:
        dit.load_state_dict(resume_state["model"])
        opt.load_state_dict(resume_state["optimizer"])
        sch.load_state_dict(resume_state["scheduler"])
        start_epoch = resume_state["epoch"] + 1
        best_val = resume_state["best_val"]
        print(
            f"[dit] resumed from epoch {resume_state['epoch']} "
            f"(best_val={best_val:.5f}), continuing at epoch {start_epoch}"
        )

    for epoch in range(start_epoch, CFG.dit_epochs + 1):
        epoch_start = time.time()
        # --- train ---
        dit.train()
        train_loss = 0.0
        for batch in train_loader:
            audio, clap_embeds, _wav = _unpack_batch(batch, DEVICE)

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
            for batch in val_loader:
                audio, clap_embeds, _wav = _unpack_batch(batch, DEVICE)
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
        epoch_time = time.time() - epoch_start
        print(
            f"DiT epoch {epoch:3d}/{CFG.dit_epochs} | "
            f"train {train_loss:.5f} | val {val_loss:.5f} | "
            f"time {epoch_time:.1f}s"
        )

        _save_ckpt(
            f"{CFG.ckpt_dir}/dit_last.pt",
            phase="dit",
            model=dit,
            optimizer=opt,
            scheduler=sch,
            epoch=epoch,
            best_val=best_val,
        )

        if val_loss < best_val:
            best_val = val_loss
            _save_ckpt(
                f"{CFG.ckpt_dir}/dit_best.pt",
                phase="dit",
                model=dit,
                optimizer=opt,
                scheduler=sch,
                epoch=epoch,
                best_val=best_val,
            )
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
        "--vae-eta-min",
        type=float,
        default=None,
        help=f"CosineAnnealingLR floor for VAE training. The default (0) "
             f"means the LR decays all the way to 0 at the last epoch, so "
             f"late epochs stop learning. A small nonzero value (e.g. 1e-6, "
             f"which is 1%% of the default --lr=1e-4) keeps the tail "
             f"productive. Safe range: 0 to --lr/10. "
             f"Default: CFG.vae_eta_min ({CFG.vae_eta_min}).",
    )
    parser.add_argument(
        "--vae-ckpt",
        default=None,
        help="Path to a trained VAE checkpoint to load for the DiT phase. "
             "Overrides the default '{ckpt_dir}/vae_best.pt' lookup. "
             "vae_latent_dim is auto-detected from the checkpoint, so you "
             "don't need to pass --vae-latent-dim. Use this to run a DiT "
             "training pass with a VAE checkpoint from a different ckpt_dir.",
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
        "--vae-stft-weighted",
        action="store_true",
        help="Use a per-frequency-bin weighted linear MRSTFT (drum weight "
             "curve: 0x below 20 Hz, 3x in 40-320 Hz kick band, 1x mids, "
             "rolloff above 15 kHz) in place of auraloss's uniform-weight "
             "linear MRSTFT. Targets the specific failure mode where linear "
             "STFT spends most of its ~2048 bins on high frequencies and "
             "under-weights the ~10 bins where kick fundamentals live. The "
             "outer --vae-stft-weight coefficient still applies on top. "
             "Phase term is not available on the weighted path. Off by "
             "default.",
    )
    parser.add_argument(
        "--vae-lowpass-weight",
        type=float,
        default=None,
        help=f"Weight for an L1 loss on linear-phase-lowpassed waveforms. "
             f"Direct sample-space target on sub-cutoff content (kick body, "
             f"snare fundamentals) — picks up where STFT losses are "
             f"proportionally weakest. Waveform L1 at low frequencies is "
             f"robust to small transient phase drift because low-frequency "
             f"content is smooth over many samples. 0 disables. Typical "
             f"values 0.1-0.3. "
             f"Default: CFG.vae_lowpass_weight ({CFG.vae_lowpass_weight}).",
    )
    parser.add_argument(
        "--vae-lowpass-cutoff",
        type=float,
        default=None,
        help=f"Cutoff frequency in Hz for --vae-lowpass-weight. 500 Hz "
             f"captures kick fundamentals (50-100 Hz), first harmonics, "
             f"and snare body. Higher values include tom fundamentals. "
             f"Default: CFG.vae_lowpass_cutoff ({CFG.vae_lowpass_cutoff}).",
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
    parser.add_argument(
        "--dac-bf16",
        action="store_true",
        help="Run DAC forward passes under torch.autocast bf16. Cuts DAC "
             "decode memory ~2x and speeds it up ~1.3-1.8x on GB10. Safe for "
             "gradient flow through DAC (bf16 has fp32 dynamic range). "
             "Off by default.",
    )
    parser.add_argument(
        "--resume-ckpt",
        default=None,
        help="Path to a checkpoint (vae_last.pt / vae_best.pt / dit_last.pt / "
             "dit_best.pt) to resume training from. Restores model, optimizer, "
             "scheduler, epoch counter, and best_val. The "
             "checkpoint's phase must match --phase, and its scheduler T_max "
             "must match the current --{phase}-epochs value.",
    )
    parser.add_argument(
        "--dac-compile",
        action="store_true",
        help="Wrap DAC's decode method with torch.compile(mode='reduce-overhead'). "
             "Can give 1.5-2x speedup on the decode forward/backward. Adds "
             "~30-90s to first-decode latency (compile warmup). GB10 SM_121 "
             "support in TorchInductor is variable — if it errors, disable. "
             "Off by default.",
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

    if args.vae_eta_min is not None:
        CFG.vae_eta_min = args.vae_eta_min
        print(f"[train] vae_eta_min: {args.vae_eta_min}")

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

    if args.vae_stft_weighted:
        CFG.vae_stft_weighted = True
        print("[train] vae_stft_weighted: True (drum weight curve)")

    if args.vae_lowpass_weight is not None:
        CFG.vae_lowpass_weight = args.vae_lowpass_weight
        print(f"[train] vae_lowpass_weight: {args.vae_lowpass_weight}")

    if args.vae_lowpass_cutoff is not None:
        CFG.vae_lowpass_cutoff = args.vae_lowpass_cutoff
        print(f"[train] vae_lowpass_cutoff: {args.vae_lowpass_cutoff}")

    # DAC inference optimizations. Weight-norm fusion always on (configured
    # inside codec.py at DAC load time). bf16 autocast and torch.compile are
    # opt-in because they have higher risk profiles on GB10.
    if args.dac_bf16 or args.dac_compile:
        set_dac_optim(bf16=args.dac_bf16, compile=args.dac_compile)
        print(f"[train] dac optim: bf16={args.dac_bf16}  compile={args.dac_compile}")

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

    # Load resume checkpoint once (if any). It applies to whichever phase's
    # loop starts first; subsequent phases in a --phase both run start fresh.
    resume_vae = None
    resume_dit = None
    if args.resume_ckpt is not None:
        ckpt = torch.load(args.resume_ckpt, map_location=DEVICE)
        if not (isinstance(ckpt, dict) and "phase" in ckpt):
            raise ValueError(
                f"--resume-ckpt {args.resume_ckpt}: not a resumable "
                f"checkpoint (missing 'phase' field; likely a legacy "
                f"state-dict-only file)"
            )
        ckpt_phase = ckpt["phase"]
        if args.phase not in (ckpt_phase, "both"):
            raise ValueError(
                f"--resume-ckpt phase mismatch: checkpoint is '{ckpt_phase}' "
                f"but --phase is '{args.phase}'"
            )
        expected_T = CFG.vae_epochs if ckpt_phase == "vae" else CFG.dit_epochs
        sched_T = ckpt["scheduler"].get("T_max")
        if sched_T != expected_T:
            raise ValueError(
                f"--resume-ckpt scheduler T_max mismatch: checkpoint has "
                f"T_max={sched_T} but current --{ckpt_phase}-epochs is "
                f"{expected_T}. Rerun with --{ckpt_phase}-epochs {sched_T} "
                f"to resume, or start a fresh run."
            )
        if ckpt_phase == "vae":
            resume_vae = ckpt
        else:
            resume_dit = ckpt
        print(
            f"[train] resume: {args.resume_ckpt} "
            f"(phase={ckpt_phase}, epoch={ckpt['epoch']})"
        )

    if args.phase in ("vae", "both"):
        print("\n=== Phase 1: VAE training ===")
        vae = train_vae(train_loader, val_loader, resume_state=resume_vae)

    if args.phase in ("dit", "both"):
        print("\n=== Phase 2: DiT training ===")
        vae_ckpt_path = args.vae_ckpt or f"{CFG.ckpt_dir}/vae_best.pt"
        print(f"[dit] loading VAE from: {vae_ckpt_path}")

        # Load first, auto-detect vae_latent_dim from the mu_head shape, set
        # CFG so DrumVAE *and* DrumDiT (which uses CFG.vae_latent_dim at
        # construction time) both match the checkpoint.
        vae_state = _unwrap_model_state(
            torch.load(vae_ckpt_path, map_location=DEVICE, weights_only=False)
        )
        mu_head = vae_state.get("encoder.mu_head.weight")
        if mu_head is not None:
            ckpt_latent_dim = mu_head.shape[0]
            if ckpt_latent_dim != CFG.vae_latent_dim:
                print(
                    f"[dit] auto-detected vae_latent_dim={ckpt_latent_dim} "
                    f"from checkpoint (was {CFG.vae_latent_dim})"
                )
                CFG.vae_latent_dim = ckpt_latent_dim

        vae = DrumVAE().to(DEVICE)
        vae.load_state_dict(vae_state)
        train_dit(train_loader, val_loader, vae, resume_state=resume_dit)


if __name__ == "__main__":
    main()
