"""
codec.py
--------
Shared DAC (Descript Audio Codec) encode/decode helpers.
Lazy-loads the DAC model to avoid import cost when not needed.

Optional inference-time optimizations (configured via set_dac_optim, typically
called from train.py after parsing CLI flags):

    set_dac_optim(bf16=True, compile=True)

Weight norm is ALWAYS fused out of the model after load (strictly equivalent
math, removes per-forward reparameterization overhead). bf16 autocast and
torch.compile are opt-in because they have higher risk profiles.
"""

import contextlib

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Lazy DAC model loading + inference optimizations
# ---------------------------------------------------------------------------

_dac_model = None
_dac_device = None

# Opt-in optimizations. Set via set_dac_optim() before the first call.
_dac_bf16 = False
_dac_compile = False


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_dac_optim(bf16: bool | None = None, compile: bool | None = None) -> None:
    """Configure opt-in DAC inference optimizations.

    Must be called BEFORE the first get_dac() / encode / decode to take
    effect on compile (compile wraps the model at load time). bf16 can be
    toggled any time since it's applied per-call.
    """
    global _dac_bf16, _dac_compile
    if bf16 is not None:
        _dac_bf16 = bool(bf16)
    if compile is not None:
        _dac_compile = bool(compile)


def _strip_weight_norm(model: nn.Module) -> int:
    """Fuse weight norm reparameterization (g, v) into a single weight tensor.

    Weight norm is a training-time reparameterization: w = g * v / ||v||. At
    inference time (or when parameters are frozen, as with our DAC usage) the
    reparameterization can be collapsed into a cached weight with no change
    to the forward output. Removing it eliminates per-forward normalization
    overhead from every conv in the DAC encoder/decoder.

    This is safe even when gradients flow BACK through the DAC decoder —
    autograd treats the fused `weight` parameter identically to the
    reparameterized version for backward computation.

    Returns the number of modules that had weight norm removed.
    """
    from torch.nn.utils import remove_weight_norm
    n = 0
    for module in model.modules():
        try:
            remove_weight_norm(module)
            n += 1
        except ValueError:
            # Module didn't have weight norm registered. Not an error.
            pass
    return n


def get_dac(device: str | None = None):
    """Load DAC 44 kHz model (singleton, lazy-loaded)."""
    global _dac_model, _dac_device
    if device is None:
        device = _default_device()
    if _dac_model is None:
        import dac

        model = dac.DAC.load(dac.utils.download(model_type="44khz"))
        model = model.to(device).eval()

        # Always fuse weight norm — strictly equivalent math, removes the
        # reparameterization overhead from every conv forward pass.
        n = _strip_weight_norm(model)
        print(f"[dac] loaded 44kHz model, stripped weight_norm from {n} modules")

        # Opt-in torch.compile on the decode method. We use mode="default"
        # rather than "reduce-overhead" because the latter uses CUDAGraphs
        # that reuse output buffers across calls — which corrupts gradient
        # backward when the training loop invokes decode twice per step
        # (target + recon). default mode still does kernel fusion and is
        # compatible with autograd-through-compiled-functions. Compile
        # warm-up adds ~5-10 s to the first decode call.
        if _dac_compile:
            try:
                model.decode = torch.compile(
                    model.decode,
                    mode="default",
                    fullgraph=True,
                )
                print("[dac] torch.compile enabled on decode (mode=default)")
            except Exception as e:
                print(f"[dac] torch.compile unavailable: {e}")

        _dac_model = model
        _dac_device = device
    elif _dac_device != device:
        _dac_model = _dac_model.to(device)
        _dac_device = device
    return _dac_model


def _autocast_ctx():
    """Optional bf16 autocast context for DAC forward passes.

    bf16 is chosen over fp16 because it has the same dynamic range as fp32
    (just fewer mantissa bits). fp16 is prone to gradient underflow on the
    backward path through the DAC decoder, which would silently corrupt
    VAE training.
    """
    if _dac_bf16:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def encode_to_dac_latent(waveform: torch.Tensor, device: str | None = None) -> torch.Tensor:
    """Encode waveforms to continuous DAC latents (pre-quantization).

    Args:
        waveform: (B, N_SAMPLES) float32
        device: target device (auto-detects CUDA if None)

    Returns:
        (B, 1024, ~129) continuous latent
    """
    if device is None:
        device = _default_device()
    dac_model = get_dac(device)
    with torch.no_grad(), _autocast_ctx():
        wav = waveform.unsqueeze(1).to(device)  # (B, 1, N)
        z, _, _, _, _ = dac_model.encode(wav)  # continuous encoder output
    return z.float()  # back to fp32 for VAE


def decode_from_dac_latent(
    dac_z: torch.Tensor,
    device: str | None = None,
    no_grad: bool = True,
) -> torch.Tensor:
    """Decode continuous DAC latents back to waveforms.

    Args:
        dac_z: (B, 1024, T) continuous DAC latent
        device: target device (auto-detects CUDA if None)
        no_grad: if True (default), wraps decode in torch.no_grad() — set
            False when you need gradients to flow through the DAC decoder,
            e.g., VAE training with an STFT loss computed in waveform space.

    Returns:
        (B, N_SAMPLES) float32 waveform
    """
    if device is None:
        device = _default_device()
    dac_model = get_dac(device)
    grad_ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
    with grad_ctx, _autocast_ctx():
        waveform = dac_model.decode(dac_z.to(device))  # (B, 1, N_SAMPLES)
    # .clone() is required when --dac-compile is on with mode=reduce-overhead:
    # CUDAGraphs reuse output buffers across calls, which corrupts pending
    # backward computations if the training loop invokes decode twice per
    # step (target + recon). Cloning breaks the buffer sharing at ~8 MB
    # extra memcpy per call — negligible cost for correctness.
    # .float() brings bf16 autocast output back to fp32 for downstream losses.
    return waveform.squeeze(1).float().clone()  # (B, N_SAMPLES)
