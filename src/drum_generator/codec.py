"""
codec.py
--------
Shared DAC (Descript Audio Codec) encode/decode helpers.
Lazy-loads the DAC model to avoid import cost when not needed.
"""

import torch

# ---------------------------------------------------------------------------
# Lazy DAC model loading
# ---------------------------------------------------------------------------

_dac_model = None
_dac_device = None


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dac(device: str | None = None):
    """Load DAC 44kHz model (singleton, lazy-loaded)."""
    global _dac_model, _dac_device
    if device is None:
        device = _default_device()
    if _dac_model is None:
        import dac

        _dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
        _dac_model = _dac_model.to(device).eval()
        _dac_device = device
    elif _dac_device != device:
        _dac_model = _dac_model.to(device)
        _dac_device = device
    return _dac_model


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
    with torch.no_grad():
        wav = waveform.unsqueeze(1).to(device)  # (B, 1, N)
        z, _, _, _, _ = dac_model.encode(wav)  # continuous encoder output
    return z  # (B, 1024, T)


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
    import contextlib
    if device is None:
        device = _default_device()
    dac_model = get_dac(device)
    ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
    with ctx:
        waveform = dac_model.decode(dac_z.to(device))  # (B, 1, N_SAMPLES)
    return waveform.squeeze(1)  # (B, N_SAMPLES)
