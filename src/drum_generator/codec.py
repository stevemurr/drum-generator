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


def get_dac(device: str = "cpu"):
    """Load DAC 44kHz model (singleton, lazy-loaded)."""
    global _dac_model
    if _dac_model is None:
        import dac

        _dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
        _dac_model = _dac_model.to(device).eval()
    return _dac_model


def encode_to_dac_latent(waveform: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Encode waveforms to continuous DAC latents (pre-quantization).

    Args:
        waveform: (B, N_SAMPLES) float32
        device: target device

    Returns:
        (B, dac_latent_dim=64, T=~130) continuous latent
    """
    dac_model = get_dac(device)
    with torch.no_grad():
        wav = waveform.unsqueeze(1).to(device)  # (B, 1, N)
        z, _, _, _, _ = dac_model.encode(wav)  # continuous encoder output
    return z  # (B, 64, T)


def decode_from_dac_latent(dac_z: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Decode continuous DAC latents back to waveforms.

    Args:
        dac_z: (B, 64, T) continuous DAC latent

    Returns:
        (B, N_SAMPLES) float32 waveform
    """
    dac_model = get_dac(device)
    with torch.no_grad():
        waveform = dac_model.decode(dac_z.to(device))  # (B, 1, N_SAMPLES)
    return waveform.squeeze(1)  # (B, N_SAMPLES)
