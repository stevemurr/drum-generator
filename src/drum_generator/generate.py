"""
generate.py
-----------
Generate drum one-shots from a text prompt, optionally conditioned on
a reference audio file for tonal steering.

Usage:
    python generate.py --prompt "punchy 808 kick, sub-heavy, dry" --n 4
    python generate.py --prompt "tight snare" --ref kick.wav --ref-cfg 2.0
"""

import argparse

import numpy as np
import scipy.io.wavfile as wav
import torch

from drum_generator.codec import decode_from_dac_latent, encode_to_dac_latent
from drum_generator.config import CFG
from drum_generator.dit import DrumDiT
from drum_generator.dit import generate as fm_generate
from drum_generator.vae import DrumVAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    vae = DrumVAE().to(DEVICE)
    vae.load_state_dict(torch.load(f"{CFG.ckpt_dir}/vae_best.pt", map_location=DEVICE))
    vae.eval()

    dit = DrumDiT().to(DEVICE)
    dit.load_state_dict(
        torch.load(f"{CFG.ckpt_dir}/dit_best.pt", map_location=DEVICE),
        strict=False,  # allows loading text-only checkpoints into ref-enabled model
    )
    dit.eval()

    return vae, dit


def encode_prompt(prompt: str) -> torch.Tensor:
    from transformers import ClapModel, ClapProcessor

    proc = ClapProcessor.from_pretrained("laion/larger_clap_general")
    model = ClapModel.from_pretrained("laion/larger_clap_general").eval()
    inputs = proc(text=prompt, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
        embed = out.pooler_output if hasattr(out, "pooler_output") else out  # (1, 512)
    return embed.to(DEVICE)


def encode_reference(ref_path: str, vae: DrumVAE) -> torch.Tensor:
    """Load reference audio → DAC → VAE → ref_z latent."""
    from drum_generator.dataset.caption import load_audio_file

    waveform = load_audio_file(ref_path)  # (N_SAMPLES,)
    waveform = waveform.unsqueeze(0)  # (1, N_SAMPLES)

    with torch.no_grad():
        dac_z = encode_to_dac_latent(waveform, DEVICE)  # (1, 64, T)
        mu, logvar = vae.encode(dac_z)
        ref_z = vae.reparameterize(mu, logvar)  # (1, 16, T)

    return ref_z


def decode_to_audio(vae: DrumVAE, vae_latent: torch.Tensor) -> np.ndarray:
    """VAE latent → DAC latent → waveform."""
    with torch.no_grad():
        dac_z = vae.decode(vae_latent)  # (B, 64, T)
        waveform = decode_from_dac_latent(dac_z, DEVICE)  # (B, N_SAMPLES)
    return waveform.squeeze().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", type=str, default="punchy kick drum, sub-heavy, dry, hard transient"
    )
    parser.add_argument(
        "--n", type=int, default=4, help="Number of variations to generate"
    )
    parser.add_argument("--steps", type=int, default=CFG.fm_steps_infer)
    parser.add_argument("--cfg", type=float, default=CFG.cfg_scale)
    parser.add_argument(
        "--ref", type=str, default=None, help="Reference audio file for tonal conditioning"
    )
    parser.add_argument("--ref-cfg", type=float, default=CFG.ref_cfg_scale)
    args = parser.parse_args()

    print(f"Prompt: {args.prompt!r}")
    if args.ref:
        print(f"Reference: {args.ref} (ref-cfg={args.ref_cfg})")
    print(
        f"Generating {args.n} variations, {args.steps} ODE steps, CFG scale {args.cfg}"
    )

    vae, dit = load_models()

    # Encode prompt → CLAP embed, repeat for batch
    clap_embed = encode_prompt(args.prompt)  # (1, 512)
    clap_batch = clap_embed.expand(args.n, -1)  # (N, 512)

    # Encode reference audio (if provided)
    ref_z = None
    if args.ref:
        ref_z = encode_reference(args.ref, vae)  # (1, 16, T)
        ref_z = ref_z.expand(args.n, -1, -1)  # (N, 16, T)

    # Flow matching: noise → VAE latent
    vae_latent = fm_generate(
        dit, clap_batch,
        ref_z=ref_z,
        steps=args.steps,
        cfg_scale=args.cfg,
        ref_cfg_scale=args.ref_cfg,
        device=DEVICE,
    )

    # Decode each variation
    for i in range(args.n):
        audio = decode_to_audio(vae, vae_latent[i : i + 1])

        # Normalize + save
        audio = audio / (np.abs(audio).max() + 1e-8)
        audio_int = (audio * 32767).astype(np.int16)
        fname = f"generated_{i + 1:02d}.wav"
        wav.write(fname, CFG.sample_rate, audio_int)
        print(f"  saved {fname}")


if __name__ == "__main__":
    main()
