"""
generate.py
-----------
Generate drum one-shots from a text prompt.

Usage:
    python generate.py --prompt "punchy 808 kick, sub-heavy, dry" --n 4
    python generate.py --prompt "tight snare, cracking, room reverb"
"""

import argparse

import numpy as np
import scipy.io.wavfile as wav
import torch

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
    dit.load_state_dict(torch.load(f"{CFG.ckpt_dir}/dit_best.pt", map_location=DEVICE))
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


def decode_to_audio(vae: DrumVAE, vae_latent: torch.Tensor) -> np.ndarray:
    """VAE latent → DAC latent → waveform via DAC decoder."""
    import dac

    dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
    dac_model = dac_model.to(DEVICE).eval()

    with torch.no_grad():
        dac_z = vae.decode(vae_latent)  # (B, 64, T)
        # Decode continuous latent directly (skip re-quantization)
        waveform = dac_model.decode(dac_z)  # (B, 1, N_SAMPLES)

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
    args = parser.parse_args()

    print(f"Prompt: {args.prompt!r}")
    print(
        f"Generating {args.n} variations, {args.steps} ODE steps, CFG scale {args.cfg}"
    )

    vae, dit = load_models()

    # Encode prompt → CLAP embed, repeat for batch
    clap_embed = encode_prompt(args.prompt)  # (1, 512)
    clap_batch = clap_embed.expand(args.n, -1)  # (N, 512)

    # Flow matching: noise → VAE latent
    vae_latent = fm_generate(dit, clap_batch, args.steps, args.cfg, DEVICE)

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
