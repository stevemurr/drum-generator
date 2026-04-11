# drum-generator

Text-conditioned drum one-shot generator using a Diffusion Transformer (DiT) with Conditional Flow Matching, trained in a compressed VAE latent space.

## How it works

```
Text prompt → [CLAP] → embedding (512d)
                            ↓
Waveform → [DAC encoder] → (64×130) → [VAE encoder] → (16×130)
                                                          ↓
                                        [DiT + Flow Matching] ← CLAP embedding
                                                  ↓
                           VAE decoder → DAC decoder → Waveform
```

- **DAC** (Descript Audio Codec) compresses raw audio into a continuous latent
- **VAE** further compresses DAC latents from 64 to 16 channels
- **DiT** learns to generate in VAE latent space via flow matching, conditioned on CLAP text embeddings
- **CLAP** encodes text prompts into the same embedding space used during training

## Install

```bash
git clone https://github.com/stevemurr/drum-generator.git
cd drum-generator
uv pip install -e .
```

For CUDA support, the `pyproject.toml` is pre-configured with the PyTorch cu130 index via `[tool.uv.sources]`.

## Quick start

### Generate drums (requires trained checkpoints)

```bash
drum-generate --prompt "punchy kick drum, sub-heavy, dry" --n 4
drum-generate --prompt "tight snare, crisp, electronic" --n 4 --steps 16
drum-generate --prompt "closed hi-hat, bright" --n 2 --cfg 6.0
```

### Train

```bash
# Download drum samples from Freesound (requires FREESOUND_TOKEN env var)
export FREESOUND_TOKEN="your-token-here"
drum-download

# Train both VAE and DiT
drum-train --phase both

# Or train phases separately
drum-train --phase vae
drum-train --phase dit
```

### Train with synthetic data (no downloads needed)

See [docs/tutorials/synthetic_training.md](docs/tutorials/synthetic_training.md) for a full walkthrough using procedurally generated drum sounds.

```python
from drum_generator.dataset import build_dataset

ds = build_dataset(
    freesound_dir=None,
    synthetic_size=2000,
    augment=True,
    augment_multiplier=2,
)
```

## Dataset sources

The dataset module supports three composable sources, all returning `(waveform, clap_embed)` tuples:

| Source | Description |
|--------|-------------|
| **Freesound** | CC0 drum one-shots downloaded via API, with metadata-driven captions |
| **Disk** | Arbitrary audio files (.wav, .mp3, .flac, .ogg, .aif) from local directories |
| **Synthetic** | Procedurally generated kicks, snares, hi-hats, claps, toms, rimshots, cymbals |

Sources are combined via `build_dataset()` and optionally wrapped with on-the-fly augmentation (pitch shift, gain, noise, reverb, filtering, polarity inversion, time offset).

```python
from drum_generator.dataset import build_dataset

ds = build_dataset(
    freesound_dir="data",
    disk_dirs=["/path/to/my/samples"],
    synthetic_size=1000,
    augment=True,
)
```

### Disk dataset captions

When loading arbitrary audio files, captions are resolved in order:

1. Sidecar `.txt` file (e.g. `kick_01.txt` next to `kick_01.wav`)
2. Sidecar `.json` with a `caption` key
3. External `labels.csv` (filename,caption rows)
4. Heuristic from filename + parent directory (e.g. `kicks/808_sub_hard_01.wav` → `"808 kick, sub, hard"`)

## Architecture

| Component | Details |
|-----------|---------|
| Audio | 44.1kHz, 1.5s, mono (66150 samples) |
| DAC latent | 64 channels × 130 frames |
| VAE latent | 16 channels × 130 frames |
| DiT | 256-dim, 8 heads, 6 layers, patch size 4 (~32 tokens) |
| CLAP | `laion/larger_clap_general`, 512-dim text embeddings |
| Flow matching | Straight-line CFM, Euler ODE (8 steps), CFG scale 4.0 |

## Project structure

```
src/drum_generator/
    config.py              Configuration dataclass
    vae.py                 1D-CNN VAE (ResNet-based)
    dit.py                 DiT + flow matching loss + generation
    train.py               Two-phase training (VAE then DiT)
    generate.py            Text-to-drum inference
    dataset/
        __init__.py        build_dataset() factory
        caption.py         Caption building + CLAP embedder
        freesound.py       Freesound API dataset
        disk.py            Arbitrary audio file dataset
        synthetic.py       Procedural drum synthesis
        augment.py         Audio augmentation transforms
```
