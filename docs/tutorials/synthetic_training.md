# Training with Synthetic Drums: End-to-End Tutorial

This tutorial walks through the full pipeline — dataset creation, VAE training, DiT training, and generation — using only synthetic drum samples. No Freesound account or downloaded data needed.

## Prerequisites

```bash
cd /path/to/drum-generator
uv pip install -e .
```

Verify GPU is available (strongly recommended):

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 1. Explore the Synthetic Dataset

The dataset cycles through all 8 drum types in round-robin order (kick, snare, hihat closed/open, clap, tom, rimshot, cymbal), generating variations of each with randomized synthesis parameters. Characteristic tags like "punchy", "distorted", or "warm" directly shape the sound — controlling attack time, waveshaping drive, pitch ranges, and more.

Generate a set of samples to see what you're working with:

```python
from drum_generator.dataset.synthetic import SyntheticDrumDataset

ds = SyntheticDrumDataset(size=16, seed=42)  # 2 variations per drum type
for i in range(len(ds)):
    wav, emb = ds[i]
    p = ds.params[i]
    print(f"{p['drum_type']:15s} chars={p['characteristics']}")
```

Save some to disk and listen:

```python
import scipy.io.wavfile as wavfile
import numpy as np
from drum_generator.config import CFG

ds = SyntheticDrumDataset(size=16, seed=42)
for i in range(len(ds)):
    wav, _ = ds[i]
    audio = (wav.numpy() * 32767).astype(np.int16)
    fname = f"synth_preview_{ds.params[i]['drum_type']}_{i}.wav"
    wavfile.write(fname, CFG.sample_rate, audio)
    print(f"saved {fname}")
```

## 2. Configure for Synthetic-Only Training

Edit `src/drum_generator/config.py` or override at runtime. The key settings:

```python
# In config.py, or pass to build_dataset() directly:
synthetic_size = 2000   # number of synthetic samples
augment = True          # on-the-fly augmentation
augment_multiplier = 2  # each sample seen 2x per epoch with different augmentations
```

For a quick test run, reduce epochs:

```python
# config.py overrides for fast iteration
vae_epochs = 20
dit_epochs = 50
```

## 3. Pre-encode DAC Latents (optional but recommended)

DAC encoding is the biggest per-batch bottleneck — it's a full neural network forward pass on every sample, every epoch, and the output never changes. Pass `cache=True` to encode everything once upfront:

```python
ds = build_dataset(
    freesound_dir=None,
    synthetic_size=2000,
    augment=True,
    augment_multiplier=2,
    cache=True,                 # pre-encode through DAC, save to disk
    cache_dir="cache",          # default location
)
```

First run encodes all samples through DAC and saves `(dac_latent, clap_embed)` pairs to disk (~35KB each). Subsequent runs load directly from cache — no DAC model needed. The cache auto-rebuilds if the dataset size changes.

## 4. Train the VAE

The VAE learns to compress DAC encoder latents (1024x129) into a smaller space (16x129) via gradual channel reduction (1024 → 512 → 256 → 128 → 16).

```python
python -c "
from drum_generator.config import CFG
from drum_generator.dataset import build_dataset
from drum_generator.train import train_vae, DEVICE
from torch.utils.data import DataLoader, random_split
import os, torch

# Override for synthetic-only
CFG.vae_epochs = 20

os.makedirs(CFG.ckpt_dir, exist_ok=True)

ds = build_dataset(
    freesound_dir=None,        # skip Freesound
    synthetic_size=2000,
    augment=True,
    augment_multiplier=2,
    cache=True,                # pre-encode DAC latents
)
print(f'Dataset size: {len(ds)}')

n_val = max(1, int(len(ds) * 0.1))
train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])

train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=4, pin_memory=True)

vae = train_vae(train_loader, val_loader)
print('VAE training complete — saved checkpoints/vae_best.pt')
"
```

Or more concisely via the CLI after temporarily editing config.py:

```bash
drum-train --phase vae
```

## 5. Train the DiT

The DiT learns flow matching in the VAE latent space, conditioned on CLAP text embeddings and audio references. During training, each sample is paired with a random other sample from the batch as its reference. Reference conditioning is dropped 50% of the time so the model works well with text alone.

```python
python -c "
from drum_generator.config import CFG
from drum_generator.dataset import build_dataset
from drum_generator.train import train_dit, DEVICE
from drum_generator.vae import DrumVAE
from torch.utils.data import DataLoader, random_split
import torch

CFG.dit_epochs = 50

ds = build_dataset(
    freesound_dir=None,
    synthetic_size=2000,
    augment=True,
    augment_multiplier=2,
    cache=True,                # reuses the same cache from VAE phase
)

n_val = max(1, int(len(ds) * 0.1))
train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])

train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Load trained VAE
vae = DrumVAE().to(DEVICE)
vae.load_state_dict(torch.load(f'{CFG.ckpt_dir}/vae_best.pt', map_location=DEVICE))

train_dit(train_loader, val_loader, vae)
print('DiT training complete — saved checkpoints/dit_best.pt')
"
```

## 6. Generate Drum Sounds

Once both checkpoints exist, generate from text prompts:

```bash
drum-generate --prompt "punchy kick drum, sub, hard" --n 4 --steps 8 --cfg 4.0
drum-generate --prompt "tight snare, crisp, electronic" --n 4
drum-generate --prompt "closed hi-hat, bright, clean" --n 4
```

Output files are saved as `generated_01.wav`, `generated_02.wav`, etc. in the current directory.

## 7. Reference-Conditioned Generation

Provide a reference audio file to steer the tonal character of the output. The model uses the reference's spectral/tonal qualities while following the text prompt for what kind of sound to generate.

```bash
# Generate a snare that lives in the same "tonal universe" as your kick
drum-generate --prompt "tight snare, crisp" --ref my_kick.wav --ref-cfg 2.0

# Higher --ref-cfg = stronger tonal influence from reference
drum-generate --prompt "closed hi-hat, bright" --ref my_kick.wav --ref-cfg 4.0

# Lower --ref-cfg = subtler influence, text dominates
drum-generate --prompt "snare, punchy" --ref my_kick.wav --ref-cfg 1.0
```

The reference audio is encoded through DAC and VAE into the same latent space the model operates in, then attended to via cross-attention at each transformer layer.

You can use any audio file as reference — a synthetic preview from step 1, a Freesound sample from `data/`, or your own recordings.

## 8. Quick Sanity Check Script

Run the whole pipeline in miniature (tiny dataset, few epochs) to verify everything connects:

```python
python -c "
from drum_generator.config import CFG
from drum_generator.dataset import build_dataset
from drum_generator.train import train_vae, train_dit, DEVICE
from drum_generator.vae import DrumVAE
from torch.utils.data import DataLoader, random_split
import os, torch

# Minimal config for smoke test
CFG.vae_epochs = 2
CFG.dit_epochs = 2
CFG.batch_size = 4
os.makedirs(CFG.ckpt_dir, exist_ok=True)

ds = build_dataset(freesound_dir=None, synthetic_size=32, augment=False)
train_ds, val_ds = random_split(ds, [28, 4])
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

print('=== VAE (2 epochs) ===')
vae = train_vae(train_loader, val_loader)

print('=== DiT (2 epochs) ===')
vae = DrumVAE().to(DEVICE)
vae.load_state_dict(torch.load(f'{CFG.ckpt_dir}/vae_best.pt', map_location=DEVICE))
train_dit(train_loader, val_loader, vae)

print('=== Generate (text-only) ===')
from drum_generator.generate import load_models, encode_prompt, decode_to_audio
from drum_generator.dit import generate as fm_generate
import numpy as np, scipy.io.wavfile as wavfile

vae, dit = load_models()
clap_embed = encode_prompt('kick drum, punchy')
latent = fm_generate(dit, clap_embed, steps=4, cfg_scale=2.0, device=DEVICE)
audio = decode_to_audio(vae, latent)
audio = audio / (np.abs(audio).max() + 1e-8)
wavfile.write('smoke_test.wav', CFG.sample_rate, (audio * 32767).astype(np.int16))
print('Saved smoke_test.wav — pipeline works end to end')
"
```

## Notes

- **Synthetic quality**: Synthesis uses ADSR envelopes, tanh waveshaping, inharmonic metallic partials (physical cymbal ratios), comb-filtered snare wires, and reverb convolution. Characteristic tags directly control parameters — "punchy" means faster attack, "distorted" means higher waveshaping drive, "sub" means lower pitch target. They won't match real recordings, but they provide musically meaningful training signal. Mix in real samples for best results.
- **Dataset size**: 2000 synthetic samples with 2x augmentation multiplier gives 4000 effective training examples per epoch, with even coverage across all 8 drum types (250 each). For serious training, use 5000+ synthetic samples alongside real data.
- **DAC caching**: Pass `cache=True` to pre-encode all samples through DAC once. Eliminates the biggest per-batch bottleneck for both VAE and DiT training. Cache auto-rebuilds when dataset size changes. ~35KB per sample on disk.
- **CLAP embeddings**: Pre-computed at dataset init time for synthetic sounds. First run takes ~30s for 2000 samples; subsequent items are instant.
- **Augmentation**: Enabled by default. Applies random pitch shift, gain, noise, reverb, filtering, and polarity inversion. With `cache=True`, augmented variants are baked into the cache — a fixed set of augmented samples rather than infinite random ones, but with `augment_multiplier=2` you get plenty of diversity.
- **Reference conditioning**: The DiT trains with audio references from the start (random batch pairings, 50% dropout). At inference, `--ref-cfg 0.0` disables reference influence entirely, matching text-only behavior.
- **Combining sources**: Once you're satisfied with the pipeline, add real data:
  ```python
  ds = build_dataset(
      freesound_dir="data",          # Freesound downloads
      disk_dirs=["/path/to/samples"], # your own audio files
      synthetic_size=1000,            # supplement with synthetic
      augment=True,
      augment_multiplier=2,
  )
  ```
