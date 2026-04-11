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

Generate a few samples to see what you're working with:

```python
from drum_generator.dataset.synthetic import SyntheticDrumDataset

ds = SyntheticDrumDataset(size=8, seed=42)
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

ds = SyntheticDrumDataset(size=8, seed=42)
for i in range(len(ds)):
    wav, _ = ds[i]
    audio = (wav.numpy() * 32767).astype(np.int16)
    wavfile.write(f"synth_preview_{i}.wav", CFG.sample_rate, audio)
    print(f"saved synth_preview_{i}.wav — {ds.params[i]['drum_type']}")
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

## 3. Train the VAE

The VAE learns to compress DAC encoder latents (64×130) into a smaller space (16×130).

```bash
# Point away from the Freesound data dir so only synthetic data is used
python -c "
from drum_generator.config import CFG
from drum_generator.dataset import build_dataset
from drum_generator.train import train_vae, encode_to_dac_latent, DEVICE
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

## 4. Train the DiT

The DiT learns flow matching in the VAE latent space, conditioned on CLAP text embeddings.

```python
python -c "
from drum_generator.config import CFG
from drum_generator.dataset import build_dataset
from drum_generator.train import train_dit, encode_to_dac_latent, DEVICE
from drum_generator.vae import DrumVAE
from torch.utils.data import DataLoader, random_split
import torch

CFG.dit_epochs = 50

ds = build_dataset(
    freesound_dir=None,
    synthetic_size=2000,
    augment=True,
    augment_multiplier=2,
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

## 5. Generate Drum Sounds

Once both checkpoints exist, generate from text prompts:

```bash
drum-generate --prompt "punchy kick drum, sub, hard" --n 4 --steps 8 --cfg 4.0
drum-generate --prompt "tight snare, crisp, electronic" --n 4
drum-generate --prompt "closed hi-hat, bright, clean" --n 4
```

Output files are saved as `generated_01.wav`, `generated_02.wav`, etc. in the current directory.

## 6. Quick Sanity Check Script

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

print('=== Generate ===')
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

- **Synthetic quality**: These are basic sine/noise synthesized drums. They won't sound like real recordings, but they exercise the full model pipeline. Mix in real samples (Freesound or your own via `disk_dirs`) for better results.
- **Dataset size**: 2000 synthetic samples with 2x augmentation multiplier gives 4000 effective training examples per epoch. For serious training, use 5000+ synthetic samples alongside real data.
- **CLAP embeddings**: Pre-computed at dataset init time for synthetic sounds. First run takes ~30s for 2000 samples; subsequent items are instant.
- **Augmentation**: Enabled by default. Applies random pitch shift, gain, noise, reverb, filtering, and polarity inversion. Disable with `augment=False` if you want to isolate training issues.
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
