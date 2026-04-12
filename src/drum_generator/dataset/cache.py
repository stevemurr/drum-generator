"""
cache.py
--------
CachedDACDataset: pre-encode waveforms through DAC and cache to disk.

Wraps any (waveform, clap_embed) dataset. On first access, encodes all
waveforms through DAC and saves (dac_latent, clap_embed) pairs as .pt
files. Subsequent accesses load directly from cache — no DAC model needed.
"""

import json
import os

import torch
from torch.utils.data import Dataset

from drum_generator.config import CFG


class CachedDACDataset(Dataset):
    """Pre-encodes waveforms through DAC and caches to disk.

    Returns (dac_latent [64, 130], clap_embed [512]) instead of
    (waveform [66150], clap_embed [512]).
    """

    def __init__(self, base_dataset: Dataset, cache_dir: str, device: str = "cpu"):
        self.base = base_dataset
        self.cache_dir = cache_dir
        self.device = device
        self._size = len(base_dataset)

        if not self._cache_valid():
            self._build_cache()

    def _meta_path(self) -> str:
        return os.path.join(self.cache_dir, "meta.json")

    def _item_path(self, idx: int) -> str:
        return os.path.join(self.cache_dir, f"{idx:06d}.pt")

    def _cache_valid(self) -> bool:
        """Check if cache exists and matches current dataset size."""
        meta_path = self._meta_path()
        if not os.path.exists(meta_path):
            return False
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("size") != self._size:
            return False
        # Spot-check: first and last files exist
        if not os.path.exists(self._item_path(0)):
            return False
        if not os.path.exists(self._item_path(self._size - 1)):
            return False
        return True

    def _build_cache(self):
        """Encode all waveforms through DAC and save to disk."""
        from tqdm import tqdm

        from drum_generator.codec import encode_to_dac_latent

        os.makedirs(self.cache_dir, exist_ok=True)

        batch_size = 16
        waveforms = []
        clap_embeds = []
        indices = []

        pbar = tqdm(total=self._size, desc="[cache] encoding DAC latents", unit="sample")

        for i in range(self._size):
            waveform, clap_embed = self.base[i]
            waveforms.append(waveform)
            clap_embeds.append(clap_embed)
            indices.append(i)

            if len(waveforms) == batch_size or i == self._size - 1:
                wav_batch = torch.stack(waveforms)
                dac_batch = encode_to_dac_latent(wav_batch, self.device)

                for j, idx in enumerate(indices):
                    torch.save(
                        {"dac_z": dac_batch[j].cpu(), "clap_embed": clap_embeds[j]},
                        self._item_path(idx),
                    )

                pbar.update(len(waveforms))
                waveforms.clear()
                clap_embeds.clear()
                indices.clear()

        pbar.close()

        # Write metadata
        with open(self._meta_path(), "w") as f:
            json.dump(
                {
                    "size": self._size,
                    "dac_dim": CFG.dac_latent_dim,
                    "dac_frames": CFG.dac_time_frames,
                    "clap_dim": CFG.clap_dim,
                },
                f,
            )
        print(f"[cache] done — {self._size} DAC latents cached")

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = torch.load(self._item_path(idx), map_location="cpu", weights_only=True)
        return data["dac_z"], data["clap_embed"]
