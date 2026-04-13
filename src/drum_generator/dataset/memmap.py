"""
memmap.py
---------
MemmapDACDataset: read precomputed DAC latents and CLAP text embeddings
from numpy memmap files.

This is the fast-path training dataset, intended to be used with a directory
produced by the companion `dataset-caption` pipeline. It skips all on-the-fly
work — no wav decoding, no DAC encoder forward, no CLAP text encoding, no
augmentation. Each __getitem__ is two memmap reads.

Expected layout:
    <memmap_dir>/
        dac_latents.npy      (N, 1024, 129) float32 — DAC continuous latents
        embeddings_text.npy  (N, 512)       float32 — CLAP text embeddings
        index.json           list[str] — sha16 per row (row-aligned)

If you need waveform-level augmentation (pitch shift, reverb, noise, etc.)
use the live path via DiskAudioDataset + CachedDACDataset instead — memmap
mode cannot re-synthesize waveforms from stored latents.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapDACDataset(Dataset):
    """Training dataset that serves (dac_latent, clap_embed) from memmap files.

    Interface matches CachedDACDataset exactly — drop-in replacement for the
    DiT training loop.
    """

    DAC_FILE = "dac_latents.npy"
    TEXT_FILE = "embeddings_text.npy"
    INDEX_FILE = "index.json"

    def __init__(self, memmap_dir: str):
        self.memmap_dir = memmap_dir

        dac_path = os.path.join(memmap_dir, self.DAC_FILE)
        text_path = os.path.join(memmap_dir, self.TEXT_FILE)
        index_path = os.path.join(memmap_dir, self.INDEX_FILE)

        for p in (dac_path, text_path, index_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"MemmapDACDataset: missing {p} — run the dataset-caption "
                    f"precompute pipeline and point CFG.memmap_dir at the output "
                    f"directory."
                )

        self.dac = np.load(dac_path, mmap_mode="r")
        self.text = np.load(text_path, mmap_mode="r")
        with open(index_path) as f:
            self.index: list[str] = json.load(f)

        if not (len(self.dac) == len(self.text) == len(self.index)):
            raise ValueError(
                f"MemmapDACDataset: row mismatch in {memmap_dir} — "
                f"dac={len(self.dac)} text={len(self.text)} index={len(self.index)}"
            )

        print(
            f"[memmap] loaded {len(self.dac)} rows from {memmap_dir}: "
            f"dac {self.dac.shape} {self.dac.dtype}, "
            f"text {self.text.shape} {self.text.dtype}"
        )

    def __len__(self) -> int:
        return len(self.dac)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # .copy() materializes the mmap slice into a regular array. Without
        # it, DataLoader workers may hold references to kernel page-cache
        # pages that PyTorch then frees prematurely.
        dac_z = torch.from_numpy(self.dac[idx].copy())   # (1024, 129)
        clap = torch.from_numpy(self.text[idx].copy())    # (512,)
        return dac_z, clap

    def sha16(self, idx: int) -> str:
        """Return the sha16 content hash for row idx (useful for debugging)."""
        return self.index[idx]
