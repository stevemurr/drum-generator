"""
cache.py
--------
CachedDACDataset: pre-encode waveforms through DAC and cache to disk.

Wraps any (waveform, clap_embed) dataset. On first access, encodes all
waveforms through DAC and saves (dac_latent, clap_embed) pairs as .pt
files. Subsequent accesses load directly from cache — no DAC model needed.

Uses pipelined prefetching so CPU sample preparation and GPU DAC encoding
overlap.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread

import torch
from torch.utils.data import Dataset

from drum_generator.config import CFG


class CachedDACDataset(Dataset):
    """Pre-encodes waveforms through DAC and caches to disk.

    Returns (dac_latent [1024, 129], clap_embed [512]) instead of
    (waveform [66150], clap_embed [512]).
    """

    def __init__(self, base_dataset: Dataset, cache_dir: str, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
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
        """Encode all waveforms through DAC and save to disk.

        Uses a prefetch pipeline: a background thread prepares the next
        batch on CPU while the current batch is being DAC-encoded on GPU.
        Disk writes are also offloaded to a thread pool.
        """
        from tqdm import tqdm

        from drum_generator.codec import encode_to_dac_latent

        os.makedirs(self.cache_dir, exist_ok=True)

        batch_size = 16
        prefetch_queue: Queue = Queue(maxsize=2)

        # --- Background prefetch thread ---
        def _prefetch():
            waveforms = []
            clap_embeds = []
            indices = []

            for i in range(self._size):
                waveform, clap_embed = self.base[i]
                waveforms.append(waveform)
                clap_embeds.append(clap_embed)
                indices.append(i)

                if len(waveforms) == batch_size or i == self._size - 1:
                    wav_batch = torch.stack(waveforms)
                    prefetch_queue.put((wav_batch, list(clap_embeds), list(indices)))
                    waveforms = []
                    clap_embeds = []
                    indices = []

            prefetch_queue.put(None)  # sentinel

        prefetch_thread = Thread(target=_prefetch, daemon=True)
        prefetch_thread.start()

        # --- Async disk writes ---
        save_pool = ThreadPoolExecutor(max_workers=4)
        save_futures = []

        def _save_item(path, dac_z_cpu, clap_embed):
            torch.save({"dac_z": dac_z_cpu, "clap_embed": clap_embed}, path)

        # --- Main loop: GPU encode + dispatch saves ---
        pbar = tqdm(total=self._size, desc="[cache] encoding DAC latents", unit="sample")

        while True:
            batch = prefetch_queue.get()
            if batch is None:
                break

            wav_batch, clap_embeds, indices = batch
            dac_batch = encode_to_dac_latent(wav_batch, self.device)

            for j, idx in enumerate(indices):
                fut = save_pool.submit(
                    _save_item, self._item_path(idx), dac_batch[j].cpu(), clap_embeds[j]
                )
                save_futures.append(fut)

            pbar.update(len(indices))

        # Wait for all disk writes to finish
        for fut in save_futures:
            fut.result()

        save_pool.shutdown(wait=True)
        prefetch_thread.join()
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
