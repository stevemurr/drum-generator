"""
disk.py
-------
DiskAudioDataset: load arbitrary audio files from directories on disk.
Supports wav, mp3, flac, ogg, aif/aiff.
"""

import csv
import hashlib
import json
import os

import torch
from torch.utils.data import Dataset

from drum_generator.config import CFG
from drum_generator.dataset.caption import (
    ClapEmbedder,
    build_caption,
    build_caption_from_filename,
    load_audio_file,
)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff"}


class DiskAudioDataset(Dataset):
    """Load arbitrary audio files from directory trees.

    Caption resolution priority:
      1. Sidecar .txt file ({stem}.txt next to audio)
      2. Sidecar .json file with 'caption' key, or Freesound-style metadata
      3. External labels file (CSV: filename,caption)
      4. Heuristic from filename + parent directory name

    Returns (waveform [N_SAMPLES], clap_embed [CLAP_DIM]).
    """

    def __init__(
        self,
        root_dirs: list[str],
        clap_embedder: ClapEmbedder | None = None,
        cache_dir: str | None = None,
        label_file: str | None = None,
    ):
        self.clap = clap_embedder or ClapEmbedder.get()
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # Discover all audio files
        self.files: list[str] = []
        for root in root_dirs:
            for dirpath, _, filenames in os.walk(root):
                for fname in sorted(filenames):
                    if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTENSIONS:
                        self.files.append(os.path.join(dirpath, fname))

        # Load external labels if provided
        self.labels: dict[str, str] = {}
        if label_file:
            self.labels = self._load_labels(label_file)

        print(f"[dataset] {len(self.files)} audio files found across {len(root_dirs)} dir(s)")

    @staticmethod
    def _load_labels(label_file: str) -> dict[str, str]:
        """Load filename -> caption mapping from CSV or JSON."""
        ext = os.path.splitext(label_file)[1].lower()
        if ext == ".json":
            with open(label_file) as f:
                return json.load(f)
        # Assume CSV: filename,caption
        labels = {}
        with open(label_file, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    labels[row[0].strip()] = row[1].strip()
        return labels

    def _get_caption(self, filepath: str) -> str:
        """Resolve caption for a file using the priority chain."""
        stem = os.path.splitext(filepath)[0]

        # 1. Sidecar .txt
        txt_path = stem + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                return f.read().strip()

        # 2. Sidecar .json
        json_path = stem + ".json"
        if os.path.exists(json_path):
            with open(json_path) as f:
                meta = json.load(f)
            if "caption" in meta:
                return meta["caption"]
            # Try Freesound-style metadata
            if "tags" in meta:
                return build_caption(meta)

        # 3. External labels dict
        basename = os.path.basename(filepath)
        if basename in self.labels:
            return self.labels[basename]

        # 4. Heuristic from filename + directory
        return build_caption_from_filename(filepath)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.files[idx]
        waveform = load_audio_file(path, CFG.sample_rate, CFG.n_samples)
        caption = self._get_caption(path)

        cache_path = None
        if self.cache_dir:
            cache_key = hashlib.md5(path.encode()).hexdigest()
            cache_path = os.path.join(self.cache_dir, f"{cache_key}_clap.pt")

        embed = self.clap.embed(caption, cache_path=cache_path)
        return waveform, embed
