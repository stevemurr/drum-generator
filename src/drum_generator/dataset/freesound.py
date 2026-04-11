"""
freesound.py
------------
Freesound API downloader and FreesoundDataset for loading downloaded
drum one-shots with JSON metadata.
"""

import argparse
import json
import os
import time

import requests
import torch
from torch.utils.data import Dataset

from drum_generator.config import CFG
from drum_generator.dataset.caption import ClapEmbedder, build_caption, load_audio_file

# ---------------------------------------------------------------------------
# Freesound downloader
# ---------------------------------------------------------------------------

DRUM_QUERIES = [
    "kick drum one shot",
    "snare drum one shot",
    "hi hat one shot closed",
    "hi hat one shot open",
    "clap one shot",
    "tom drum one shot",
    "cymbal crash one shot",
    "808 kick one shot",
    "rim shot one shot",
    "cowbell one shot",
]


def _search_freesound(query: str, token: str, page: int = 1) -> dict:
    r = requests.get(
        "https://freesound.org/apiv2/search/text/",
        params={
            "query": query,
            "filter": 'license:"Creative Commons 0" duration:[0 TO 2]',
            "fields": "id,name,tags,description,duration,previews",
            "page_size": 150,
            "page": page,
            "token": token,
        },
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def download_dataset(token: str, out_dir: str, max_per_query: int = 300):
    """Download CC0 drum one-shots and save metadata JSON alongside each file."""
    os.makedirs(out_dir, exist_ok=True)
    seen = set()

    for query in DRUM_QUERIES:
        print(f"\n[freesound] query: {query!r}")
        collected = 0
        page = 1

        while collected < max_per_query:
            data = _search_freesound(query, token, page)
            results = data.get("results", [])
            if not results:
                break

            for sound in results:
                sid = sound["id"]
                if sid in seen:
                    continue
                seen.add(sid)

                mp3_path = os.path.join(out_dir, f"{sid}.mp3")
                meta_path = os.path.join(out_dir, f"{sid}.json")

                if not os.path.exists(mp3_path):
                    url = sound["previews"].get("preview-hq-mp3")
                    if not url:
                        continue
                    try:
                        audio_bytes = requests.get(url, timeout=15).content
                        with open(mp3_path, "wb") as f:
                            f.write(audio_bytes)
                        with open(meta_path, "w") as f:
                            json.dump(sound, f)
                        time.sleep(0.3)
                    except Exception as e:
                        print(f"  skip {sid}: {e}")
                        continue

                collected += 1
                if collected >= max_per_query:
                    break

            page += 1
            if not data.get("next"):
                break

        print(f"  collected {collected} sounds")


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class FreesoundDataset(Dataset):
    """Loads Freesound drum one-shots with JSON metadata.

    Returns (waveform_tensor [N_SAMPLES], clap_embed [CLAP_DIM]).
    CLAP embeddings are computed once and cached as .pt files.
    """

    def __init__(self, data_dir: str = CFG.data_dir, clap_embedder: ClapEmbedder | None = None):
        self.data_dir = data_dir
        self.clap = clap_embedder or ClapEmbedder.get()

        self.ids = [f[:-5] for f in os.listdir(data_dir) if f.endswith(".json")]
        print(f"[dataset] {len(self.ids)} Freesound sounds found in {data_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sid = self.ids[idx]
        with open(os.path.join(self.data_dir, f"{sid}.json")) as f:
            meta = json.load(f)

        waveform = load_audio_file(os.path.join(self.data_dir, f"{sid}.mp3"))
        caption = build_caption(meta)
        cache_path = os.path.join(self.data_dir, f"{sid}_clap.pt")
        embed = self.clap.embed(caption, cache_path=cache_path)

        return waveform, embed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def cli_download():
    """CLI entry point for drum-download command."""
    parser = argparse.ArgumentParser(description="Download drum one-shots from Freesound")
    parser.add_argument(
        "--token",
        type=str,
        default=CFG.freesound_token,
        help="Freesound API token (or set FREESOUND_TOKEN env var)",
    )
    parser.add_argument("--out-dir", type=str, default=CFG.data_dir)
    parser.add_argument("--max-per-query", type=int, default=500)
    args = parser.parse_args()

    token = args.token
    if not token:
        raise SystemExit(
            "Error: Freesound API token required. "
            "Set FREESOUND_TOKEN env var or pass --token."
        )
    download_dataset(token, args.out_dir, args.max_per_query)
