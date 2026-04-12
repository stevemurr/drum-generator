"""
Download a small curated selection of drum sounds from Freesound for audition.

Usage:
    python audition.py [--out-dir audition_samples] [--per-type 5]
"""

import argparse
import os
import time

import requests

TOKEN = "DptVP1lEgNwu3qoiMR8M7M1LeA7g9CaxGheiYbi3"

QUERIES = {
    "kick": "kick drum one shot",
    "snare": "snare drum one shot",
    "hihat_closed": "hi hat closed one shot",
    "hihat_open": "hi hat open one shot",
    "clap": "clap one shot",
    "tom": "tom drum one shot",
    "rimshot": "rim shot one shot",
    "cymbal": "cymbal crash one shot",
    "808_kick": "808 kick one shot",
    "perc": "percussion one shot",
}


def _get(url: str, **kwargs) -> requests.Response:
    """GET with retries."""
    kwargs.setdefault("timeout", 30)
    for attempt in range(3):
        try:
            r = requests.get(url, **kwargs)
            r.raise_for_status()
            return r
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt == 2:
                raise
            print(f"  retry ({e.__class__.__name__})...")
            time.sleep(2 ** attempt)


def search(query: str, page: int = 1) -> dict:
    r = _get(
        "https://freesound.org/apiv2/search/text/",
        params={
            "query": query,
            "filter": 'license:"Creative Commons 0" duration:[0.05 TO 2]',
            "fields": "id,name,tags,duration,avg_rating,num_ratings,previews",
            "sort": "rating_desc",
            "page_size": 15,
            "page": page,
            "token": TOKEN,
        },
    )
    return r.json()


def download(sound: dict, out_dir: str, prefix: str) -> str | None:
    url = sound["previews"].get("preview-hq-mp3")
    if not url:
        return None
    name = sound["name"].replace("/", "_").replace(" ", "_")[:40]
    fname = f"{prefix}_{sound['id']}_{name}.mp3"
    path = os.path.join(out_dir, fname)
    if os.path.exists(path):
        return path
    audio = _get(url).content
    with open(path, "wb") as f:
        f.write(audio)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="audition_samples")
    parser.add_argument("--per-type", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    total = 0
    for drum_type, query in QUERIES.items():
        print(f"\n--- {drum_type} ({query!r}) ---")
        data = search(query)
        results = data.get("results", [])

        count = 0
        for sound in results:
            if count >= args.per_type:
                break
            rating = sound.get("avg_rating", 0)
            n_ratings = sound.get("num_ratings", 0)
            dur = sound.get("duration", 0)

            path = download(sound, args.out_dir, drum_type)
            if path:
                print(f"  {os.path.basename(path):60s} {dur:.2f}s  rating={rating:.1f} ({n_ratings})")
                count += 1
                time.sleep(0.3)

        total += count

    print(f"\nDownloaded {total} samples to {args.out_dir}/")
    print(f"Listen: python listen.py {args.out_dir}")


if __name__ == "__main__":
    main()
