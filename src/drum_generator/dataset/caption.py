"""
caption.py
----------
Shared caption-building utilities, audio loading, and CLAP embedding.
"""

import os
import re

import torch

from drum_generator.config import CFG

# ---------------------------------------------------------------------------
# Instrument + characteristic tag vocabularies
# ---------------------------------------------------------------------------

INSTRUMENT_MAP = {
    "kick": "kick drum",
    "bd": "kick drum",
    "bass drum": "kick drum",
    "808": "808 kick",
    "snare": "snare",
    "sd": "snare",
    "hihat": "hi-hat",
    "hi-hat": "hi-hat",
    "hh": "hi-hat",
    "open": "open hi-hat",
    "closed": "closed hi-hat",
    "clap": "clap",
    "tom": "tom",
    "crash": "crash cymbal",
    "ride": "ride cymbal",
    "rim": "rimshot",
    "cowbell": "cowbell",
    "perc": "percussion",
}

CHAR_TAGS = {
    "punchy",
    "tight",
    "dry",
    "wet",
    "roomy",
    "reverb",
    "warm",
    "bright",
    "dark",
    "snappy",
    "boomy",
    "sub",
    "crisp",
    "soft",
    "hard",
    "compressed",
    "distorted",
    "saturated",
    "clean",
    "acoustic",
    "electronic",
    "vintage",
    "trap",
    "house",
}

# ---------------------------------------------------------------------------
# Caption builders
# ---------------------------------------------------------------------------


def build_caption(meta: dict) -> str:
    """Build caption from Freesound metadata dict."""
    tags = [t.lower() for t in meta.get("tags", [])]
    instrument = next((INSTRUMENT_MAP[t] for t in tags if t in INSTRUMENT_MAP), "drum")
    chars = [t for t in tags if t in CHAR_TAGS]
    caption = instrument
    if chars:
        caption += ", " + ", ".join(chars[:4])
    desc = meta.get("description", "")[:80]
    if desc:
        caption += f". {desc}"
    return caption


def build_caption_from_filename(filepath: str) -> str:
    """Infer caption from filename and parent directory.

    Examples:
      'kicks/808_punchy_01.wav'  -> '808 kick, punchy'
      'snare_bright.flac'        -> 'snare, bright'
      'my_samples/hihat_open.wav' -> 'open hi-hat'
    """
    parts = os.path.normpath(filepath).split(os.sep)
    stem = os.path.splitext(parts[-1])[0]

    # Tokenize: split on _, -, spaces
    file_tokens = re.split(r"[_\-\s]+", stem.lower())
    parent_tokens = []
    if len(parts) > 1:
        parent_tokens = re.split(r"[_\-\s]+", parts[-2].lower())

    # Strip pure-digit tokens that look like sequence numbers, but keep
    # meaningful numeric tokens like "808" that appear in INSTRUMENT_MAP
    def _keep_token(t: str) -> bool:
        if not t.isdigit():
            return True
        return t in INSTRUMENT_MAP  # e.g. "808"

    file_tokens = [t for t in file_tokens if _keep_token(t)]
    parent_tokens = [t for t in parent_tokens if _keep_token(t)]

    # Prioritize filename tokens over directory tokens for instrument detection
    all_tokens = file_tokens + parent_tokens

    # Find instrument (strip trailing 's' for plurals like "kicks" -> "kick")
    instrument = "drum"
    for t in all_tokens:
        key = t.rstrip("s") if t.endswith("s") and t[:-1] in INSTRUMENT_MAP else t
        if key in INSTRUMENT_MAP:
            instrument = INSTRUMENT_MAP[key]
            break

    tokens = all_tokens

    # Find characteristics
    chars = [t for t in tokens if t in CHAR_TAGS]

    caption = instrument
    if chars:
        caption += ", " + ", ".join(chars[:4])
    return caption


def build_synthetic_caption(drum_type: str, characteristics: list[str]) -> str:
    """Build caption for a synthetically generated drum sound."""
    # Normalize drum type to display name
    type_map = {
        "kick": "kick drum",
        "snare": "snare",
        "hihat_closed": "closed hi-hat",
        "hihat_open": "open hi-hat",
        "clap": "clap",
        "tom": "tom",
        "rimshot": "rimshot",
        "cymbal": "crash cymbal",
    }
    instrument = type_map.get(drum_type, drum_type)
    caption = instrument
    if characteristics:
        caption += ", " + ", ".join(characteristics[:4])
    return caption


# ---------------------------------------------------------------------------
# Audio loading utility
# ---------------------------------------------------------------------------


def load_audio_file(
    path: str,
    target_sr: int = CFG.sample_rate,
    target_samples: int = CFG.n_samples,
) -> torch.Tensor:
    """Load any audio file -> mono float32 tensor of exact length.

    Supports wav, mp3, flac, ogg, aif/aiff via torchaudio.
    """
    import torchaudio

    wav, sr = torchaudio.load(path)

    # Mono
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)

    # Resample
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    # Pad or trim
    n = wav.shape[-1]
    if n < target_samples:
        wav = torch.nn.functional.pad(wav, (0, target_samples - n))
    else:
        wav = wav[:, :target_samples]

    return wav.squeeze(0)  # (target_samples,)


# ---------------------------------------------------------------------------
# CLAP embedder (singleton)
# ---------------------------------------------------------------------------


class ClapEmbedder:
    """Shared CLAP text embedding computation with disk caching.

    Use ClapEmbedder.get() to obtain the singleton instance, avoiding
    multiple loads of the ~600MB model.
    """

    _instance = None

    @classmethod
    def get(cls) -> "ClapEmbedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        from transformers import ClapModel, ClapProcessor

        print("[clap] loading CLAP model...")
        self.proc = ClapProcessor.from_pretrained("laion/larger_clap_general")
        self.model = ClapModel.from_pretrained("laion/larger_clap_general")
        self.model.eval()

    def embed(self, caption: str, cache_path: str | None = None) -> torch.Tensor:
        """Compute CLAP text embedding, with optional file cache."""
        if cache_path and os.path.exists(cache_path):
            return torch.load(cache_path, map_location="cpu", weights_only=True)

        inputs = self.proc(text=caption, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
            # transformers >=5: returns BaseModelOutputWithPooling
            emb = out.pooler_output if hasattr(out, "pooler_output") else out
            emb = emb.squeeze(0)  # (512,)

        if cache_path:
            torch.save(emb, cache_path)
        return emb
