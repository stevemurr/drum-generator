"""
drum_generator.dataset
----------------------
Public API for dataset construction.

Usage:
    from drum_generator.dataset import build_dataset
    ds = build_dataset()  # uses CFG defaults

    from drum_generator.dataset import FreesoundDataset, DiskAudioDataset, SyntheticDrumDataset
"""

import os

from torch.utils.data import ConcatDataset, Dataset

from drum_generator.config import CFG
from drum_generator.dataset.augment import AugmentedDataset, build_transforms
from drum_generator.dataset.cache import CachedDACDataset
from drum_generator.dataset.caption import ClapEmbedder
from drum_generator.dataset.disk import DiskAudioDataset
from drum_generator.dataset.freesound import FreesoundDataset, cli_download, download_dataset
from drum_generator.dataset.synthetic import SyntheticDrumDataset

# Backward-compat alias
DrumDataset = FreesoundDataset

__all__ = [
    "build_dataset",
    "DrumDataset",
    "FreesoundDataset",
    "DiskAudioDataset",
    "SyntheticDrumDataset",
    "AugmentedDataset",
    "CachedDACDataset",
    "ClapEmbedder",
    "download_dataset",
    "cli_download",
]


def build_dataset(
    freesound_dir: str | None = None,
    disk_dirs: list[str] | None = None,
    disk_label_file: str | None = None,
    synthetic_size: int | None = None,
    synthetic_seed: int | None = None,
    augment: bool | None = None,
    augment_transforms: list[str] | None = None,
    augment_p: float | None = None,
    augment_multiplier: int | None = None,
    clap_cache_dir: str | None = None,
    cache: bool | None = None,
    cache_dir: str | None = None,
) -> Dataset:
    """Build a combined dataset from all configured sources + augmentation.

    When cache=True, pre-encodes all samples through DAC and caches to disk.
    Returns (dac_latent, clap_embed) instead of (waveform, clap_embed).

    Parameters fall back to CFG defaults when not specified.
    """
    # Resolve defaults from CFG
    freesound_dir = freesound_dir if freesound_dir is not None else CFG.data_dir
    disk_dirs = disk_dirs if disk_dirs is not None else CFG.disk_dirs
    disk_label_file = disk_label_file if disk_label_file is not None else CFG.disk_label_file
    synthetic_size = synthetic_size if synthetic_size is not None else CFG.synthetic_size
    synthetic_seed = synthetic_seed if synthetic_seed is not None else CFG.synthetic_seed
    do_augment = augment if augment is not None else CFG.augment
    aug_transform_names = augment_transforms if augment_transforms is not None else CFG.augment_transforms
    aug_p = augment_p if augment_p is not None else CFG.augment_p
    aug_multiplier = augment_multiplier if augment_multiplier is not None else CFG.augment_multiplier
    clap_dir = clap_cache_dir if clap_cache_dir is not None else CFG.clap_cache_dir
    do_cache = cache if cache is not None else CFG.cache
    dac_cache_dir = cache_dir if cache_dir is not None else CFG.dac_cache_dir

    clap = ClapEmbedder.get()
    sources: list[Dataset] = []

    # 1. Freesound data
    if freesound_dir and os.path.isdir(freesound_dir):
        has_json = any(f.endswith(".json") for f in os.listdir(freesound_dir))
        if has_json:
            sources.append(FreesoundDataset(freesound_dir, clap_embedder=clap))

    # 2. Disk audio files
    if disk_dirs:
        valid_dirs = [d for d in disk_dirs if os.path.isdir(d)]
        if valid_dirs:
            sources.append(
                DiskAudioDataset(
                    root_dirs=valid_dirs,
                    clap_embedder=clap,
                    cache_dir=clap_dir,
                    label_file=disk_label_file,
                )
            )

    # 3. Synthetic data
    if synthetic_size > 0:
        sources.append(
            SyntheticDrumDataset(
                size=synthetic_size,
                clap_embedder=clap,
                seed=synthetic_seed,
            )
        )

    if not sources:
        raise ValueError(
            f"No data sources found. Check that '{freesound_dir}' contains data, "
            f"or configure disk_dirs/synthetic_size."
        )

    # Combine sources
    combined: Dataset = ConcatDataset(sources) if len(sources) > 1 else sources[0]

    # 4. Augmentation wrapper
    if do_augment:
        transforms = build_transforms(aug_transform_names)
        if transforms:
            combined = AugmentedDataset(
                combined,
                transforms=transforms,
                p_each=aug_p,
                multiplier=aug_multiplier,
            )

    # 5. DAC latent caching
    if do_cache:
        combined = CachedDACDataset(combined, dac_cache_dir)

    return combined
