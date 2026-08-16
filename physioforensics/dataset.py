"""Build a feature table from a directory of videos.

Expected layout, matching FaceForensics++ / Celeb-DF / the synthetic corpus:

    root/
      real/                  -> label 0
      fake_<generator>/      -> label 1, generator = <generator>

Generator names are kept in the table for leave-one-generator-out evaluation.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
from pathlib import Path

import pandas as pd

from .features import compute_features
from .regions import extract_region_traces

log = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
BOOKKEEPING = {"path", "label", "generator", "split", "error"}
NON_FEATURE = {"fps", "n_frames", "duration_sec", "n_regions", "face_detected"}


def discover_videos(root: str | Path) -> list[dict]:
    """Find labelled videos under root."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    items = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        name = folder.name
        if name == "real":
            label, generator = 0, "real"
        elif name.startswith("fake_"):
            label, generator = 1, name[len("fake_"):]
        else:
            log.warning("skipping unrecognised folder %s (expected 'real' or 'fake_*')", name)
            continue
        for video in sorted(folder.rglob("*")):
            if video.suffix.lower() in VIDEO_SUFFIXES:
                items.append({"path": str(video), "label": label, "generator": generator})
    return items


def _process_one(item: dict, method: str, roi_mode: str, max_frames: int | None) -> dict:
    row = dict(item)
    try:
        rt = extract_region_traces(item["path"], roi_mode=roi_mode, max_frames=max_frames)
        row.update(compute_features(rt, method=method))
        row["error"] = ""
    except Exception as exc:                       # keep one bad file from killing a run
        log.warning("failed on %s: %s", item["path"], exc)
        row["error"] = str(exc)
    return row


def build_feature_table(
    root: str | Path,
    method: str = "pos",
    roi_mode: str = "auto",
    max_frames: int | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    items = discover_videos(root)
    if not items:
        raise ValueError(f"no videos found under {root}")

    log.info("extracting features from %d videos (method=%s, roi=%s)", len(items), method, roi_mode)

    if workers > 1:
        with cf.ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_process_one, items,
                                 [method] * len(items),
                                 [roi_mode] * len(items),
                                 [max_frames] * len(items)))
    else:
        rows = [_process_one(it, method, roi_mode, max_frames) for it in items]

    return pd.DataFrame(rows)


# feature families, for the ablation that isolates the coherence contribution
QUALITY_PREFIXES = ("snr_", "entropy_", "periodicity_", "hr_stability_", "hr_")
COHERENCE_PREFIXES = ("plv_", "corr_", "hr_spread", "hr_std", "hr_gap", "coherence_")


def feature_names(df: pd.DataFrame, family: str = "all") -> list[str]:
    """Model-input columns for a given feature family: quality, coherence, or all."""
    numeric = df.select_dtypes(include="number").columns
    # all-NaN columns carry no information and make the imputer complain
    cols = [c for c in numeric
            if c not in BOOKKEEPING and c not in NON_FEATURE and not df[c].isna().all()]

    if family == "all":
        return cols
    if family == "coherence":
        return [c for c in cols if c.startswith(COHERENCE_PREFIXES)]
    if family == "quality":
        return [c for c in cols
                if c.startswith(QUALITY_PREFIXES) and not c.startswith(COHERENCE_PREFIXES)]
    raise ValueError(f"unknown feature family {family!r}")
