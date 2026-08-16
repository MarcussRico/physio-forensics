"""Waveform visualization: the visual argument for cross-region phase coherence.

Renders forehead-vs-neck pulse traces for each synthetic class side by side.
Reuses the production feature pipeline (compute_pulses / compute_features)
rather than recomputing anything, so the plot can never drift out of sync
with the numbers the classifier actually sees.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import compute_features, compute_pulses
from .regions import RegionTraces
from .synth import synth_region_traces

DEFAULT_KINDS = ("real", "incoherent", "drifting", "swap")


def plot_pulse_comparison(
    out_path: str | Path,
    kinds: tuple[str, ...] = DEFAULT_KINDS,
    duration_sec: float = 12.0,
    fps: float = 30.0,
    hr_bpm: float = 68.0,
    seed: int = 7,
) -> Path:
    """Save a grid comparing forehead vs. neck pulse waveforms per class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_cols = 2
    n_rows = int(np.ceil(len(kinds) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.0 * n_rows), squeeze=False)
    flat_axes = axes.flat

    for ax, kind in zip(flat_axes, kinds):
        traces, truth = synth_region_traces(
            kind=kind, duration_sec=duration_sec, fps=fps, hr_bpm=hr_bpm, seed=seed,
        )
        rt = RegionTraces(
            traces=traces, fps=fps, n_frames=truth["n_frames"],
            source=f"synthetic:{kind}", roi_mode="fixed", face_detected=False,
        )
        pulses = compute_pulses(rt)
        feats = compute_features(rt)

        forehead = pulses.get("forehead")
        neck = pulses.get("neck")
        if forehead is None or neck is None:
            ax.axis("off")
            continue

        t = np.arange(len(forehead)) / fps
        plv = feats.get("plv_forehead__neck", float("nan"))

        ax.plot(t, forehead, lw=1.4, color="#2563eb", label="forehead")
        ax.plot(t, neck, lw=1.4, color="#dc2626", alpha=0.85, label="neck")
        title = kind if not np.isfinite(plv) else f"{kind}  (PLV = {plv:.2f})"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("time (s)")
        ax.set_yticks([])
        ax.set_xlim(float(t[0]), float(t[-1]))

    for ax in flat_axes:          # turn off any unused trailing subplots
        ax.axis("off")

    axes.flat[0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(
        "Forehead vs. neck pulse waveform, by synthetic class\n"
        "high PLV = one heart drives both patches; low PLV = it doesn't",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
