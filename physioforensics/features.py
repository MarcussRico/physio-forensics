"""Physiological feature extraction: video -> named, interpretable scalars.

Two families:
  1. per-region quality    -- does this patch of skin carry a cardiac signal?
  2. cross-region agreement -- do the patches agree they belong to one body?

Family 2 is the harder constraint to fake and the project's actual contribution.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import rppg
from .regions import FACE_REGIONS, REGION_NAMES, RegionTraces

# Pairs of regions that must agree if a single heart is driving them.
WITHIN_FACE_PAIRS = list(itertools.combinations(FACE_REGIONS, 2))
FACE_NECK_PAIRS = [(r, "neck") for r in FACE_REGIONS]

MIN_FRAMES = 64


def _nan_safe(values, fn, default=np.nan):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return fn(vals) if vals else default


def compute_pulses(rt: RegionTraces, method: str = "pos") -> dict[str, np.ndarray]:
    """Extract one normalised pulse waveform per region."""
    pulses = {}
    for name in REGION_NAMES:
        trace = rt.traces.get(name)
        if trace is None or len(trace) < MIN_FRAMES:
            continue
        pulses[name] = rppg.extract_pulse(trace, rt.fps, method=method)
    return pulses


def compute_features(rt: RegionTraces, method: str = "pos") -> dict[str, float]:
    """Full feature vector for one video."""
    fps = rt.fps
    pulses = compute_pulses(rt, method=method)

    feats: dict[str, float] = {
        "fps": float(fps),
        "n_frames": float(rt.n_frames),
        "duration_sec": float(rt.n_frames / fps) if fps else np.nan,
        "face_detected": float(rt.face_detected),
        "n_regions": float(len(pulses)),
    }

    if not pulses:
        return feats

    # ---- family 1: per-region signal quality ---------------------------
    hrs, snrs, ents, pers, stabs = {}, {}, {}, {}, {}
    for name, sig in pulses.items():
        hrs[name] = rppg.estimate_hr_bpm(sig, fps)
        snrs[name] = rppg.pulse_snr_db(sig, fps)
        ents[name] = rppg.spectral_entropy(sig, fps)
        pers[name] = rppg.periodicity(sig, fps)
        stabs[name] = rppg.hr_stability(sig, fps)

        feats[f"hr_{name}"] = hrs[name]
        feats[f"snr_{name}"] = snrs[name]
        feats[f"entropy_{name}"] = ents[name]
        feats[f"periodicity_{name}"] = pers[name]
        feats[f"hr_stability_{name}"] = stabs[name]

    feats["snr_mean"] = _nan_safe(snrs.values(), np.mean)
    feats["snr_min"] = _nan_safe(snrs.values(), np.min)
    feats["entropy_mean"] = _nan_safe(ents.values(), np.mean)
    feats["periodicity_mean"] = _nan_safe(pers.values(), np.mean)
    feats["hr_stability_mean"] = _nan_safe(stabs.values(), np.mean)

    # ---- family 2: cross-region agreement ------------------------------
    # A single reference frequency for phase comparison: the consensus HR of
    # the face regions, which is more robust than any single region's peak.
    face_hr = [hrs[r] for r in FACE_REGIONS if r in hrs and np.isfinite(hrs[r])]
    consensus_hz = (np.median(face_hr) / 60.0) if face_hr else np.nan
    feats["hr_consensus_bpm"] = consensus_hz * 60.0 if np.isfinite(consensus_hz) else np.nan

    within_plv, within_corr = [], []
    for a, b in WITHIN_FACE_PAIRS:
        if a not in pulses or b not in pulses:
            continue
        plv = rppg.phase_locking_value(pulses[a], pulses[b], fps, consensus_hz)
        corr = rppg.signal_correlation(pulses[a], pulses[b])
        feats[f"plv_{a}__{b}"] = plv
        feats[f"corr_{a}__{b}"] = corr
        within_plv.append(plv)
        within_corr.append(corr)

    feats["plv_within_face_mean"] = _nan_safe(within_plv, np.mean)
    feats["plv_within_face_min"] = _nan_safe(within_plv, np.min)
    feats["corr_within_face_mean"] = _nan_safe(within_corr, np.mean)

    neck_plv, neck_corr, neck_hr_gap = [], [], []
    for a, b in FACE_NECK_PAIRS:
        if a not in pulses or b not in pulses:
            continue
        plv = rppg.phase_locking_value(pulses[a], pulses[b], fps, consensus_hz)
        corr = rppg.signal_correlation(pulses[a], pulses[b])
        feats[f"plv_{a}__{b}"] = plv
        neck_plv.append(plv)
        neck_corr.append(corr)
        if np.isfinite(hrs.get(a, np.nan)) and np.isfinite(hrs.get("neck", np.nan)):
            neck_hr_gap.append(abs(hrs[a] - hrs["neck"]))

    feats["plv_face_neck_mean"] = _nan_safe(neck_plv, np.mean)
    feats["corr_face_neck_mean"] = _nan_safe(neck_corr, np.mean)
    feats["hr_gap_face_neck"] = _nan_safe(neck_hr_gap, np.mean)

    # Spread of heart rate estimates. One body => one heart rate.
    all_hr = [v for v in hrs.values() if np.isfinite(v)]
    feats["hr_spread_bpm"] = float(np.max(all_hr) - np.min(all_hr)) if len(all_hr) >= 2 else np.nan
    feats["hr_std_bpm"] = float(np.std(all_hr)) if len(all_hr) >= 2 else np.nan

    # Single headline score: high when the body is physiologically coherent.
    parts = [feats.get("plv_within_face_mean"), feats.get("plv_face_neck_mean")]
    feats["coherence_score"] = _nan_safe(parts, np.mean)

    return feats


def feature_columns(features: dict[str, float]) -> list[str]:
    """Model-input columns: drop bookkeeping fields."""
    skip = {"fps", "n_frames", "duration_sec", "n_regions", "face_detected"}
    return [k for k in features if k not in skip]


def explain(features: dict[str, float]) -> list[str]:
    """Plain-language reasons behind the physiological verdict."""
    lines = []
    plv_face = features.get("plv_within_face_mean", np.nan)
    plv_neck = features.get("plv_face_neck_mean", np.nan)
    snr = features.get("snr_mean", np.nan)
    spread = features.get("hr_spread_bpm", np.nan)
    gap = features.get("hr_gap_face_neck", np.nan)

    if np.isfinite(snr):
        lines.append(
            f"Pulse signal strength across regions: {snr:+.1f} dB "
            f"({'a cardiac peak stands out from noise' if snr > 0 else 'no clear cardiac peak'})."
        )
    if np.isfinite(plv_face):
        lines.append(
            f"Phase agreement between forehead and cheeks: {plv_face:.2f} "
            f"({'consistent with one heart driving the face' if plv_face > 0.6 else 'regions are out of step with each other'})."
        )
    if np.isfinite(plv_neck):
        lines.append(
            f"Phase agreement between face and neck: {plv_neck:.2f} "
            f"({'face and body match' if plv_neck > 0.6 else 'face and body disagree, typical of a swap'})."
        )
    if np.isfinite(spread):
        lines.append(f"Spread of heart-rate estimates across regions: {spread:.1f} bpm.")
    if np.isfinite(gap):
        lines.append(f"Face-vs-neck heart-rate difference: {gap:.1f} bpm.")
    return lines
