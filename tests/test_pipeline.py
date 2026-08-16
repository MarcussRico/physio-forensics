"""Tests that assert physical correctness, not just that the code runs.

The central claim of the project is that a genuine recording carries a
spatially coherent cardiac signal and a forgery does not. These tests check
that claim against synthetic physiology whose ground truth we control.
"""

from __future__ import annotations

import numpy as np
import pytest

from physioforensics import rppg, synth
from physioforensics.features import compute_features
from physioforensics.regions import RegionTraces, extract_region_traces

FPS = 30.0


def _traces_to_rt(traces, n_frames, fps=FPS):
    return RegionTraces(traces=traces, fps=fps, n_frames=n_frames,
                        source="synthetic", roi_mode="fixed", face_detected=False)


# ---------------------------------------------------------------- extraction

# Tolerances encode the accuracy ordering reported in the rPPG literature:
# the chrominance projections cancel shared illumination drift, the raw green
# channel does not, so GREEN is measurably worse. Holding all three to the
# same bar would either hide that or force us to weaken the real extractors.
HR_TOLERANCE_BPM = {"pos": 4.0, "chrom": 4.0, "green": 8.0}


@pytest.mark.parametrize("method", ["pos", "chrom", "green"])
@pytest.mark.parametrize("true_hr", [54.0, 72.0, 96.0])
def test_extractor_recovers_known_heart_rate(method, true_hr):
    """Every extractor must find the heart rate we actually injected."""
    traces, truth = synth.synth_region_traces(
        kind="real", hr_bpm=true_hr, duration_sec=15.0, seed=42)
    pulse = rppg.extract_pulse(traces["forehead"], FPS, method=method)
    assert abs(rppg.estimate_hr_bpm(pulse, FPS) - true_hr) < HR_TOLERANCE_BPM[method]


def _mean_hr_error(method, motion, hrs=(54.0, 66.0, 78.0, 90.0, 102.0)):
    errs = []
    for i, hr in enumerate(hrs):
        traces, _ = synth.synth_region_traces(
            kind="real", hr_bpm=hr, duration_sec=15.0, motion=motion, seed=42 + i)
        pulse = rppg.extract_pulse(traces["forehead"], FPS, method=method)
        errs.append(abs(rppg.estimate_hr_bpm(pulse, FPS) - hr))
    return float(np.mean(errs))


def test_pos_beats_green_under_motion():
    """POS must survive in-band motion artifacts that destroy the green channel.

    This is the reason the pipeline defaults to POS rather than the simpler
    green-channel baseline. With no motion the two are comparable; once a
    channel-common intensity artifact lands inside the heart-rate band, the
    projection cancels it and the raw green channel does not.
    """
    assert _mean_hr_error("pos", motion=0.030) < _mean_hr_error("green", motion=0.030)


def test_green_baseline_is_adequate_without_motion():
    """Stated explicitly so the motion result above is not over-claimed."""
    assert _mean_hr_error("green", motion=0.0) < 5.0


def test_no_pulse_video_yields_low_snr():
    traces, truth = synth.synth_region_traces(kind="nopulse", duration_sec=15.0, seed=1)
    pulse = rppg.extract_pulse(traces["forehead"], FPS)
    assert rppg.pulse_snr_db(pulse, FPS) < 3.0


def test_bandpass_rejects_out_of_band_energy():
    t = np.arange(int(20 * FPS)) / FPS
    x = np.sin(2 * np.pi * 0.05 * t) + 0.01 * np.sin(2 * np.pi * 1.2 * t)
    assert abs(rppg.dominant_frequency(rppg.bandpass(x, FPS), FPS) - 1.2) < 0.1


# ----------------------------------------------------------------- coherence

def test_real_video_is_phase_coherent_across_regions():
    traces, truth = synth.synth_region_traces(kind="real", duration_sec=15.0, seed=3)
    feats = compute_features(_traces_to_rt(traces, truth["n_frames"]))
    assert feats["plv_within_face_mean"] > 0.8
    assert feats["plv_face_neck_mean"] > 0.8
    assert feats["hr_spread_bpm"] < 5.0


@pytest.mark.parametrize("kind", ["nopulse", "incoherent", "drifting"])
def test_forgeries_lose_phase_coherence(kind):
    real, rt_truth = synth.synth_region_traces(kind="real", duration_sec=15.0, seed=5)
    fake, ft_truth = synth.synth_region_traces(kind=kind, duration_sec=15.0, seed=5)

    real_plv = compute_features(_traces_to_rt(real, rt_truth["n_frames"]))["plv_within_face_mean"]
    fake_plv = compute_features(_traces_to_rt(fake, ft_truth["n_frames"]))["plv_within_face_mean"]
    assert fake_plv < real_plv - 0.2


def test_face_swap_shows_face_neck_mismatch():
    """A swapped face on a real body must disagree with its own neck."""
    traces, truth = synth.synth_region_traces(kind="swap", duration_sec=15.0, seed=9)
    feats = compute_features(_traces_to_rt(traces, truth["n_frames"]))
    assert feats["plv_face_neck_mean"] < 0.6
    assert feats["hr_gap_face_neck"] > 5.0


def test_coherence_catches_what_signal_quality_misses():
    """The contribution claim, stated as an executable assertion.

    The 'incoherent' forgery has a strong, clean spectral peak in every region
    -- a quality-only detector sees a healthy pulse and passes it. Only the
    cross-region phase test rejects it.
    """
    real, r_truth = synth.synth_region_traces(kind="real", duration_sec=15.0, seed=11)
    fake, f_truth = synth.synth_region_traces(kind="incoherent", duration_sec=15.0, seed=11)

    rf = compute_features(_traces_to_rt(real, r_truth["n_frames"]))
    ff = compute_features(_traces_to_rt(fake, f_truth["n_frames"]))

    assert ff["snr_mean"] > rf["snr_mean"] - 3.0      # quality gives no signal
    assert ff["plv_within_face_mean"] < rf["plv_within_face_mean"] - 0.3


# ------------------------------------------------------------- video round trip

def test_end_to_end_video_roundtrip(tmp_path):
    """Render a real video file, read it back, recover the heart rate.

    Guards the parts unit tests on traces cannot see: colour conversion, uint8
    quantisation, codec compression and ROI placement.
    """
    truth = synth.render_video(tmp_path / "real.mp4", kind="real",
                               hr_bpm=72.0, duration_sec=12.0, seed=17)
    rt = extract_region_traces(truth["path"], roi_mode="fixed")

    assert rt.n_frames > 300
    feats = compute_features(rt)
    assert abs(feats["hr_consensus_bpm"] - 72.0) < 6.0
    assert feats["plv_within_face_mean"] > 0.6


def test_video_forgery_separates_from_real(tmp_path):
    real = synth.render_video(tmp_path / "r.mp4", kind="real", duration_sec=12.0, seed=21)
    fake = synth.render_video(tmp_path / "f.mp4", kind="incoherent", duration_sec=12.0, seed=21)

    rf = compute_features(extract_region_traces(real["path"], roi_mode="fixed"))
    ff = compute_features(extract_region_traces(fake["path"], roi_mode="fixed"))
    assert ff["coherence_score"] < rf["coherence_score"]


# ------------------------------------------------------------------- learning

def test_leave_one_generator_out_runs_and_beats_chance(tmp_path):
    """The full protocol must execute and clear chance on held-out generators."""
    from physioforensics.dataset import build_feature_table
    from physioforensics.train import evaluate_logo

    synth.build_synthetic_dataset(tmp_path / "ds", n_per_class=6,
                                  duration_sec=8.0, seed=2)
    df = build_feature_table(tmp_path / "ds", roi_mode="fixed")
    results = evaluate_logo(df, family="all")

    assert results, "leave-one-generator-out produced no folds"
    mean = [r for r in results if r.group == "MEAN"][0]
    assert mean.auc > 0.6
