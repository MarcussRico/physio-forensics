"""Synthetic video generator: ground truth without dataset access.

The public deepfake corpora (FaceForensics++, Celeb-DF, Celeb-DF++) are all
gated behind approval forms that take days. This module removes that
dependency from the critical path: it renders videos whose physiology we
control exactly, so the extraction pipeline can be validated end to end before
any real data arrives.

Each video is a skin-toned field with four region patches whose brightness is
modulated by a synthetic blood-volume-pulse waveform. What varies between
classes is *how physically coherent* that modulation is:

    real            one heart rate, one stable phase, small transit delays
                    between regions -- what a real body must produce

    nopulse         no cardiac modulation at all -- an early-generation
                    forgery that ignores physiology entirely

    incoherent      every region invents its own heart rate -- a generator
                    that synthesises regions semi-independently

    drifting        correct average heart rate, but the phase performs a
                    random walk -- a generator that has learned the *spectrum*
                    of a pulse from training data without learning that the
                    phase must stay locked. This is the hard, realistic case,
                    and the one where per-region SNR features fail but
                    cross-region coherence still succeeds.

    swap            face regions carry one (weak, drifting) pulse while the
                    neck carries a genuine, different one -- the signature of
                    a face-swap forgery pasted onto a real body

These are physical models of failure modes, not imitations of any specific
generator. They exist to prove the pipeline measures what it claims to
measure. Headline numbers must come from real corpora.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .regions import REGION_NAMES, _FRAME_RELATIVE_BOXES

# Blood volume changes absorb green light most strongly, red least.
CHANNEL_GAIN = np.array([0.30, 1.00, 0.55])      # R, G, B
BASE_SKIN_RGB = np.array([196.0, 152.0, 128.0])  # a mid skin tone in RGB

# Pulse transit delay: the wave reaches the forehead, cheeks and neck at
# slightly different times. Small, but constant for a given body.
TRANSIT_PHASE = {
    "forehead": 0.00,
    "left_cheek": 0.18,
    "right_cheek": 0.21,
    "neck": 0.42,
}

CLASSES = ("real", "nopulse", "incoherent", "drifting", "swap")


def _bvp_waveform(t: np.ndarray, freq_hz: float, phase: np.ndarray | float) -> np.ndarray:
    """A blood-volume-pulse shape: fundamental plus a systolic harmonic."""
    theta = 2.0 * np.pi * freq_hz * t + phase
    return np.sin(theta) + 0.35 * np.sin(2.0 * theta + 0.4)


def _drifting_phase(n: int, fps: float, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Random-walk phase: preserves the spectrum, destroys phase locking."""
    steps = rng.normal(0.0, rate / np.sqrt(fps), size=n)
    return np.cumsum(steps)


def _motion_artifact(t: np.ndarray, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """In-band specular/motion intensity artifact.

    Head motion and changing specular reflection alter overall brightness by
    roughly the same *fraction* in every colour channel. Crucially this lands
    inside the heart-rate band (subjects nod and shift at ~1-2 Hz), so a
    bandpass filter cannot remove it.

    This is precisely the nuisance POS and CHROM are built to cancel: their
    projection coefficients sum to zero, so any perturbation common to all
    three channels drops out, while the raw green channel absorbs it directly.
    Without this term the synthetic data is unrealistically clean and the
    ordering between extractors reported in the literature does not reproduce.
    """
    if amplitude <= 0:
        return np.zeros_like(t)
    out = np.zeros_like(t)
    for _ in range(3):                                   # a few motion components
        f = rng.uniform(0.8, 2.6)
        out += np.sin(2.0 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return amplitude * out / 3.0


def synth_region_signals(
    kind: str = "real",
    duration_sec: float = 10.0,
    fps: float = 30.0,
    hr_bpm: float = 72.0,
    amplitude: float = 0.010,
    seed: int | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Generate the per-region modulation signals and the ground truth."""
    rng = np.random.default_rng(seed)
    n = int(round(duration_sec * fps))
    t = np.arange(n) / fps
    f0 = hr_bpm / 60.0

    truth = {"kind": kind, "hr_bpm": hr_bpm, "fps": fps, "n_frames": n}
    signals: dict[str, np.ndarray] = {}

    if kind == "real":
        # One heart, one phase, fixed transit delays. Respiratory sinus
        # arrhythmia (the heart rate rising and falling with the breath) keeps
        # this from being an unrealistically perfect sinusoid.
        #
        # The instantaneous frequency must be *integrated* into phase. Writing
        # sin(2*pi*f(t)*t) instead would make the frequency deviation grow with
        # elapsed time rather than oscillate, which biases the recovered rate
        # more and more as the clip gets longer.
        inst_freq = f0 * (1.0 + 0.02 * np.sin(2.0 * np.pi * 0.15 * t))
        base_phase = 2.0 * np.pi * np.cumsum(inst_freq) / fps
        for name in REGION_NAMES:
            signals[name] = amplitude * _bvp_waveform(t, 0.0, base_phase + TRANSIT_PHASE[name])

    elif kind == "nopulse":
        for name in REGION_NAMES:
            signals[name] = np.zeros(n)

    elif kind == "incoherent":
        for name in REGION_NAMES:
            f = rng.uniform(0.8, 3.0)
            signals[name] = amplitude * _bvp_waveform(t, f, rng.uniform(0, 2 * np.pi))
        truth["hr_bpm"] = np.nan

    elif kind == "drifting":
        for name in REGION_NAMES:
            phase = TRANSIT_PHASE[name] + _drifting_phase(n, fps, 2.2, rng)
            signals[name] = amplitude * _bvp_waveform(t, f0, phase)

    elif kind == "swap":
        neck_hr = hr_bpm * rng.uniform(1.25, 1.5)
        for name in REGION_NAMES:
            if name == "neck":
                signals[name] = amplitude * _bvp_waveform(t, neck_hr / 60.0, TRANSIT_PHASE[name])
            else:
                phase = TRANSIT_PHASE[name] + _drifting_phase(n, fps, 2.6, rng)
                signals[name] = 0.7 * amplitude * _bvp_waveform(t, f0, phase)
        truth["neck_hr_bpm"] = neck_hr

    else:
        raise ValueError(f"unknown synthetic class {kind!r}; choose from {CLASSES}")

    return signals, truth


def synth_region_traces(
    kind: str = "real",
    duration_sec: float = 10.0,
    fps: float = 30.0,
    hr_bpm: float = 72.0,
    amplitude: float = 0.010,
    noise: float = 0.002,
    drift: float = 0.02,
    motion: float = 0.0,
    seed: int | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """RGB traces straight from the model, bypassing video encoding.

    Used for fast unit tests of the signal-processing layer in isolation.

    ``motion`` adds an in-band, channel-common intensity artifact. It is
    applied to every class equally -- it is a property of the capture, not of
    the forgery, and making it class-dependent would leak the label.
    """
    rng = np.random.default_rng(seed)
    signals, truth = synth_region_signals(kind, duration_sec, fps, hr_bpm, amplitude, seed)
    n = truth["n_frames"]
    t = np.arange(n) / fps

    # Illumination drift shared by every region, as in a real recording.
    common = drift * np.sin(2.0 * np.pi * 0.05 * t + rng.uniform(0, 2 * np.pi))
    motion_common = _motion_artifact(t, motion, rng)

    traces = {}
    for name, sig in signals.items():
        # Motion hits each region a little differently: turning the head does
        # not change the forehead and the neck by the same amount.
        region_motion = motion_common * rng.uniform(0.6, 1.4)
        modulation = (1.0
                      + sig[:, None] * CHANNEL_GAIN[None, :]
                      + common[:, None]
                      + region_motion[:, None])
        obs = BASE_SKIN_RGB[None, :] * modulation
        obs = obs + rng.normal(0.0, noise * BASE_SKIN_RGB.mean(), size=obs.shape)
        traces[name] = obs

    truth["motion"] = motion
    return traces, truth


def render_video(
    path: str | Path,
    kind: str = "real",
    duration_sec: float = 10.0,
    fps: float = 30.0,
    hr_bpm: float = 72.0,
    amplitude: float = 0.010,
    motion: float = 0.0,
    width: int = 192,
    height: int = 144,
    pixel_noise: float = 3.0,
    seed: int | None = None,
) -> dict:
    """Render an actual video file exercising the full video -> verdict path.

    Per-pixel noise is essential rather than incidental: the pulse is ~1% of
    the skin tone, i.e. under two 8-bit levels. Without spatial noise to dither
    the quantiser, uint8 rounding would erase the signal before it reached the
    extractor. Real sensors provide this dither for free.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    signals, truth = synth_region_signals(kind, duration_sec, fps, hr_bpm, amplitude, seed)
    n = truth["n_frames"]
    t = np.arange(n) / fps
    common = 0.02 * np.sin(2.0 * np.pi * 0.05 * t + rng.uniform(0, 2 * np.pi))
    motion_common = _motion_artifact(t, motion, rng)

    boxes = {}
    region_motion = {}
    for name, (rx, ry, rw, rh) in _FRAME_RELATIVE_BOXES.items():
        x0, y0 = int(rx * width), int(ry * height)
        boxes[name] = (x0, y0, x0 + int(rw * width), y0 + int(rh * height))
        region_motion[name] = motion_common * rng.uniform(0.6, 1.4)

    writer, used_path = _open_writer(path, fps, width, height)
    base_bgr = BASE_SKIN_RGB[::-1]
    gain_bgr = CHANNEL_GAIN[::-1]

    for i in range(n):
        frame = np.tile(base_bgr * (1.0 + common[i]), (height, width, 1))
        for name, (x0, y0, x1, y1) in boxes.items():
            m = 1.0 + signals[name][i] * gain_bgr + region_motion[name][i]
            frame[y0:y1, x0:x1] *= m[None, None, :]
        frame += rng.normal(0.0, pixel_noise, size=frame.shape)
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))

    writer.release()
    truth["path"] = str(used_path)
    truth["motion"] = motion
    return truth


def _open_writer(path: Path, fps: float, width: int, height: int):
    """mp4v where available, MJPG/AVI as a fallback on headless builds."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if writer.isOpened():
        return writer, path
    alt = path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(alt), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open any video writer (tried mp4v and MJPG)")
    return writer, alt


def build_synthetic_dataset(
    out_dir: str | Path,
    n_per_class: int = 12,
    duration_sec: float = 10.0,
    fps: float = 30.0,
    classes: tuple[str, ...] = CLASSES,
    motion_range: tuple[float, float] = (0.0, 0.012),
    seed: int = 0,
) -> list[dict]:
    """Write a labelled corpus laid out like a real deepfake dataset.

        out_dir/real/<id>.mp4
        out_dir/fake_<class>/<id>.mp4

    The per-class folders act as distinct "generators", which lets the
    leave-one-generator-out protocol run before real data is available.
    """
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    manifest = []

    for kind in classes:
        folder = out_dir / ("real" if kind == "real" else f"fake_{kind}")
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            hr = float(rng.uniform(55, 95))
            amp = float(rng.uniform(0.007, 0.014))
            # Motion is drawn from the same distribution for every class, so
            # it cannot act as a shortcut the classifier could exploit.
            motion = float(rng.uniform(*motion_range))
            truth = render_video(
                folder / f"{kind}_{i:03d}.mp4",
                kind=kind,
                duration_sec=duration_sec,
                fps=fps,
                hr_bpm=hr,
                amplitude=amp,
                motion=motion,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            truth.update(label=0 if kind == "real" else 1, generator=kind)
            manifest.append(truth)

    return manifest
