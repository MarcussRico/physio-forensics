"""Face region tracking and per-region RGB trace extraction.

We deliberately pull traces from several *separate* patches of skin rather than
one averaged face crop. The whole method rests on comparing regions against
each other, so keeping them separate is the point.

Regions:
    forehead, left_cheek, right_cheek   -- inside the synthesised face
    neck                                -- below the face box

The neck matters. In a face-swap forgery the neck is usually the original,
unmodified person, so it still carries a genuine pulse while the face does not.
Face-vs-neck disagreement is therefore direct evidence of a swap.

Two ROI modes:
    auto  -- OpenCV Haar cascade face detection, boxes derived from face geometry
    fixed -- boxes at fixed normalised coordinates (used by the synthetic
             validation harness, and as a fallback for pre-cropped datasets)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REGION_NAMES = ("forehead", "left_cheek", "right_cheek", "neck")
FACE_REGIONS = ("forehead", "left_cheek", "right_cheek")

# Boxes as (x, y, w, h) fractions of the detected face box. y > 1 sits below it.
_FACE_RELATIVE_BOXES = {
    "forehead":    (0.30, 0.04, 0.40, 0.16),
    "left_cheek":  (0.10, 0.55, 0.24, 0.22),
    "right_cheek": (0.66, 0.55, 0.24, 0.22),
    "neck":        (0.28, 1.04, 0.44, 0.20),
}

# Boxes as fractions of the whole frame, for roi_mode="fixed".
_FRAME_RELATIVE_BOXES = {
    "forehead":    (0.40, 0.10, 0.20, 0.14),
    "left_cheek":  (0.22, 0.42, 0.18, 0.16),
    "right_cheek": (0.60, 0.42, 0.18, 0.16),
    "neck":        (0.38, 0.72, 0.24, 0.16),
}


@dataclass
class RegionTraces:
    """Mean RGB per region per frame, plus capture metadata."""
    traces: dict[str, np.ndarray]   # region -> (T, 3) float array, RGB order
    fps: float
    n_frames: int
    source: str
    roi_mode: str
    face_detected: bool

    def available(self) -> list[str]:
        return [r for r, t in self.traces.items() if len(t) > 0]


def _clip_box(box, width: int, height: int):
    x, y, w, h = (int(round(v)) for v in box)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return x0, y0, x1, y1


def _boxes_from_face(face, width: int, height: int):
    fx, fy, fw, fh = face
    out = {}
    for name, (rx, ry, rw, rh) in _FACE_RELATIVE_BOXES.items():
        clipped = _clip_box((fx + rx * fw, fy + ry * fh, rw * fw, rh * fh), width, height)
        if clipped:
            out[name] = clipped
    return out


def _boxes_from_frame(width: int, height: int):
    out = {}
    for name, (rx, ry, rw, rh) in _FRAME_RELATIVE_BOXES.items():
        clipped = _clip_box((rx * width, ry * height, rw * width, rh * height), width, height)
        if clipped:
            out[name] = clipped
    return out


_cascade = None


def _get_cascade():
    global _cascade
    if _cascade is None:
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        _cascade = cv2.CascadeClassifier(str(path))
    return _cascade


def detect_face(frame_bgr) -> tuple | None:
    """Return the largest detected face box (x, y, w, h), or None."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _get_cascade().detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                            minSize=(60, 60))
    if len(faces) == 0:
        return None
    return tuple(max(faces, key=lambda f: f[2] * f[3]))


def _skin_mask(patch_bgr) -> np.ndarray:
    """Loose YCrCb skin-tone mask; falls back to the full patch if too strict."""
    ycrcb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 180, 127)) > 0
    if mask.mean() < 0.15:
        return np.ones(mask.shape, dtype=bool)
    return mask


def extract_region_traces(
    video_path: str | Path,
    roi_mode: str = "auto",
    max_frames: int | None = None,
    redetect_every: int = 30,
    use_skin_mask: bool = True,
) -> RegionTraces:
    """Walk a video and accumulate the mean RGB of each region per frame."""
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(fps) or fps <= 1:
        fps = 30.0

    series: dict[str, list] = {r: [] for r in REGION_NAMES}
    boxes = None
    face_detected = False
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and idx >= max_frames:
            break
        h, w = frame.shape[:2]

        if boxes is None or (roi_mode == "auto" and idx % redetect_every == 0):
            if roi_mode == "auto":
                face = detect_face(frame)
                if face is not None:
                    boxes = _boxes_from_face(face, w, h)
                    face_detected = True
                elif boxes is None:
                    boxes = _boxes_from_frame(w, h)   # fall back, keep going
            else:
                boxes = _boxes_from_frame(w, h)

        for name in REGION_NAMES:
            box = boxes.get(name)
            if box is None:
                series[name].append([np.nan, np.nan, np.nan])
                continue
            x0, y0, x1, y1 = box
            patch = frame[y0:y1, x0:x1]
            if patch.size == 0:
                series[name].append([np.nan, np.nan, np.nan])
                continue
            if use_skin_mask:
                mask = _skin_mask(patch)
                pix = patch[mask]
            else:
                pix = patch.reshape(-1, 3)
            if len(pix) == 0:
                series[name].append([np.nan, np.nan, np.nan])
                continue
            b, g, r = pix.mean(axis=0)
            series[name].append([r, g, b])       # store RGB order

        idx += 1

    cap.release()

    traces = {}
    for name, rows in series.items():
        arr = np.asarray(rows, dtype=np.float64)
        if len(arr) == 0 or np.isnan(arr).all():
            traces[name] = np.zeros((0, 3))
        else:
            traces[name] = _interpolate_nans(arr)

    return RegionTraces(
        traces=traces,
        fps=float(fps),
        n_frames=idx,
        source=video_path,
        roi_mode=roi_mode,
        face_detected=face_detected,
    )


def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """Fill dropped frames by linear interpolation so filtering stays valid."""
    out = arr.copy()
    n = len(out)
    t = np.arange(n)
    for c in range(out.shape[1]):
        col = out[:, c]
        bad = np.isnan(col)
        if bad.all():
            out[:, c] = 0.0
        elif bad.any():
            out[bad, c] = np.interp(t[bad], t[~bad], col[~bad])
    return out
