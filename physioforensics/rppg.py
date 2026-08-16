"""Classical rPPG signal extraction.

Implements the three standard chrominance-based algorithms for recovering a
blood-volume-pulse waveform from a sequence of mean RGB values taken over a
patch of skin:

    GREEN  -- Verkruysse et al. 2008. Baseline. Just the green channel.
    CHROM  -- de Haan & Jeanne 2013. Projects RGB onto a chrominance plane.
    POS    -- Wang et al. 2017. Projects onto a plane orthogonal to skin tone,
              which cancels most motion-induced intensity artefacts.

Everything here is pure numpy/scipy and runs on CPU in milliseconds. There is
deliberately no neural network in this module: on rPPG benchmarks the classical
projections remain competitive with (and often beat) learned extractors, and
they give us interpretable, auditable features.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

# Plausible human heart rate expressed in Hz (42-240 bpm).
HR_BAND_HZ = (0.7, 4.0)


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------

def detrend(x: np.ndarray, fps: float, cutoff_hz: float = 0.5) -> np.ndarray:
    """Remove slow illumination drift with a moving-average high-pass."""
    x = np.asarray(x, dtype=np.float64)
    win = max(3, int(round(fps / cutoff_hz)))
    if win % 2 == 0:
        win += 1
    if win >= len(x):
        return x - x.mean()
    kernel = np.ones(win) / win
    baseline = np.convolve(x, kernel, mode="same")
    # convolve edge effects: fall back to the global mean near the boundaries
    half = win // 2
    baseline[:half] = baseline[half]
    baseline[-half:] = baseline[-half - 1]
    return x - baseline


def bandpass(x: np.ndarray, fps: float, band=HR_BAND_HZ, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass restricted to plausible heart rates."""
    x = np.asarray(x, dtype=np.float64)
    nyq = fps / 2.0
    low = max(band[0] / nyq, 1e-4)
    high = min(band[1] / nyq, 0.99)
    if low >= high:
        return x - x.mean()
    b, a = sps.butter(order, [low, high], btype="band")
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        return x - x.mean()
    return sps.filtfilt(b, a, x)


def _normalise_windowed(c: np.ndarray) -> np.ndarray:
    """Divide each channel by its temporal mean inside the window."""
    mu = c.mean(axis=0)
    mu[mu == 0] = 1e-9
    return c / mu


# --------------------------------------------------------------------------
# extraction algorithms:  (T, 3) RGB traces -> (T,) pulse waveform
# --------------------------------------------------------------------------

def green(rgb: np.ndarray, fps: float) -> np.ndarray:
    """Baseline: the raw green channel, detrended."""
    return detrend(np.asarray(rgb, dtype=np.float64)[:, 1], fps)


def chrom(rgb: np.ndarray, fps: float, window_sec: float = 1.6) -> np.ndarray:
    """CHROM (de Haan & Jeanne 2013) with 50%-overlap Hann reconstruction."""
    rgb = np.asarray(rgb, dtype=np.float64)
    n = len(rgb)
    win = int(round(window_sec * fps))
    if win < 8 or n < win:
        return _chrom_global(rgb, fps)

    step = win // 2
    out = np.zeros(n)
    hann = np.hanning(win)

    for start in range(0, n - win + 1, step):
        seg = _normalise_windowed(rgb[start:start + win])
        xs = 3.0 * seg[:, 0] - 2.0 * seg[:, 1]
        ys = 1.5 * seg[:, 0] + seg[:, 1] - 1.5 * seg[:, 2]
        xf = bandpass(xs, fps)
        yf = bandpass(ys, fps)
        sy = yf.std()
        alpha = (xf.std() / sy) if sy > 1e-12 else 0.0
        pulse = xf - alpha * yf
        out[start:start + win] += hann * (pulse - pulse.mean())

    return detrend(out, fps)


def _chrom_global(rgb: np.ndarray, fps: float) -> np.ndarray:
    seg = _normalise_windowed(rgb)
    xs = 3.0 * seg[:, 0] - 2.0 * seg[:, 1]
    ys = 1.5 * seg[:, 0] + seg[:, 1] - 1.5 * seg[:, 2]
    xf, yf = bandpass(xs, fps), bandpass(ys, fps)
    sy = yf.std()
    alpha = (xf.std() / sy) if sy > 1e-12 else 0.0
    return xf - alpha * yf


def pos(rgb: np.ndarray, fps: float, window_sec: float = 1.6) -> np.ndarray:
    """POS (Wang et al. 2017), plane-orthogonal-to-skin, overlap-added.

    The projection matrix P = [[0, 1, -1], [-2, 1, 1]] maps temporally
    normalised RGB onto two signals whose combination is orthogonal to the
    specular (motion/intensity) direction of skin reflectance.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    n = len(rgb)
    win = int(round(window_sec * fps))
    if win < 8 or n < win:
        win = n
    if win < 4:
        return np.zeros(n)

    proj = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    out = np.zeros(n)

    for end in range(win, n + 1):
        start = end - win
        seg = _normalise_windowed(rgb[start:end])
        s = seg @ proj.T                      # (win, 2)
        s2 = s[:, 1].std()
        alpha = (s[:, 0].std() / s2) if s2 > 1e-12 else 0.0
        h = s[:, 0] + alpha * s[:, 1]
        out[start:end] += h - h.mean()

    return detrend(out, fps)


EXTRACTORS = {"green": green, "chrom": chrom, "pos": pos}


def extract_pulse(rgb: np.ndarray, fps: float, method: str = "pos") -> np.ndarray:
    """Run one extractor and return the band-limited pulse waveform."""
    if method not in EXTRACTORS:
        raise ValueError(f"unknown rPPG method {method!r}; choose from {sorted(EXTRACTORS)}")
    raw = EXTRACTORS[method](rgb, fps)
    filtered = bandpass(raw, fps)
    sd = filtered.std()
    return filtered / sd if sd > 1e-12 else filtered


# --------------------------------------------------------------------------
# spectral analysis
# --------------------------------------------------------------------------

def power_spectrum(x: np.ndarray, fps: float, nfft: int = 8192):
    """Hann-windowed, zero-padded periodogram. Returns (freqs_hz, power)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if len(x) < 4:
        return np.array([0.0]), np.array([0.0])
    nfft = max(nfft, len(x))
    win = np.hanning(len(x))
    spec = np.fft.rfft(x * win, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fps)
    return freqs, np.abs(spec) ** 2


def dominant_frequency(x: np.ndarray, fps: float, band=HR_BAND_HZ) -> float:
    """Frequency (Hz) of the strongest spectral peak inside the HR band."""
    freqs, power = power_spectrum(x, fps)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not mask.any():
        return float("nan")
    return float(freqs[mask][np.argmax(power[mask])])


def estimate_hr_bpm(x: np.ndarray, fps: float, band=HR_BAND_HZ) -> float:
    return dominant_frequency(x, fps, band) * 60.0


def pulse_snr_db(x: np.ndarray, fps: float, bandwidth_hz: float = 0.2) -> float:
    """Signal-to-noise ratio in the sense used by the rPPG literature.

    Power inside a narrow window around the dominant frequency *and* its first
    harmonic, divided by all remaining power in the heart-rate band. A genuine
    cardiac signal concentrates its energy at f0 and 2*f0; noise does not.
    """
    freqs, power = power_spectrum(x, fps)
    in_band = (freqs >= HR_BAND_HZ[0]) & (freqs <= HR_BAND_HZ[1])
    if not in_band.any() or power[in_band].sum() <= 0:
        return float("nan")

    f0 = dominant_frequency(x, fps)
    if not np.isfinite(f0):
        return float("nan")

    signal_mask = np.zeros_like(freqs, dtype=bool)
    for centre in (f0, 2.0 * f0):
        signal_mask |= np.abs(freqs - centre) <= bandwidth_hz
    signal_mask &= in_band

    sig = power[signal_mask].sum()
    noise = power[in_band & ~signal_mask].sum()
    if noise <= 0 or sig <= 0:
        return float("nan")
    return float(10.0 * np.log10(sig / noise))


def spectral_entropy(x: np.ndarray, fps: float) -> float:
    """Normalised Shannon entropy of the in-band spectrum.

    Low entropy => energy concentrated at one frequency (a heartbeat).
    High entropy => energy smeared across the band (noise).
    """
    freqs, power = power_spectrum(x, fps)
    mask = (freqs >= HR_BAND_HZ[0]) & (freqs <= HR_BAND_HZ[1])
    p = power[mask]
    total = p.sum()
    if total <= 0 or len(p) < 2:
        return float("nan")
    p = p / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(freqs[mask])))


def periodicity(x: np.ndarray, fps: float) -> float:
    """Height of the strongest autocorrelation peak at a plausible HR lag."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if x.std() < 1e-12 or len(x) < 8:
        return float("nan")
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / ac[0]
    lo = int(fps / HR_BAND_HZ[1])
    hi = min(int(fps / HR_BAND_HZ[0]), len(ac) - 1)
    if hi <= lo:
        return float("nan")
    return float(np.max(ac[lo:hi]))


def hr_stability(x: np.ndarray, fps: float, window_sec: float = 4.0) -> float:
    """Std-dev (bpm) of the heart rate estimated in sliding windows.

    A real heart rate wanders a little. A spurious peak picked out of noise
    jumps around wildly.
    """
    win = int(round(window_sec * fps))
    if win < 8 or len(x) < win * 2:
        return float("nan")
    step = max(1, win // 2)
    rates = [
        estimate_hr_bpm(x[s:s + win], fps)
        for s in range(0, len(x) - win + 1, step)
    ]
    rates = [r for r in rates if np.isfinite(r)]
    return float(np.std(rates)) if len(rates) >= 2 else float("nan")


# --------------------------------------------------------------------------
# cross-region agreement  --  the core of the method
# --------------------------------------------------------------------------

def narrowband(x: np.ndarray, fps: float, centre_hz: float, half_width: float = 0.3) -> np.ndarray:
    lo = max(centre_hz - half_width, 0.5)
    hi = min(centre_hz + half_width, fps / 2 - 0.05)
    if hi <= lo:
        return x
    return bandpass(x, fps, band=(lo, hi), order=3)


def phase_locking_value(a: np.ndarray, b: np.ndarray, fps: float, centre_hz: float) -> float:
    """Phase Locking Value between two pulse signals at the cardiac frequency.

    Both regions are filtered narrowly around the same frequency, the
    instantaneous phase is taken via the Hilbert transform, and we measure how
    constant their phase difference stays over time.

        PLV = 1  ->  the two patches of skin are driven by one heart
        PLV = 0  ->  their phase relationship is random

    This is the single most discriminative feature in the pipeline: a
    generator can paint a plausible-looking face, but making every region of
    that face share one coherent cardiac phase is a far harder constraint.
    """
    if not np.isfinite(centre_hz) or len(a) != len(b) or len(a) < 16:
        return float("nan")
    fa = narrowband(a, fps, centre_hz)
    fb = narrowband(b, fps, centre_hz)
    if fa.std() < 1e-12 or fb.std() < 1e-12:
        return float("nan")
    pa = np.angle(sps.hilbert(fa))
    pb = np.angle(sps.hilbert(fb))
    # discard edges, where the Hilbert transform is unreliable
    edge = max(1, len(pa) // 10)
    d = (pa - pb)[edge:-edge]
    if len(d) < 4:
        return float("nan")
    return float(np.abs(np.mean(np.exp(1j * d))))


def signal_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) < 4 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
