"""Reproducible offline infrasound processing. Amplitudes are uncalibrated FS."""
from dataclasses import asdict, dataclass
from fractions import Fraction
import numpy as np
from scipy import signal
import soundfile as sf

@dataclass(frozen=True)
class Settings:
    low_hz: float = 0.5
    high_hz: float = 20.0
    order: int = 8  # per-edge Butterworth order
    phase: str = 'zero-phase'
    speed: float = 20.0
    normalize: bool = False

    def validate(self, sample_rate):
        if not 0 < self.low_hz < self.high_hz <= 20:
            raise ValueError('Use 0 < lower cutoff < upper cutoff ≤ 20 Hz.')
        if self.high_hz >= sample_rate / 2:
            raise ValueError('Upper cutoff must be below the recording Nyquist frequency.')
        if not 1 <= self.speed <= 100:
            raise ValueError('Acceleration must be between 1× and 100×.')
        if self.phase not in ('zero-phase', 'causal'):
            raise ValueError('Unknown filtering mode.')
        if not 1 <= self.order <= 12:
            raise ValueError('Filter order must be between 1 and 12.')


def load_recording(path):
    data, sr = sf.read(path, dtype='float64', always_2d=True)
    if not data.size or not np.isfinite(data).all():
        raise ValueError('Recording is empty or contains non-finite samples.')
    return data, sr


def filter_sos(sr, settings):
    settings.validate(sr)
    return np.vstack([
        signal.butter(settings.order, settings.low_hz, 'highpass', fs=sr, output='sos'),
        signal.butter(settings.order, settings.high_hz, 'lowpass', fs=sr, output='sos'),
    ])


def isolate(data, sr, settings):
    """Filter a single channel. Zero-phase mode squares the magnitude response."""
    sos = filter_sos(sr, settings)
    data = np.asarray(data, dtype=float)
    if data.ndim != 1 or len(data) < max(128, 3 * (2 * len(sos) + 1)):
        raise ValueError('Select a longer interval (at least 128 samples).')
    if not np.isfinite(data).all():
        raise ValueError('Input contains non-finite samples.')
    if settings.phase == 'zero-phase':
        return signal.sosfiltfilt(sos, data)
    zi = signal.sosfilt_zi(sos) * data[0]
    return signal.sosfilt(sos, data, zi=zi)[0]


def accelerated(data, sr, speed, normalize=False, output_rate=48000):
    """Resample for playback at a standard output rate; pitch and time both change."""
    if not 1 <= speed <= 100:
        raise ValueError('Acceleration must be between 1× and 100×.')
    ratio = Fraction(output_rate / (sr * speed)).limit_denominator(10000)
    out = signal.resample_poly(data, ratio.numerator, ratio.denominator)
    peak = np.max(np.abs(out)) if len(out) else 0
    if normalize and peak > 0:
        out = out * (0.89 / peak)
    return out, output_rate


def spectrum(data, sr):
    n = min(len(data), max(256, int(sr * 20)))
    return signal.welch(data, sr, window='hann', nperseg=n, noverlap=n//2, detrend='constant', scaling='density')


def summary(data, sr, settings):
    f, p = spectrum(data, sr)
    band = (f >= settings.low_hz) & (f <= settings.high_hz)
    rms = float(np.sqrt(np.mean(data ** 2)))
    return dict(rms_fs=rms, rms_dbfs=float(20*np.log10(max(rms, 1e-15))),
                peak_fs=float(np.max(np.abs(data))),
                dominant_hz=float(f[band][np.argmax(p[band])]) if band.any() and rms > 1e-14 else None,
                psd_bin_spacing_hz=float(f[1]-f[0]), duration_seconds=len(data)/sr,
                audible_duration_seconds=len(data)/sr/settings.speed)


def response(sr, settings):
    f = np.geomspace(max(.01, settings.low_hz/10), min(sr*.499, 100), 1000)
    _, h = signal.sosfreqz(filter_sos(sr, settings), worN=f, fs=sr)
    magnitude = np.abs(h) ** (2 if settings.phase == 'zero-phase' else 1)
    return f, 20*np.log10(np.maximum(magnitude, 1e-10))
