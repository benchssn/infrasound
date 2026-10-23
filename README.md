# Infrasound analysis

A local Python desktop tool for examining recorded infrasound and listening to an accelerated version. No website or upload is required.

## Run

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python infrasound.py
# Or open a recording directly:
python infrasound.py recording.wav
```

Linux may also require your distribution's PortAudio library and Qt platform dependencies. macOS/Windows sounddevice wheels normally include PortAudio.

On this Mac, the project environment is already installed. Double-click **Launch Infrasound.command**, or run `.venv/bin/python infrasound.py` from this folder.

Run the numerical checks with `.venv/bin/python -m unittest -v test_processing`.

## Workflow

1. Open a WAV/FLAC (AIFF/OGG also supported by libsndfile), or click **Demo** for 8, 14, and 40 Hz test tones.
2. Choose a channel and an interval in original seconds. Channel averaging is explicit because opposing channel phases can cancel signals.
3. Set lower/upper cutoffs, filter order per edge, phase mode, and acceleration. Click **Analyze selection** after any change.
4. Inspect waveform, Welch power spectral density, isolated-signal spectrogram, and filter response. Plot axes use original recording coordinates. PyQtGraph supports zooming, panning, and a right-click plot menu.
5. Listen at low volume. Original and isolated listening both use the selected acceleration, allowing a direct comparison. Volume changes affect listening only; normalization affects isolated listening and accelerated export only.
6. Export an accelerated 48 kHz float WAV, an unnormalized filtered float WAV at the input sample rate, and a JSON processing record (including source SHA-256, channel, selection, parameters, measurements, and library versions). The current plot can be exported as PNG.

## Scientific interpretation

- `processing.py` contains independently usable NumPy/SciPy functions; `infrasound.py` provides the PySide6/PyQtGraph interface.
- Filtering cascades a Butterworth high-pass and low-pass, each with the selected order. The default band is 0.5–20 Hz. This removes DC and drift; it does **not** preserve all frequencies approaching 0 Hz.
- Zero-phase mode uses forward/backward filtering (`sosfiltfilt`): effective order doubles, magnitude response is squared, and each nominal cutoff is approximately −6 dB rather than the causal filter's −3 dB. Causal mode uses a single forward pass and has phase delay. Both modes process offline; this is not a live acquisition tool.
- Finite filters have transition bands: content above 20 Hz is attenuated, not eliminated with a brick-wall cutoff. Inspect the response plot when selecting settings.
- Filtering is applied to the selected interval. Shaded waveform edges indicate a heuristic 3/lower-cutoff-second transient caution, capped at half the selection; this is not a rigorous settling-time guarantee. Use long intervals with margins and inspect edge behavior before interpreting events.
- PSD uses a Hann window, 50% overlap, and up to 20-second Welch segments. Reported bin spacing is not the same as true spectral resolving power. Spectrogram uses up to 8-second windows with 75% overlap. Short recordings have coarser frequency information.
- RMS and peak values use filtered audio **before normalization**, including selection edges. Amplitudes are digital full scale, with dBFS defined here as 20 log10(RMS / 1 FS). PSD is FS²/Hz. There is no sensor calibration or conversion to pascals/dB SPL.
- Acceleration changes pitch and duration together; at 20×, 8 Hz becomes 160 Hz and 60 seconds becomes 3 seconds. Resampling includes anti-alias filtering. The exported float WAV preserves numerical values above ±1 if present; playback is clipped to the device range after applying volume. Normalization targets a peak of 0.89 FS.
- Most everyday microphones and recording paths suppress infrasound. This software cannot recover frequencies absent from the recording or distinguish sensor drift, wind noise, and genuine acoustic signals by itself.
- Files are loaded and processed in memory. Use manageable recordings or prepare smaller files for very long acquisitions.
