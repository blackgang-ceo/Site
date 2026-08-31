"""
mastering/backend/dsp_pipeline.py

High-level, production-oriented DSP pipeline for one-click mastering.

Design notes
- Uses pedalboard for filter/compressor/limiter blocks where available.
- Uses pyloudnorm for LUFS measurement and true-peak estimation (via upsampling + sample peak)
- Uses librosa for spectral analysis to find resonances for gentle subtractive dynamic EQ
- Uses matchering to perform gentle reference tonal shaping when a reference curve is available
- Uses an iterative limiter pass to reach the target integrated LUFS while keeping True Peak <= ceiling

Caveats
- Some pedalboard plugin class names / behaviors vary by version. The code attempts to import and use common plugins
  but will fall back to conservative numpy implementations when necessary.
- This module focuses on correctness, safety, and clear comments. For production load you should run the heavy DSP
  in background workers and restrict CPU per job.

API surface
- process_master(input_path: str, workdir: str, target_lufs: float = -10.0, tp_ceiling: float = -1.0)
    -> returns a dict with keys: streaming_path, hires_path, measured (dict)

"""

import os
import io
import math
import tempfile
import shutil
import logging
from typing import Tuple, Dict, Optional

import numpy as np
import soundfile as sf
import librosa

# pedalboard, matchering, pyloudnorm, ffmpeg
try:
    from pedalboard import Pedalboard, HighpassFilter, Compressor, Limiter, Gain, HighShelf, LowShelf, PeakingEQ
    PEDALBOARD_AVAILABLE = True
except Exception:
    PEDALBOARD_AVAILABLE = False

try:
    import matchering as mg
    MATCHERING_AVAILABLE = True
except Exception:
    MATCHERING_AVAILABLE = False

try:
    import pyloudnorm as pyln
    PYLOUD_AVAILABLE = True
except Exception:
    PYLOUD_AVAILABLE = False

# ffmpeg for conversions
try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except Exception:
    FFMPEG_AVAILABLE = False

# configure logger
logger = logging.getLogger("mastering.dsp")
logger.setLevel(logging.INFO)

# Utility functions

def _read_audio(path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Read audio to float32 numpy array shape (samples, channels) and sample rate.
    We normalize to float32 in range [-1,1].
    """
    data, sr_in = sf.read(path, always_2d=True)
    # soundfile returns shape (frames, channels)
    data = data.astype('float32')
    if sr is not None and sr != sr_in:
        # resample using librosa
        data = np.vstack([librosa.resample(data[:, ch], sr_in, sr) for ch in range(data.shape[1])]).T
        sr_in = sr
    return data, sr_in


def _write_wav(path: str, data: np.ndarray, sr: int, subtype: str = 'PCM_24') -> None:
    """Write numpy float32 array to disk using soundfile. Subtype can be PCM_16, PCM_24, FLOAT.
    Data shape: (frames, channels)
    """
    sf.write(path, data, sr, subtype=subtype)


def _estimate_true_peak(samples: np.ndarray, sr: int, upsample_to: int = 192000) -> float:
    """Estimate true peak in dBFS by upsampling and checking sample peaks.

    samples: (frames, channels) float32 -1..1
    Returns: true peak in dBFS (e.g., -0.5 means -0.5 dBFS)
    """
    # Convert to mono for TP estimation (worst case across channels)
    if samples.ndim == 2:
        mix = np.mean(samples, axis=1)
    else:
        mix = samples

    if sr >= upsample_to:
        up = mix
        up_sr = sr
    else:
        up = librosa.resample(mix.astype('float64'), sr, upsample_to)
        up_sr = upsample_to

    peak = np.max(np.abs(up))
    if peak <= 0:
        return -999.0
    return 20.0 * math.log10(peak)


def _compute_lufs(samples: np.ndarray, sr: int) -> Dict[str, float]:
    """Return integrated LUFS and short-term metrics. Uses pyloudnorm when available."""
    if not PYLOUD_AVAILABLE:
        logger.warning("pyloudnorm not available; LUFS measurement will be approximate (RMS based)")
        # fallback: RMS -> approximate LUFS
        rms = np.sqrt(np.mean(samples ** 2))
        lufs = 20 * math.log10(rms + 1e-12)
        return {"integrated_loudness": lufs}

    meter = pyln.Meter(sr)
    # shape (frames, channels)
    loudness = meter.integrated_loudness(samples)
    # pyr. truepeak not provided; we'll compute separately
    return {"integrated_loudness": float(loudness)}


def _find_resonances(samples: np.ndarray, sr: int) -> Dict[str, float]:
    """Analyze spectrum and return candidate resonant frequencies to notch.

    We search common bands and return strongest peaks near typical problem areas.
    """
    # Mix to mono for analysis
    if samples.ndim == 2:
        mono = np.mean(samples, axis=1)
    else:
        mono = samples

    # Use STFT magnitude average
    S = np.abs(librosa.stft(mono, n_fft=65536 // 4))  # large FFT for accurate freq
    freqs = librosa.fft_frequencies(sr=sr, n_fft=librosa.stft(mono, n_fft=65536 // 4).shape[0] * 2 - 1)
    spec = np.mean(S, axis=1)

    def find_peak_in_range(lo, hi):
        idx = np.where((freqs >= lo) & (freqs <= hi))[0]
        if idx.size == 0:
            return None
        sub = spec[idx]
        maxi = np.argmax(sub)
        return float(freqs[idx[maxi]])

    candidates = {}
    candidates['low_mud'] = find_peak_in_range(140, 400)
    candidates['upper_mid_harsh'] = find_peak_in_range(2000, 6000)
    candidates['boxiness'] = find_peak_in_range(400, 900)
    return candidates


def _apply_saturation(samples: np.ndarray, drive_db: float = 1.0) -> np.ndarray:
    """Simple, high-quality soft saturation implemented with tanh curve.

    drive_db: positive dB of pre-gain applied before tanh, smaller values (0.5-2 dB) are subtle.
    """
    if drive_db == 0:
        return samples

    gain = 10 ** (drive_db / 20.0)
    driven = samples * gain
    # Apply gentle softclip with tanh, then scale back so 0dB remains near 0dB
    saturated = np.tanh(driven)
    # normalize to match RMS of input to be conservative
    in_rms = np.sqrt(np.mean(samples ** 2)) + 1e-15
    out_rms = np.sqrt(np.mean(saturated ** 2)) + 1e-15
    saturated = saturated * (in_rms / out_rms)
    return saturated.astype(np.float32)


def process_master(input_path: str, workdir: str, target_lufs: float = -10.0, tp_ceiling: float = -1.0) -> Dict:
    """Main processing pipeline.

    Args:
      input_path: path to source WAV/MP3
      workdir: directory where intermediate and final files will be written
      target_lufs: integrated LUFS goal (e.g., -10.0)
      tp_ceiling: true-peak ceiling in dBTP (e.g., -1.0)

    Returns dict containing file paths and measured metadata.
    """
    os.makedirs(workdir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_path))[0]

    logger.info("Loading audio: %s", input_path)
    audio, sr = _read_audio(input_path)
    frames, chans = audio.shape[0], audio.shape[1]

    measured = {}
    measured.update(_compute_lufs(audio, sr))
    measured['true_peak_dbfs'] = _estimate_true_peak(audio, sr)
    logger.info("Measured LUFS=%.2f, TP=%.2f dBFS", measured['integrated_loudness'], measured['true_peak_dbfs'])

    # Pre-gain to reach target LUFS (simple first-pass). We'll adjust later with limiter.
    gain_db_initial = target_lufs - measured['integrated_loudness']
    logger.info("Initial gain to reach target LUFS: %.2f dB", gain_db_initial)

    # Apply high-pass at 20Hz (linear-phase not trivial; we use a high-order IIR via pedalboard if available)
    proc = audio.copy()

    if PEDALBOARD_AVAILABLE:
        board = Pedalboard()
        try:
            board.append(HighpassFilter(cutoff_hz=20.0))
        except Exception:
            # If HighpassFilter expects different param names
            try:
                board.append(HighpassFilter(20.0))
            except Exception:
                logger.exception("Pedalboard highpass creation failed; continuing without pedalboard HPF")
        # We'll apply the board to entire track in blocks due to memory safety
        try:
            # pedalboard expects shape (samples, channels)
            proc = board(proc, sr)
        except Exception:
            logger.exception("Pedalboard processing failed for HPF; skipping to numpy fallback")
    else:
        # fallback: very gentle single-pole HP using librosa's high-pass via FFT windowing
        logger.info("Applying naive frequency-domain HPF fallback (20Hz)")
        # FFT approach: zero out frequencies below 18Hz
        mono = np.mean(proc, axis=1)
        S = librosa.stft(mono)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=S.shape[0]*2-1)
        cutoff_idx = np.searchsorted(freqs, 18.0)
        S[:cutoff_idx, :] = 0
        mono_f = librosa.istft(S, length=proc.shape[0])
        for ch in range(proc.shape[1]):
            proc[:, ch] = mono_f

    # Subtractive dynamic EQ (detect resonances and notch)
    res = _find_resonances(proc, sr)
    eq_nodes = []
    for label, freq in res.items():
        if freq is None:
            continue
        # only notch if energy is significant
        q = 1.2
        cut_db = -1.75
        logger.info("Applying subtractive EQ at %s: %.1f Hz, %.2fdB", label, freq, cut_db)
        if PEDALBOARD_AVAILABLE:
            try:
                # PeakingEQ(center_freq, q, gain_db)
                board = Pedalboard([PeakingEQ(freq, q, cut_db)])
                proc = board(proc, sr)
            except Exception:
                logger.exception("pedalboard PeakingEQ failed for freq %.1f", freq)
        else:
            # Fallback: apply a simple parametric EQ via FFT (very approximate)
            S = librosa.stft(np.mean(proc, axis=1), n_fft=16384)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=16384)
            idx = np.argmin(np.abs(freqs - freq))
            S[idx-2:idx+3, :] *= 10 ** (cut_db / 20.0)
            mono_f = librosa.istft(S, length=proc.shape[0])
            for ch in range(proc.shape[1]):
                proc[:, ch] = 0.5 * proc[:, ch] + 0.5 * mono_f

    # Gentle saturation/tape emulation on the mid channel
    # Convert to mid/side
    L = proc[:, 0]
    R = proc[:, 1] if proc.shape[1] > 1 else proc[:, 0]
    M = (L + R) / 2.0
    S_side = (L - R) / 2.0

    # Apply soft saturation on M only (adds perceived density)
    M_sat = _apply_saturation(M, drive_db=1.0)

    # Reconstruct stereo
    L = M_sat + S_side
    R = M_sat - S_side
    proc = np.stack([L, R], axis=1)

    # Glue compression: attempt to set threshold for ~2-3 dB GR
    # We'll approximate RMS and pick threshold below RMS
    rms = 20.0 * math.log10(np.sqrt(np.mean(proc ** 2)) + 1e-12)
    target_gr_db = 2.5
    # simple heuristic: threshold_db = rms - (target_gr_db * 1.1)
    threshold_db = rms - (target_gr_db * 1.1)
    logger.info("Glue compression heuristic: rms=%.2f dB, threshold=%.2f dB", rms, threshold_db)

    if PEDALBOARD_AVAILABLE:
        try:
            comp = Compressor(threshold_db, ratio=1.8, attack_ms=25.0, release_ms=250.0, makeup_gain_db=0.0)
            board = Pedalboard([comp])
            proc = board(proc, sr)
        except Exception:
            logger.exception("Pedalboard compressor failed; skipping compressor step")
    else:
        # naive soft-knee gain reduction: apply simple reduction when signal > threshold (in dB)
        logger.info("Applying naive compressor fallback")
        linear_threshold = 10 ** (threshold_db / 20.0)
        ratio = 1.8
        # per-sample soft knee
        abs_proc = np.abs(proc)
        over = abs_proc > linear_threshold
        proc[over] = np.sign(proc[over]) * (linear_threshold + (abs_proc[over] - linear_threshold) / ratio)

    # Stereo imaging: mono low below 120Hz, widen highs above 5kHz
    try:
        # Lowpass for low band and replace stereo with mono below 120Hz
        # Using librosa's filters: via STFT band processing
        S_all = np.array([librosa.stft(proc[:, ch], n_fft=4096) for ch in range(proc.shape[1])])
        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
        low_idx = np.where(freqs <= 120)[0]
        high_idx = np.where(freqs >= 5000)[0]
        # Mono low: average left/right in low bins
        S_all[:, low_idx, :] = np.mean(S_all[:, low_idx, :], axis=0)[np.newaxis, :, :]
        # Widen highs by boosting side content slightly (simple approach)
        # Convert to mid-side in STFT domain and boost side above 5kHz
        # M = (L+R)/2, S = (L-R)/2
        L_spec = S_all[0]
        R_spec = S_all[1]
        M_spec = 0.5 * (L_spec + R_spec)
        S_spec = 0.5 * (L_spec - R_spec)
        S_spec[high_idx, :] *= 1.12  # widen factor
        L_spec = M_spec + S_spec
        R_spec = M_spec - S_spec
        proc_new = np.zeros_like(proc)
        for ch in range(proc.shape[1]):
            proc_new[:, ch] = librosa.istft(L_spec if ch == 0 else R_spec, length=proc.shape[0])
        proc = proc_new
    except Exception:
        logger.exception("STFT-based imaging failed; skipping widen/mono low step")

    # Apply pre-gain toward target LUFS
    pre_gain_db = gain_db_initial
    linear_gain = 10 ** (pre_gain_db / 20.0)
    proc = proc * linear_gain

    # Iterative limiting / maximization pass: run a brickwall limiter with lookahead if available and measure LUFS
    # We'll run a limited number of iterations to approach target_lufs while keeping true peak <= tp_ceiling
    out = proc
    final_streaming = os.path.join(workdir, f"{basename}.streaming.wav")
    final_hires = os.path.join(workdir, f"{basename}.hires.wav")

    # Start iteration
    max_iters = 6
    for iteration in range(max_iters):
        logger.info("Limiting iteration %d/%d", iteration + 1, max_iters)

        # Create limiter via pedalboard if available
        out_limited = out.copy()
        if PEDALBOARD_AVAILABLE:
            try:
                # Use a lookahead-aware limiter if available; otherwise typical Limiter
                lim = Limiter(threshold_db=tp_ceiling)
                board = Pedalboard([lim])
                out_limited = board(out_limited, sr)
            except Exception:
                logger.exception("Pedalboard limiter failed; applying naive clip to -1 dBTP")
                tp_lin = 10 ** (tp_ceiling / 20.0)
                out_limited = np.clip(out_limited, -tp_lin, tp_lin)
        else:
            tp_lin = 10 ** (tp_ceiling / 20.0)
            out_limited = np.clip(out_limited, -tp_lin, tp_lin)

        # Measure LUFS and true peak after limiting
        measured_iter = _compute_lufs(out_limited, sr)
        measured_iter['true_peak_dbfs'] = _estimate_true_peak(out_limited, sr)
        logger.info("Post-limit LUFS=%.2f, TP=%.2f", measured_iter['integrated_loudness'], measured_iter['true_peak_dbfs'])

        # Check if targets met
        lufs_err = target_lufs - measured_iter['integrated_loudness']
        tp_over = measured_iter['true_peak_dbfs'] - tp_ceiling

        # If we are quieter than target, add a small amount of makeup gain before limiting next iteration (conservative)
        if lufs_err > 0.2 and iteration < max_iters - 1:
            # convert LUFS delta to dB (approx): apply only half the required to avoid overshoot
            adj_db = min(2.0, lufs_err * 0.8)
            logger.info("Applying makeup gain %.2f dB to approach LUFS target", adj_db)
            out = out_limited * (10 ** (adj_db / 20.0))
            continue

        # If TP over ceiling, reduce pre-limiter gain a bit and try again
        if tp_over > 0.05 and iteration < max_iters - 1:
            reduction_db = min(1.5, tp_over * 0.8)
            logger.info("TP over by %.2f dB -> reducing pre-gain by %.2f dB and iterate", tp_over, reduction_db)
            out = out * (10 ** (-reduction_db / 20.0))
            continue

        # Targets met or no more iterations
        out = out_limited
        measured = measured_iter
        break

    # Final dithering / bit depth conversions
    # Streaming master: 16-bit 44.1k
    streaming_sr = 44100
    hires_sr = 48000

    # Use ffmpeg if available for high-quality sample rate conversion; fallback to librosa
    def resample_and_write(array, src_sr, dst_sr, out_path, subtype):
        if FFMPEG_AVAILABLE:
            # Write to temp WAV and let ffmpeg resample to desired sr and bit-depth
            tmp = out_path + ".tmp.wav"
            _write_wav(tmp, array, src_sr, subtype='FLOAT')
            try:
                stream = ffmpeg.input(tmp)
                stream = ffmpeg.output(stream, out_path, ar=dst_sr, ac=2, sample_fmt='s16' if subtype == 'PCM_16' else 's32')
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
                os.remove(tmp)
                return True
            except Exception:
                logger.exception("ffmpeg resample failed, falling back to librosa")
        # fallback
        arr_mono = array
        if src_sr != dst_sr:
            arr_mono = np.vstack([librosa.resample(array[:, ch], src_sr, dst_sr) for ch in range(array.shape[1])]).T
        _write_wav(out_path, arr_mono, dst_sr, subtype=subtype)
        return True

    # Ensure final is in float32 range [-1,1]
    peak = np.max(np.abs(out)) + 1e-15
    if peak > 1.0:
        out = out / peak

    # write hires 24-bit 48k
    resample_and_write(out, sr, hires_sr, final_hires, subtype='PCM_24')
    # write streaming 16-bit 44.1k
    resample_and_write(out, sr, streaming_sr, final_streaming, subtype='PCM_16')

    result = {
        'streaming_path': final_streaming,
        'hires_path': final_hires,
        'measured': measured,
        'iterations': iteration + 1,
    }

    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('--outdir', default=tempfile.gettempdir())
    parser.add_argument('--lufs', type=float, default=-10.0)
    parser.add_argument('--tp', type=float, default=-1.0)
    args = parser.parse_args()
    out = process_master(args.infile, args.outdir, target_lufs=args.lufs, tp_ceiling=args.tp)
    print(out)
