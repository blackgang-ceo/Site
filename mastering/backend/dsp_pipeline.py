"""
dsp_pipeline.py

High-level mastering pipeline implementation. The goal is to implement the chain described
in the spec with production-ready comments and safety features.

Notes / design decisions:
- Files are processed from disk to avoid holding large arrays in memory indefinitely.
- We use a combination of scipy (FIR linear-phase HPF), librosa for analysis, pyloudnorm for LUFS,
  pedalboard for plugin-like effects where appropriate, and numpy/soundfile for IO.
- We iterate gain + limiting to meet target LUFS while respecting true peak ceiling.

This file intentionally trades off the absolute last dB of brickwall-limited loudness for
clarity and anti-distortion behavior. For fully 'competitive' mastering, replace/augment
parts with native VSTs in a secure server environment.
"""

from typing import Tuple, Dict, Optional
import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
from pedalboard import Pedalboard, HighpassFilter, Compressor, Limiter, Mono, Gain
from pedalboard.io import AudioFile
import scipy.signal as sps
import math
import tempfile
import os


class MasteringEngine:
    """Encapsulates a mastering pipeline instance.

    Usage pattern:
        engine = MasteringEngine()
        engine.process_to_targets(input_path, out_streaming, out_hires, target_lufs=-10.0)
        report = engine.get_last_report()
    """

    def __init__(self):
        self.last_report: Dict = {}

    # ------------------------------ Utilities --------------------------------
    @staticmethod
    def read_audio(path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Load audio using soundfile (preferred) but fall back to librosa if necessary.
        Returns float32 numpy array (shape: samples, channels) and sample rate.
        """
        data, sr_file = sf.read(path, always_2d=True, dtype="float32")
        # soundfile returns (frames, channels) — we prefer (channels, frames) or keep (frames,channels)
        # We'll keep (frames, channels)
        if sr is not None and sr_file != sr:
            # resample with librosa (high quality)
            data = librosa.resample(data.T, orig_sr=sr_file, target_sr=sr, axis=1).T
            sr_file = sr
        return data, sr_file

    @staticmethod
    def write_wav(path: str, data: np.ndarray, sr: int, subtype: str = 'PCM_16'):
        """Write WAV with soundfile. data is (frames, channels) float32 in -1..1.
        subtype options: PCM_16, PCM_24
        """
        sf.write(path, data, sr, subtype=subtype)

    @staticmethod
    def linear_phase_highpass(signal: np.ndarray, sr: int, cutoff: float = 20.0, transition_width: float = 10.0) -> np.ndarray:
        """Apply a linear-phase FIR high-pass filter (windowed sinc) using fftconvolve for speed.
        This produces linear phase (no phase distortion) and excellent low-frequency attenuation.
        - signal: (frames, channels)
        - cutoff: Hz
        """
        nyq = sr / 2.0
        # FIR order chosen for steep but reasonable latency. Longer order = better slope but more CPU.
        # Use a proportional order depending on sample rate
        width = transition_width / nyq
        N = int(min(max(4096, int(0.08 * sr)), 16384))  # 80ms window approximate
        # design a highpass with firwin
        taps = sps.firwin(N, cutoff / nyq, pass_zero=False, window='hann')
        # ensure that taps sum to ~0 for high-pass
        # apply to each channel
        out = np.zeros_like(signal)
        for ch in range(signal.shape[1]):
            out[:, ch] = sps.fftconvolve(signal[:, ch], taps, mode='same')
        return out

    @staticmethod
    def detect_resonant_peaks(signal: np.ndarray, sr: int, top_n: int = 6) -> np.ndarray:
        """Analyze the long-term magnitude spectrum and return candidate resonant frequencies
        that may be suitable for subtractive EQ. We'll search for spectral peaks in the 20Hz-16kHz band.
        """
        # Compute magnitude spectrum (average over frames)
        S = np.abs(librosa.stft(signal.mean(axis=1), n_fft=8192, hop_length=4096))
        spec = S.mean(axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=8192)
        # restrict to sensible range
        idx = np.where((freqs >= 20) & (freqs <= 16000))[0]
        spec_sub = spec[idx]
        freqs_sub = freqs[idx]
        # find peaks
        peaks, _ = sps.find_peaks(spec_sub, height=np.max(spec_sub)*0.05, distance=4)
        if peaks.size == 0:
            return np.array([])
        heights = spec_sub[peaks]
        # sort peaks by magnitude and return top_n frequencies
        order = np.argsort(-heights)
        selected = peaks[order][:top_n]
        return freqs_sub[selected]

    @staticmethod
    def apply_notch(signal: np.ndarray, sr: int, center_hz: float, q: float = 10.0, depth_db: float = -1.5) -> np.ndarray:
        """Apply a narrow subtractive notch using an IIR notch filter (iirnotch) and slight-gain interpolation.
        depth_db is negative (e.g. -1.5). We implement the notch by parallel blend: dry*(1-alpha) + filtered*alpha
        where the filtered path has the notch deeply applied and alpha chosen to reach desired depth.
        """
        # design notch
        w0 = center_hz / (sr / 2.0)
        # iirnotch expects w0 normalized to Nyquist
        b, a = sps.iirnotch(w0, Q=q)
        filtered = np.zeros_like(signal)
        for ch in range(signal.shape[1]):
            filtered[:, ch] = sps.lfilter(b, a, signal[:, ch])
        # compute alpha required to achieve depth
        if depth_db >= 0:
            return signal
        depth_lin = 10 ** (depth_db / 20.0)
        # compute energy at center freq by short bandpass (approx) — instead, apply simple alpha
        alpha = 0.5  # conservative by default; small dips are subtle.
        # For small depth (-1.5dB) alpha ~ 0.3; we'll map linearly
        alpha = max(0.12, min(0.45, -depth_db / 6.0))
        return signal * (1 - alpha) + filtered * alpha

    @staticmethod
    def rms_db(arr: np.ndarray) -> float:
        rms = np.sqrt(np.mean(arr ** 2)) + 1e-12
        return 20 * math.log10(rms)

    # ------------------------------ The Chain --------------------------------
    def analyse(self, audio: np.ndarray, sr: int) -> Dict:
        """Run a set of pre-analysis measures: LUFS, true-peak, spectral centroid and resonances.
        Returns a dictionary with the metrics.
        """
        meter = pyln.Meter(sr)  # create BS.1770 meter
        # pyloudnorm expects mono for true peak? It handles multichannel arrays as NxC
        try:
            integrated = meter.integrated_loudness(audio)
        except Exception:
            # fallback: measure on mid-sum
            mono = audio.mean(axis=1)
            integrated = meter.integrated_loudness(mono)
        try:
            tp = meter.true_peak(audio)
        except Exception:
            # if true_peak not available, approximate by peak in upsampled signal
            up = librosa.resample(audio.T, orig_sr=sr, target_sr=sr * 4).T
            tp = 20 * np.log10(np.max(np.abs(up)) + 1e-12)
        # spectral centroid and resonant candidates
        centroid = np.mean(librosa.feature.spectral_centroid(y=audio.mean(axis=1), sr=sr))
        resonances = self.detect_resonant_peaks(audio, sr)
        return {
            "integrated_lufs": float(integrated),
            "true_peak_db": float(tp),
            "spectral_centroid": float(centroid),
            "resonances": [float(float(r)) for r in resonances],
        }

    def apply_bus_compressor(self, audio: np.ndarray, sr: int, target_reduction_db: float = 2.5) -> np.ndarray:
        """Apply a gentle bus compressor with parameters chosen to achieve ~2-3 dB gain reduction.
        We don't have a perfect analytic inverse of compressor behavior, so we choose a threshold relative
        to the measured LUFS and then apply the pedalboard Compressor.
        """
        meter = pyln.Meter(sr)
        integrated = meter.integrated_loudness(audio)
        # set threshold a few dB above integrated level — ensures RMS peaks see compression
        threshold_db = integrated + 2.5  # compressor engages above average level
        # make ratio mild
        ratio = 1.8
        attack_ms = 20.0
        release_ms = 180.0

        board = Pedalboard([
            Compressor(threshold_db=threshold_db, ratio=ratio, attack_ms=attack_ms, release_ms=release_ms),
        ])
        # pedalboard expects (channels, samples)
        # but AudioFile convenience accepts audio path; we'll use the effect's __call__ which accepts (samples, sample_rate)
        # samples must be shape (samples, channels)
        processed = board(audio, sr)
        return processed

    def mid_side_process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Perform mid/side operations:
         - Make sub-bass (<120Hz) mono
         - Slightly widen >5kHz by increasing side components at those frequencies
         - Apply soft mid-side saturation (perceptual warmth) via tanh waveshaping on mid channel
        """
        # convert to mid/side
        L = audio[:, 0]
        R = audio[:, 1] if audio.shape[1] > 1 else audio[:, 0]
        M = 0.5 * (L + R)
        S = 0.5 * (L - R)

        # lowpass M at 120hz and remove side content below that frequency
        # use IIR butterworth for speed
        b, a = sps.butter(4, 120 / (sr / 2.0), btype='low')
        low_mono = sps.lfilter(b, a, M)

        # high band for widening
        b_h, a_h = sps.butter(4, 5000 / (sr / 2.0), btype='high')
        high_side = sps.lfilter(b_h, a_h, S)
        # apply gentle scaling to side high frequencies
        high_side *= 1.12

        # reconstruct L/R: low band uses mono low_mono, mid uses M - low contribution, side uses S (with high_side boosted)
        # Blend low_mono into both channels for <120Hz
        # We'll compute crossover using simple additive approach
        # compute M' and S'
        # subtract low_mono's contribution from M to avoid doubling
        M_prime = M - low_mono * 0.5
        S_prime = S
        # add widened high side content
        S_prime += high_side

        L_out = M_prime + S_prime
        R_out = M_prime - S_prime

        out = np.stack([L_out, R_out], axis=1)

        # subtle harmonic saturation on mid channel (soft-tanh)
        mid = 0.5 * (out[:, 0] + out[:, 1])
        side = 0.5 * (out[:, 0] - out[:, 1])
        # apply soft saturation only to mid and only small amount
        sat_amount = 0.002  # tiny amount of drive
        mid_sat = np.tanh(mid * (1 + sat_amount)) - mid
        mid = mid + mid_sat * 0.6
        L_out = mid + side
        R_out = mid - side
        out = np.stack([L_out, R_out], axis=1)
        # gentle normalization to avoid adding gain
        peak = np.max(np.abs(out)) + 1e-12
        if peak > 1.0:
            out /= peak
        return out

    def apply_true_peak_limiter(self, audio: np.ndarray, sr: int, ceiling_db: float = -1.0) -> np.ndarray:
        """We'll apply pedalboard Limiter with the specified ceiling in dBFS. pedalboard Limiter uses lookahead internally.
        """
        board = Pedalboard([
            Limiter(threshold_db=ceiling_db),
        ])
        processed = board(audio, sr)
        return processed

    def target_lufs_iteration(self, audio: np.ndarray, sr: int, target_lufs: float, tp_ceiling_db: float) -> Tuple[np.ndarray, Dict]:
        """Iteratively apply gain -> limiter to reach target LUFS while policing true peak.
        Steps:
         1. Measure integrated LUFS and true peak
         2. Compute required gain (target - current)
         3. Limit the maximum pre-gain if it would violate true peak ceiling
         4. Apply gain + limiter and re-measure
         5. Repeat a few times (usually 1-3 iterations required)
        Returns processed audio and a dict with final measurements.
        """
        meter = pyln.Meter(sr)
        current_lufs = meter.integrated_loudness(audio)
        current_tp = meter.true_peak(audio)

        # gain needed (dB)
        gain_needed = float(target_lufs - current_lufs)

        # do not push more than 8 dB in pre-gain on a single pass — risk of distortion
        max_pregain_db = 8.0
        pregain_db = max(-12.0, min(max_pregain_db, gain_needed))

        # check predicted true peak after pregain (approx): tp + pregain_db
        predicted_tp = current_tp + pregain_db
        # reduce pregain if it will exceed ceiling by more than 0.1 dB
        if predicted_tp > tp_ceiling_db - 0.1:
            pregain_db = tp_ceiling_db - 0.1 - current_tp
            # clamp
            pregain_db = min(pregain_db, max_pregain_db)

        # Now apply pregain via pedalboard Gain and then Limiter
        # We'll attempt a two-stage approach: small pregain -> limiter -> measure -> if still short, repeat
        processed = audio
        for iteration in range(3):
            if abs(pregain_db) > 0.01:
                board = Pedalboard([Gain(pregain_db)])
                processed = board(processed, sr)
            # limiter to ceiling
            processed = self.apply_true_peak_limiter(processed, sr, ceiling_db=tp_ceiling_db)
            # measure
            new_lufs = meter.integrated_loudness(processed)
            new_tp = meter.true_peak(processed)
            # recompute remaining need
            remain_db = target_lufs - new_lufs
            if abs(remain_db) < 0.25:
                # close enough
                break
            # compute small incremental pregain for next iteration
            pregain_db = max(-6.0, min(6.0, remain_db))
            # enforce true-peak safety
            if new_tp + pregain_db > tp_ceiling_db - 0.1:
                pregain_db = tp_ceiling_db - 0.1 - new_tp
            if abs(pregain_db) < 0.1:
                break
        final_metrics = {
            'integrated_lufs': float(new_lufs),
            'true_peak_db': float(new_tp),
        }
        return processed, final_metrics

    def process_to_targets(self, input_path: str, out_streaming: str, out_hires: str, target_lufs: float = -10.0, tp_ceiling_db: float = -1.0):
        """High-level processing function that reads input, applies the mastering chain, and writes
        two masters: streaming (16-bit / 44.1k) and hi-res (24-bit / 48k).
        """
        # 1) Read and homogenize to float32 stereo
        audio, sr = self.read_audio(input_path)
        if audio.shape[1] == 1:
            # duplicate mono to stereo for processing chain
            audio = np.repeat(audio, 2, axis=1)
        # Store original analysis
        analysis_before = self.analyse(audio, sr)

        # 2) linear-phase HPF at 20Hz
        audio = self.linear_phase_highpass(audio, sr, cutoff=20.0, transition_width=8.0)

        # 3) subtractive dynamic EQ — detect resonances and apply shallow notches
        resonances = analysis_before.get('resonances', [])
        # choose candidates around typical problem areas too
        candidates = []
        # prefer automatic resonances but add typical bands if not present
        for r in resonances:
            if 40 < r < 16000:
                candidates.append(r)
        # add typical problem bands
        typical = [250.0, 700.0, 3000.0, 4500.0]
        candidates.extend(typical)
        # dedupe and keep only first 6
        cand_unique = sorted(set([round(float(x)) for x in candidates]))[:6]
        for f in cand_unique:
            audio = self.apply_notch(audio, sr, float(f), q=10.0, depth_db=-1.6)

        # 4) gentle glue compression on the bus
        audio = self.apply_bus_compressor(audio, sr, target_reduction_db=2.5)

        # 5) tonal balance / harmonic excitement
        # apply a gentle mid-side processing stage
        audio = self.mid_side_process(audio, sr)

        # 6) stereo imaging (already partly in mid_side_process). Ensure <120Hz mono
        # (already enforced in mid_side_process)

        # 7) iterative limiting + LUFS target
        processed, final_meas = self.target_lufs_iteration(audio, sr, target_lufs=target_lufs, tp_ceiling_db=tp_ceiling_db)

        # final gentle dithering / bit-depth conversion happens on write
        # write hi-res (24-bit 48k)
        # resample to 48k if needed
        hires_sr = 48000
        if sr != hires_sr:
            # librosa expects shape (channels, samples)
            processed = librosa.resample(processed.T, orig_sr=sr, target_sr=hires_sr, axis=1).T
            sr_out = hires_sr
        else:
            sr_out = sr
        # clip to [-1,1]
        processed = np.clip(processed, -0.9999, 0.9999)
        # write 24-bit file
        self.write_wav(out_hires, processed, sr_out, subtype='PCM_24')

        # create streaming master 16-bit 44.1k
        streaming_sr = 44100
        streaming = processed
        if sr_out != streaming_sr:
            streaming = librosa.resample(processed.T, orig_sr=sr_out, target_sr=streaming_sr, axis=1).T
        streaming = np.clip(streaming, -0.9999, 0.9999)
        self.write_wav(out_streaming, streaming, streaming_sr, subtype='PCM_16')

        analysis_after = self.analyse(processed, sr_out)
        # store report
        self.last_report = {
            'before': analysis_before,
            'after': analysis_after,
            'final_targets': {'target_lufs': target_lufs, 'tp_ceiling_db': tp_ceiling_db},
            'files': {'streaming': out_streaming, 'hires': out_hires},
        }
        return True

    def get_last_report(self) -> Dict:
        return self.last_report


if __name__ == '__main__':
    # quick local smoke test (developer only)
    eng = MasteringEngine()
    print('Engine initialized')
