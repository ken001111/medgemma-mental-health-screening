"""
Audio processing module for extracting audio from phone calls
and extracting prosody features for multimodal analysis.
"""
import os
import subprocess
import numpy as np
import soundfile as sf
import torch
from typing import Tuple, Dict
import librosa
from config import SR, WAV_CACHE_DIR


def extract_audio_from_call(call_path: str, output_wav_path: str = None, sr: int = SR) -> str:
    """
    Extract audio from phone call recording (supports various formats).
    
    Args:
        call_path: Path to the phone call recording
        output_wav_path: Optional output path. If None, generates from call_path
        sr: Sample rate for output audio
    
    Returns:
        Path to the extracted WAV file
    """
    if output_wav_path is None:
        os.makedirs(WAV_CACHE_DIR, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(call_path))[0]
        output_wav_path = os.path.join(WAV_CACHE_DIR, f"{base_name}.wav")
    
    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
    
    cmd = [
        "ffmpeg", "-y", "-i", call_path,
        "-vn", "-ac", "1", "-ar", str(sr),
        output_wav_path
    ]
    
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Failed to extract audio from {call_path}: {e}")
    
    return output_wav_path


def extract_prosody_features(wav_path: str) -> Dict[str, float]:
    """
    Extract prosody features from audio file.
    Prosody includes: pitch, speaking rate, pauses, energy, etc.
    
    Args:
        wav_path: Path to WAV audio file
    
    Returns:
        Dictionary of prosody features
    """
    try:
        audio, sr = sf.read(wav_path)
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Duration
        duration = len(audio) / sr
        
        # Fundamental frequency (pitch) using librosa
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        mean_pitch = np.mean(pitch_values) if pitch_values else 0.0
        pitch_std = np.std(pitch_values) if pitch_values else 0.0
        
        # Speaking rate (words per minute approximation)
        # Using zero-crossing rate as proxy for speech activity
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        speaking_rate = np.mean(zcr) * 60  # Approximate speaking rate
        
        # Energy features
        rms = librosa.feature.rms(y=audio)[0]
        mean_energy = np.mean(rms)
        energy_std = np.std(rms)
        
        # Pause detection (silence ratio)
        energy_threshold = np.percentile(rms, 20)
        silence_ratio = np.sum(rms < energy_threshold) / len(rms)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        mean_spectral_centroid = np.mean(spectral_centroids)
        
        # Jitter and shimmer (voice quality indicators)
        # Simplified: using pitch variation as jitter proxy
        jitter = pitch_std / (mean_pitch + 1e-6) if mean_pitch > 0 else 0.0
        
        prosody_features = {
            "duration": float(duration),
            "mean_pitch": float(mean_pitch),
            "pitch_std": float(pitch_std),
            "speaking_rate": float(speaking_rate),
            "mean_energy": float(mean_energy),
            "energy_std": float(energy_std),
            "silence_ratio": float(silence_ratio),
            "mean_spectral_centroid": float(mean_spectral_centroid),
            "jitter": float(jitter),
        }
        
        return prosody_features
    
    except Exception as e:
        print(f"Error extracting prosody features: {e}")
        # Return default values on error
        return {
            "duration": 0.0,
            "mean_pitch": 0.0,
            "pitch_std": 0.0,
            "speaking_rate": 0.0,
            "mean_energy": 0.0,
            "energy_std": 0.0,
            "silence_ratio": 0.0,
            "mean_spectral_centroid": 0.0,
            "jitter": 0.0,
        }


def load_audio(wav_path: str) -> Tuple[np.ndarray, int]:
    """
    Load audio file for processing.
    
    Args:
        wav_path: Path to WAV file
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    audio, sr = sf.read(wav_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    return audio, sr
