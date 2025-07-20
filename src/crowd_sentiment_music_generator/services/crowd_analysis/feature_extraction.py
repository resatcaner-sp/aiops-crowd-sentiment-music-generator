"""Audio feature extraction for crowd analysis.

This module provides functions for extracting audio features from crowd noise,
including RMS energy, spectral features, zero crossing rate, and tempo estimation.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import librosa
from librosa import feature as librosa_feature

from crowd_sentiment_music_generator.exceptions.audio_processing_error import AudioProcessingError
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.utils.error_handlers import with_error_handling

logger = logging.getLogger(__name__)


def extract_rms_energy(audio_segment: np.ndarray) -> float:
    """Extract RMS energy from audio segment.
    
    Args:
        audio_segment: Audio segment as numpy array
        
    Returns:
        RMS energy value
        
    Raises:
        AudioProcessingError: If RMS energy extraction fails
    """
    try:
        # Calculate RMS energy using librosa
        rms = librosa.feature.rms(y=audio_segment)[0]
        # Return mean RMS energy
        return float(np.mean(rms))
    except Exception as e:
        logger.error(f"Failed to extract RMS energy: {str(e)}")
        raise AudioProcessingError(f"RMS energy extraction failed: {str(e)}")


def extract_spectral_centroid(audio_segment: np.ndarray, sr: int = 22050) -> float:
    """Extract spectral centroid from audio segment.
    
    The spectral centroid represents the "center of mass" of the spectrum,
    indicating the brightness of the sound.
    
    Args:
        audio_segment: Audio segment as numpy array
        sr: Sample rate (default: 22050 Hz)
        
    Returns:
        Spectral centroid value
        
    Raises:
        AudioProcessingError: If spectral centroid extraction fails
    """
    try:
        # Calculate spectral centroid using librosa
        centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)[0]
        # Return mean spectral centroid
        return float(np.mean(centroid))
    except Exception as e:
        logger.error(f"Failed to extract spectral centroid: {str(e)}")
        raise AudioProcessingError(f"Spectral centroid extraction failed: {str(e)}")


def extract_spectral_rolloff(audio_segment: np.ndarray, sr: int = 22050, roll_percent: float = 0.85) -> float:
    """Extract spectral rolloff from audio segment.
    
    The spectral rolloff is the frequency below which roll_percent of the spectrum's energy is contained.
    
    Args:
        audio_segment: Audio segment as numpy array
        sr: Sample rate (default: 22050 Hz)
        roll_percent: Rolloff percentage (default: 0.85)
        
    Returns:
        Spectral rolloff value
        
    Raises:
        AudioProcessingError: If spectral rolloff extraction fails
    """
    try:
        # Calculate spectral rolloff using librosa
        rolloff = librosa.feature.spectral_rolloff(
            y=audio_segment, sr=sr, roll_percent=roll_percent
        )[0]
        # Return mean spectral rolloff
        return float(np.mean(rolloff))
    except Exception as e:
        logger.error(f"Failed to extract spectral rolloff: {str(e)}")
        raise AudioProcessingError(f"Spectral rolloff extraction failed: {str(e)}")


def extract_zero_crossing_rate(audio_segment: np.ndarray) -> float:
    """Extract zero crossing rate from audio segment.
    
    The zero crossing rate is the rate at which the signal changes from positive to negative or back.
    It's a measure of the noisiness or percussiveness of a sound.
    
    Args:
        audio_segment: Audio segment as numpy array
        
    Returns:
        Zero crossing rate value
        
    Raises:
        AudioProcessingError: If zero crossing rate extraction fails
    """
    try:
        # Calculate zero crossing rate using librosa
        zcr = librosa.feature.zero_crossing_rate(audio_segment)[0]
        # Return mean zero crossing rate
        return float(np.mean(zcr))
    except Exception as e:
        logger.error(f"Failed to extract zero crossing rate: {str(e)}")
        raise AudioProcessingError(f"Zero crossing rate extraction failed: {str(e)}")


def estimate_tempo(audio_segment: np.ndarray, sr: int = 22050) -> float:
    """Estimate tempo from audio segment.
    
    Args:
        audio_segment: Audio segment as numpy array
        sr: Sample rate (default: 22050 Hz)
        
    Returns:
        Estimated tempo in BPM
        
    Raises:
        AudioProcessingError: If tempo estimation fails
    """
    try:
        # Estimate tempo using librosa
        onset_env = librosa.onset.onset_strength(y=audio_segment, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        return float(tempo)
    except Exception as e:
        logger.error(f"Failed to estimate tempo: {str(e)}")
        raise AudioProcessingError(f"Tempo estimation failed: {str(e)}")


def extract_spectral_contrast(audio_segment: np.ndarray, sr: int = 22050, n_bands: int = 6) -> float:
    """Extract spectral contrast from audio segment.
    
    Spectral contrast represents the difference between peaks and valleys in the spectrum,
    which can be used to distinguish between different types of sounds.
    
    Args:
        audio_segment: Audio segment as numpy array
        sr: Sample rate (default: 22050 Hz)
        n_bands: Number of frequency bands (default: 6)
        
    Returns:
        Mean spectral contrast value
        
    Raises:
        AudioProcessingError: If spectral contrast extraction fails
    """
    try:
        # Calculate spectral contrast using librosa
        contrast = librosa.feature.spectral_contrast(y=audio_segment, sr=sr, n_bands=n_bands)
        # Return mean spectral contrast
        return float(np.mean(contrast))
    except Exception as e:
        logger.error(f"Failed to extract spectral contrast: {str(e)}")
        raise AudioProcessingError(f"Spectral contrast extraction failed: {str(e)}")


def extract_mfcc(
    audio_segment: np.ndarray, sr: int = 22050, n_mfcc: int = 13
) -> np.ndarray:
    """Extract Mel-frequency cepstral coefficients (MFCCs) from audio segment.
    
    MFCCs represent the short-term power spectrum of a sound and are commonly used
    in audio classification tasks.
    
    Args:
        audio_segment: Audio segment as numpy array
        sr: Sample rate (default: 22050 Hz)
        n_mfcc: Number of MFCCs to extract (default: 13)
        
    Returns:
        Array of MFCC values
        
    Raises:
        AudioProcessingError: If MFCC extraction fails
    """
    try:
        # Calculate MFCCs using librosa
        mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=n_mfcc)
        # Return mean MFCCs
        return np.mean(mfccs, axis=1)
    except Exception as e:
        logger.error(f"Failed to extract MFCCs: {str(e)}")
        raise AudioProcessingError(f"MFCC extraction failed: {str(e)}")


@with_error_handling
def extract_all_features(
    audio_segment: np.ndarray, 
    sr: Optional[int] = None,
    config: Optional[SystemConfig] = None
) -> Dict[str, float]:
    """Extract all audio features from audio segment.
    
    Args:
        audio_segment: Audio segment as numpy array
        sr: Sample rate (optional, uses config value if not provided)
        config: System configuration (optional, uses default values if not provided)
        
    Returns:
        Dictionary of audio features
        
    Raises:
        AudioProcessingError: If feature extraction fails
    """
    if config is None:
        config = SystemConfig()
    
    if sr is None:
        sr = config.audio_sample_rate
    
    # Check if audio segment is valid
    if audio_segment.size == 0:
        raise AudioProcessingError("Empty audio segment")
    
    # Extract features
    features = {
        "rms_energy": extract_rms_energy(audio_segment),
        "spectral_centroid": extract_spectral_centroid(audio_segment, sr),
        "spectral_rolloff": extract_spectral_rolloff(audio_segment, sr),
        "zero_crossing_rate": extract_zero_crossing_rate(audio_segment),
        "tempo": estimate_tempo(audio_segment, sr),
        "spectral_contrast": extract_spectral_contrast(audio_segment, sr)
    }
    
    # Extract MFCCs and add them to features
    mfccs = extract_mfcc(audio_segment, sr)
    for i, mfcc in enumerate(mfccs):
        features[f"mfcc_{i+1}"] = float(mfcc)
    
    return features


def isolate_crowd_noise(
    audio_segment: np.ndarray, 
    sr: int = 22050
) -> Tuple[np.ndarray, float]:
    """Isolate crowd noise from audio segment.
    
    This function attempts to separate crowd noise from commentary and other sounds
    using basic signal processing techniques.
    
    Args:
        audio_segment: Audio segment as numpy array
        sr: Sample rate (default: 22050 Hz)
        
    Returns:
        Tuple of (isolated crowd noise, isolation quality score)
        
    Raises:
        AudioProcessingError: If crowd noise isolation fails
    """
    try:
        # Simple high-pass filter to remove low-frequency commentary
        # This is a basic approach and could be improved with more sophisticated techniques
        crowd_noise = librosa.effects.high_pass_filter(audio_segment, cutoff=300)
        
        # Calculate isolation quality score (simple heuristic based on spectral characteristics)
        # In a real implementation, this would be more sophisticated
        original_centroid = extract_spectral_centroid(audio_segment, sr)
        filtered_centroid = extract_spectral_centroid(crowd_noise, sr)
        
        # Higher score means better isolation (simplified metric)
        quality_score = min(1.0, filtered_centroid / original_centroid if original_centroid > 0 else 0.5)
        
        return crowd_noise, quality_score
    except Exception as e:
        logger.error(f"Failed to isolate crowd noise: {str(e)}")
        raise AudioProcessingError(f"Crowd noise isolation failed: {str(e)}")