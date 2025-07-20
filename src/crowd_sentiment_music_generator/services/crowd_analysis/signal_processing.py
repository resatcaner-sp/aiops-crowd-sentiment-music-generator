"""Signal processing utilities for audio analysis.

This module provides utility functions for audio signal processing,
including filtering, normalization, and segmentation.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import librosa
from scipy import signal

from crowd_sentiment_music_generator.exceptions.audio_processing_error import AudioProcessingError

logger = logging.getLogger(__name__)


def normalize_audio(audio_segment: np.ndarray) -> np.ndarray:
    """Normalize audio segment to have values between -1 and 1.
    
    Args:
        audio_segment: Audio segment as numpy array
        
    Returns:
        Normalized audio segment
        
    Raises:
        AudioProcessingError: If normalization fails
    """
    try:
        if np.max(np.abs(audio_segment)) > 0:
            return audio_segment / np.max(np.abs(audio_segment))
        return audio_segment
    except Exception as e:
        logger.error(f"Failed to normalize audio: {str(e)}")
        raise AudioProcessingError(f"Audio normalization failed: {str(e)}")


def apply_bandpass_filter(
    audio_segment: np.ndarray, 
    sr: int = 22050, 
    low_cutoff: float = 300.0, 
    high_cutoff: float = 3000.0,
    order: int = 5
) -> np.ndarray:
    """Apply bandpass filter to audio segment.
    
    This function applies a bandpass filter to isolate frequencies within a specific range,
    which can be useful for isolating crowd noise from other sounds.
    
    Args:
        audio_segment: Audio segment as numpy array
        sr: Sample rate (default: 22050 Hz)
        low_cutoff: Low cutoff frequency (default: 300 Hz)
        high_cutoff: High cutoff frequency (default: 3000 Hz)
        order: Filter order (default: 5)
        
    Returns:
        Filtered audio segment
        
    Raises:
        AudioProcessingError: If filtering fails
    """
    try:
        # Calculate Nyquist frequency
        nyquist = 0.5 * sr
        
        # Normalize cutoff frequencies
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        
        # Design bandpass filter
        b, a = signal.butter(order, [low, high], btype='band')
        
        # Apply filter
        filtered_audio = signal.filtfilt(b, a, audio_segment)
        
        return filtered_audio
    except Exception as e:
        logger.error(f"Failed to apply bandpass filter: {str(e)}")
        raise AudioProcessingError(f"Bandpass filtering failed: {str(e)}")


def segment_audio(
    audio_segment: np.ndarray, 
    sr: int = 22050, 
    segment_duration: float = 2.0,
    hop_duration: Optional[float] = None
) -> List[np.ndarray]:
    """Segment audio into fixed-length chunks with optional overlap.
    
    Args:
        audio_segment: Audio segment as numpy array
        sr: Sample rate (default: 22050 Hz)
        segment_duration: Duration of each segment in seconds (default: 2.0)
        hop_duration: Hop duration in seconds (default: segment_duration/2)
        
    Returns:
        List of audio segments
        
    Raises:
        AudioProcessingError: If segmentation fails
    """
    try:
        # Calculate segment and hop lengths in samples
        segment_length = int(segment_duration * sr)
        
        if hop_duration is None:
            hop_length = segment_length // 2
        else:
            hop_length = int(hop_duration * sr)
        
        # Check if audio is long enough for at least one segment
        if len(audio_segment) < segment_length:
            return [audio_segment]
        
        # Segment audio
        segments = []
        for start in range(0, len(audio_segment) - segment_length + 1, hop_length):
            segments.append(audio_segment[start:start + segment_length])
        
        return segments
    except Exception as e:
        logger.error(f"Failed to segment audio: {str(e)}")
        raise AudioProcessingError(f"Audio segmentation failed: {str(e)}")


def detect_silence(
    audio_segment: np.ndarray, 
    threshold: float = 0.01, 
    min_duration: float = 0.1,
    sr: int = 22050
) -> List[Tuple[float, float]]:
    """Detect silent regions in audio segment.
    
    Args:
        audio_segment: Audio segment as numpy array
        threshold: Amplitude threshold for silence detection (default: 0.01)
        min_duration: Minimum silence duration in seconds (default: 0.1)
        sr: Sample rate (default: 22050 Hz)
        
    Returns:
        List of (start_time, end_time) tuples for silent regions
        
    Raises:
        AudioProcessingError: If silence detection fails
    """
    try:
        # Calculate RMS energy
        rms = librosa.feature.rms(y=audio_segment)[0]
        
        # Find silent regions
        silent_frames = np.where(rms < threshold)[0]
        
        # Convert frames to time
        frame_length = len(audio_segment) / len(rms)
        min_frames = int(min_duration * sr / frame_length)
        
        # Group consecutive silent frames
        silent_regions = []
        if len(silent_frames) > 0:
            # Initialize with the first silent frame
            region_start = silent_frames[0]
            region_end = silent_frames[0]
            
            for i in range(1, len(silent_frames)):
                if silent_frames[i] == silent_frames[i-1] + 1:
                    # Continue the current region
                    region_end = silent_frames[i]
                else:
                    # End the current region and start a new one
                    if region_end - region_start + 1 >= min_frames:
                        start_time = region_start * frame_length / sr
                        end_time = (region_end + 1) * frame_length / sr
                        silent_regions.append((start_time, end_time))
                    
                    region_start = silent_frames[i]
                    region_end = silent_frames[i]
            
            # Add the last region if it's long enough
            if region_end - region_start + 1 >= min_frames:
                start_time = region_start * frame_length / sr
                end_time = (region_end + 1) * frame_length / sr
                silent_regions.append((start_time, end_time))
        
        return silent_regions
    except Exception as e:
        logger.error(f"Failed to detect silence: {str(e)}")
        raise AudioProcessingError(f"Silence detection failed: {str(e)}")


def remove_dc_offset(audio_segment: np.ndarray) -> np.ndarray:
    """Remove DC offset from audio segment.
    
    Args:
        audio_segment: Audio segment as numpy array
        
    Returns:
        Audio segment with DC offset removed
        
    Raises:
        AudioProcessingError: If DC offset removal fails
    """
    try:
        return audio_segment - np.mean(audio_segment)
    except Exception as e:
        logger.error(f"Failed to remove DC offset: {str(e)}")
        raise AudioProcessingError(f"DC offset removal failed: {str(e)}")


def preprocess_audio(
    audio_segment: np.ndarray,
    sr: int = 22050,
    remove_offset: bool = True,
    normalize: bool = True
) -> np.ndarray:
    """Preprocess audio segment for feature extraction.
    
    This function applies common preprocessing steps to prepare audio for feature extraction.
    
    Args:
        audio_segment: Audio segment as numpy array
        sr: Sample rate (default: 22050 Hz)
        remove_offset: Whether to remove DC offset (default: True)
        normalize: Whether to normalize audio (default: True)
        
    Returns:
        Preprocessed audio segment
        
    Raises:
        AudioProcessingError: If preprocessing fails
    """
    try:
        processed = audio_segment.copy()
        
        # Remove DC offset if requested
        if remove_offset:
            processed = remove_dc_offset(processed)
        
        # Normalize if requested
        if normalize:
            processed = normalize_audio(processed)
        
        return processed
    except Exception as e:
        logger.error(f"Failed to preprocess audio: {str(e)}")
        raise AudioProcessingError(f"Audio preprocessing failed: {str(e)}")