"""Unit tests for signal processing module."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from crowd_sentiment_music_generator.exceptions.audio_processing_error import AudioProcessingError
from crowd_sentiment_music_generator.services.crowd_analysis.signal_processing import (
    normalize_audio,
    apply_bandpass_filter,
    segment_audio,
    detect_silence,
    remove_dc_offset,
    preprocess_audio
)


class TestSignalProcessing:
    """Test cases for signal processing functions."""
    
    @pytest.fixture
    def audio_segment(self):
        """Create a synthetic audio segment for testing."""
        # Generate a simple sine wave
        sr = 22050
        duration = 1.0  # seconds
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Mix of frequencies to simulate crowd noise
        audio = (
            0.5 * np.sin(2 * np.pi * 440 * t) +  # 440 Hz tone
            0.3 * np.sin(2 * np.pi * 880 * t) +  # 880 Hz tone
            0.2 * np.random.randn(len(t))        # Noise
        )
        return audio
    
    def test_normalize_audio(self, audio_segment):
        """Test audio normalization."""
        # Scale the audio to have values outside [-1, 1]
        scaled_audio = audio_segment * 2.5
        
        normalized = normalize_audio(scaled_audio)
        assert isinstance(normalized, np.ndarray)
        assert np.max(np.abs(normalized)) <= 1.0
        assert len(normalized) == len(scaled_audio)
    
    def test_normalize_audio_zero_signal(self):
        """Test normalization of zero signal."""
        zero_audio = np.zeros(1000)
        normalized = normalize_audio(zero_audio)
        assert isinstance(normalized, np.ndarray)
        assert np.array_equal(normalized, zero_audio)
    
    def test_apply_bandpass_filter(self, audio_segment):
        """Test bandpass filter application."""
        filtered = apply_bandpass_filter(audio_segment)
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == len(audio_segment)
    
    def test_segment_audio(self, audio_segment):
        """Test audio segmentation."""
        sr = 22050
        segment_duration = 0.1  # 100 ms
        
        segments = segment_audio(audio_segment, sr, segment_duration)
        assert isinstance(segments, list)
        assert len(segments) > 0
        
        # Check segment length
        expected_segment_length = int(segment_duration * sr)
        for segment in segments[:-1]:  # All but the last segment
            assert len(segment) == expected_segment_length
    
    def test_segment_audio_short_input(self):
        """Test segmentation with input shorter than segment length."""
        short_audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        sr = 22050
        segment_duration = 1.0  # 1 second
        
        segments = segment_audio(short_audio, sr, segment_duration)
        assert isinstance(segments, list)
        assert len(segments) == 1
        assert np.array_equal(segments[0], short_audio)
    
    def test_detect_silence(self, audio_segment):
        """Test silence detection."""
        # Create audio with silent regions
        audio_with_silence = np.copy(audio_segment)
        silence_start = len(audio_with_silence) // 3
        silence_end = 2 * len(audio_with_silence) // 3
        audio_with_silence[silence_start:silence_end] = 0.0
        
        silent_regions = detect_silence(audio_with_silence, threshold=0.01)
        assert isinstance(silent_regions, list)
        
        # Check if at least one silent region was detected
        if len(silent_regions) > 0:
            for start, end in silent_regions:
                assert isinstance(start, float)
                assert isinstance(end, float)
                assert start < end
    
    def test_remove_dc_offset(self):
        """Test DC offset removal."""
        # Create audio with DC offset
        audio_with_offset = np.ones(1000) * 0.5 + np.sin(np.linspace(0, 10 * np.pi, 1000))
        
        corrected = remove_dc_offset(audio_with_offset)
        assert isinstance(corrected, np.ndarray)
        assert len(corrected) == len(audio_with_offset)
        assert abs(np.mean(corrected)) < 1e-10  # Mean should be very close to zero
    
    def test_preprocess_audio(self, audio_segment):
        """Test audio preprocessing."""
        # Add DC offset
        audio_with_offset = audio_segment + 0.5
        
        # Preprocess
        processed = preprocess_audio(audio_with_offset)
        assert isinstance(processed, np.ndarray)
        assert len(processed) == len(audio_with_offset)
        assert abs(np.mean(processed)) < 1e-10  # DC offset should be removed
        assert np.max(np.abs(processed)) <= 1.0  # Should be normalized
    
    def test_preprocess_audio_no_normalization(self, audio_segment):
        """Test preprocessing without normalization."""
        processed = preprocess_audio(audio_segment, normalize=False)
        assert isinstance(processed, np.ndarray)
        assert len(processed) == len(audio_segment)
    
    def test_signal_processing_error_handling(self):
        """Test error handling in signal processing."""
        with patch("numpy.mean", side_effect=Exception("Test error")):
            with pytest.raises(AudioProcessingError):
                remove_dc_offset(np.array([0.1, 0.2, 0.3]))