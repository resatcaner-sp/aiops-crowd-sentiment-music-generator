"""Unit tests for feature extraction module."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from crowd_sentiment_music_generator.exceptions.audio_processing_error import AudioProcessingError
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.services.crowd_analysis.feature_extraction import (
    extract_rms_energy,
    extract_spectral_centroid,
    extract_spectral_rolloff,
    extract_zero_crossing_rate,
    estimate_tempo,
    extract_spectral_contrast,
    extract_mfcc,
    extract_all_features,
    isolate_crowd_noise
)


class TestFeatureExtraction:
    """Test cases for feature extraction functions."""
    
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
    
    @pytest.fixture
    def system_config(self):
        """Create a system configuration for testing."""
        return SystemConfig()
    
    def test_extract_rms_energy(self, audio_segment):
        """Test RMS energy extraction."""
        rms = extract_rms_energy(audio_segment)
        assert isinstance(rms, float)
        assert 0.0 <= rms <= 1.0
    
    def test_extract_spectral_centroid(self, audio_segment):
        """Test spectral centroid extraction."""
        centroid = extract_spectral_centroid(audio_segment)
        assert isinstance(centroid, float)
        assert centroid > 0
    
    def test_extract_spectral_rolloff(self, audio_segment):
        """Test spectral rolloff extraction."""
        rolloff = extract_spectral_rolloff(audio_segment)
        assert isinstance(rolloff, float)
        assert rolloff > 0
    
    def test_extract_zero_crossing_rate(self, audio_segment):
        """Test zero crossing rate extraction."""
        zcr = extract_zero_crossing_rate(audio_segment)
        assert isinstance(zcr, float)
        assert 0.0 <= zcr <= 1.0
    
    def test_estimate_tempo(self, audio_segment):
        """Test tempo estimation."""
        tempo = estimate_tempo(audio_segment)
        assert isinstance(tempo, float)
        assert tempo > 0
    
    def test_extract_spectral_contrast(self, audio_segment):
        """Test spectral contrast extraction."""
        contrast = extract_spectral_contrast(audio_segment)
        assert isinstance(contrast, float)
    
    def test_extract_mfcc(self, audio_segment):
        """Test MFCC extraction."""
        mfccs = extract_mfcc(audio_segment)
        assert isinstance(mfccs, np.ndarray)
        assert len(mfccs) == 13  # Default n_mfcc is 13
    
    def test_extract_all_features(self, audio_segment, system_config):
        """Test extraction of all features."""
        features = extract_all_features(audio_segment, config=system_config)
        assert isinstance(features, dict)
        
        # Check that all expected features are present
        expected_features = [
            "rms_energy", "spectral_centroid", "spectral_rolloff",
            "zero_crossing_rate", "tempo", "spectral_contrast"
        ]
        for feature in expected_features:
            assert feature in features
        
        # Check that MFCCs are present
        for i in range(1, 14):  # 13 MFCCs
            assert f"mfcc_{i}" in features
    
    def test_extract_all_features_empty_audio(self):
        """Test extraction with empty audio segment."""
        with pytest.raises(AudioProcessingError):
            extract_all_features(np.array([]))
    
    def test_isolate_crowd_noise(self, audio_segment):
        """Test crowd noise isolation."""
        crowd_noise, quality_score = isolate_crowd_noise(audio_segment)
        assert isinstance(crowd_noise, np.ndarray)
        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0
        assert len(crowd_noise) == len(audio_segment)
    
    def test_feature_extraction_error_handling(self):
        """Test error handling in feature extraction."""
        with patch("librosa.feature.rms", side_effect=Exception("Test error")):
            with pytest.raises(AudioProcessingError):
                extract_rms_energy(np.array([0.1, 0.2, 0.3]))
    
    def test_extract_all_features_with_default_config(self, audio_segment):
        """Test extract_all_features with default configuration."""
        features = extract_all_features(audio_segment)
        assert isinstance(features, dict)
        assert len(features) > 0