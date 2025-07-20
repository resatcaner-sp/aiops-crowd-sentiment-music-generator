"""Unit tests for crowd analyzer module."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from crowd_sentiment_music_generator.exceptions.audio_processing_error import AudioProcessingError
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.services.crowd_analysis.crowd_analyzer import CrowdAnalyzer


class TestCrowdAnalyzer:
    """Test cases for CrowdAnalyzer class."""
    
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
    
    @pytest.fixture
    def crowd_analyzer(self, system_config):
        """Create a CrowdAnalyzer instance for testing."""
        return CrowdAnalyzer(config=system_config)
    
    def test_initialization(self, system_config):
        """Test CrowdAnalyzer initialization."""
        analyzer = CrowdAnalyzer(config=system_config)
        assert analyzer.config == system_config
        
        # Test initialization with default config
        default_analyzer = CrowdAnalyzer()
        assert isinstance(default_analyzer.config, SystemConfig)
    
    def test_process_audio_segment(self, crowd_analyzer, audio_segment):
        """Test processing of a single audio segment."""
        crowd_noise, features = crowd_analyzer.process_audio_segment(audio_segment)
        
        assert isinstance(crowd_noise, np.ndarray)
        assert isinstance(features, dict)
        assert len(crowd_noise) == len(audio_segment)
        
        # Check that essential features are present
        essential_features = [
            "rms_energy", "spectral_centroid", "zero_crossing_rate",
            "isolation_quality"
        ]
        for feature in essential_features:
            assert feature in features
    
    def test_process_audio_segment_empty(self, crowd_analyzer):
        """Test processing of an empty audio segment."""
        with pytest.raises(AudioProcessingError):
            crowd_analyzer.process_audio_segment(np.array([]))
    
    def test_process_audio_stream(self, crowd_analyzer, audio_segment):
        """Test processing of an audio stream."""
        # Create a longer audio stream by repeating the segment
        audio_stream = np.tile(audio_segment, 5)
        
        segment_features = crowd_analyzer.process_audio_stream(
            audio_stream, segment_duration=0.5
        )
        
        assert isinstance(segment_features, dict)
        assert len(segment_features) > 0
        
        # Check that each segment has features
        for segment_id, features in segment_features.items():
            assert isinstance(segment_id, str)
            assert isinstance(features, dict)
            assert "rms_energy" in features
    
    def test_get_average_features(self, crowd_analyzer):
        """Test calculation of average features."""
        # Create mock segment features
        segment_features = {
            "segment_0": {"rms_energy": 0.5, "spectral_centroid": 1000.0},
            "segment_1": {"rms_energy": 0.7, "spectral_centroid": 1200.0},
            "segment_2": {"rms_energy": 0.6, "spectral_centroid": 1100.0}
        }
        
        avg_features = crowd_analyzer.get_average_features(segment_features)
        
        assert isinstance(avg_features, dict)
        assert "rms_energy" in avg_features
        assert "spectral_centroid" in avg_features
        
        # Check average calculations
        assert avg_features["rms_energy"] == pytest.approx((0.5 + 0.7 + 0.6) / 3)
        assert avg_features["spectral_centroid"] == pytest.approx((1000.0 + 1200.0 + 1100.0) / 3)
    
    def test_get_average_features_empty(self, crowd_analyzer):
        """Test average features calculation with empty input."""
        with pytest.raises(AudioProcessingError):
            crowd_analyzer.get_average_features({})
    
    def test_error_handling_in_process_audio_stream(self, crowd_analyzer, audio_segment):
        """Test error handling in process_audio_stream."""
        # Create a longer audio stream by repeating the segment
        audio_stream = np.tile(audio_segment, 5)
        
        # Mock process_audio_segment to fail for some segments
        original_method = crowd_analyzer.process_audio_segment
        
        def mock_process(*args, **kwargs):
            # Fail for every other call
            mock_process.counter += 1
            if mock_process.counter % 2 == 0:
                raise AudioProcessingError("Test error")
            return original_method(*args, **kwargs)
        
        mock_process.counter = 0
        
        with patch.object(crowd_analyzer, 'process_audio_segment', side_effect=mock_process):
            segment_features = crowd_analyzer.process_audio_stream(
                audio_stream, segment_duration=0.5
            )
            
            # Should still return features for segments that didn't fail
            assert isinstance(segment_features, dict)
            assert len(segment_features) > 0