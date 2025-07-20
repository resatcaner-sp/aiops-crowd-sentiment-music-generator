"""Unit tests for crowd analyzer with context-aware emotion classification."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from crowd_sentiment_music_generator.exceptions.audio_processing_error import AudioProcessingError
from crowd_sentiment_music_generator.models.data.crowd_emotion import CrowdEmotion
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.services.crowd_analysis.crowd_analyzer import CrowdAnalyzer


class TestCrowdAnalyzerEmotion:
    """Test cases for CrowdAnalyzer emotion classification functionality."""
    
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
    def sample_features(self):
        """Create sample audio features for testing."""
        return {
            "rms_energy": 0.5,
            "spectral_centroid": 2000.0,
            "spectral_rolloff": 4000.0,
            "zero_crossing_rate": 0.1,
            "tempo": 120.0,
            "spectral_contrast": 10.0,
            "isolation_quality": 0.9
        }
    
    @pytest.fixture
    def sample_crowd_emotion(self):
        """Create a sample CrowdEmotion object for testing."""
        return CrowdEmotion(
            emotion="excitement",
            intensity=75.0,
            confidence=0.85,
            timestamp=1234.56,
            audio_features={"rms_energy": 0.5}
        )
    
    @pytest.fixture
    def crowd_analyzer_with_mock_classifier(self):
        """Create a CrowdAnalyzer with mocked EmotionClassifier."""
        with patch('crowd_sentiment_music_generator.services.crowd_analysis.emotion_classification.EmotionClassifier') as mock_classifier_class:
            # Create the analyzer
            analyzer = CrowdAnalyzer()
            
            # Replace the emotion classifier with a mock
            analyzer.emotion_classifier = MagicMock()
            
            yield analyzer
    
    def test_initialization_with_emotion_classifier(self):
        """Test that CrowdAnalyzer initializes with an EmotionClassifier."""
        with patch('crowd_sentiment_music_generator.services.crowd_analysis.emotion_classification.EmotionClassifier') as mock_classifier_class:
            analyzer = CrowdAnalyzer()
            assert analyzer.emotion_classifier is not None
            mock_classifier_class.assert_called_once()
    
    def test_classify_emotion(self, crowd_analyzer_with_mock_classifier, sample_features, sample_crowd_emotion):
        """Test emotion classification."""
        analyzer = crowd_analyzer_with_mock_classifier
        
        # Configure mock
        analyzer.emotion_classifier.create_crowd_emotion.return_value = sample_crowd_emotion
        
        # Classify emotion
        timestamp = 1234.56
        result = analyzer.classify_emotion(sample_features, timestamp)
        
        # Check results
        assert result is sample_crowd_emotion
        analyzer.emotion_classifier.create_crowd_emotion.assert_called_once_with(
            sample_features, timestamp, None
        )
    
    def test_classify_emotion_with_context(self, crowd_analyzer_with_mock_classifier, sample_features, sample_crowd_emotion):
        """Test emotion classification with match context."""
        analyzer = crowd_analyzer_with_mock_classifier
        
        # Configure mock
        analyzer.emotion_classifier.create_crowd_emotion.return_value = sample_crowd_emotion
        
        # Classify emotion with context
        timestamp = 1234.56
        match_context = {"last_event_type": "goal"}
        result = analyzer.classify_emotion(sample_features, timestamp, match_context)
        
        # Check results
        assert result is sample_crowd_emotion
        analyzer.emotion_classifier.create_crowd_emotion.assert_called_once_with(
            sample_features, timestamp, match_context
        )
    
    def test_classify_audio_segment(self, crowd_analyzer_with_mock_classifier, audio_segment, sample_features, sample_crowd_emotion):
        """Test classification of an audio segment."""
        analyzer = crowd_analyzer_with_mock_classifier
        
        # Configure mocks
        with patch.object(analyzer, 'process_audio_segment') as mock_process:
            mock_process.return_value = (audio_segment, sample_features)
            analyzer.emotion_classifier.create_crowd_emotion.return_value = sample_crowd_emotion
            
            # Classify audio segment
            timestamp = 1234.56
            result = analyzer.classify_audio_segment(audio_segment, timestamp)
            
            # Check results
            assert result is sample_crowd_emotion
            mock_process.assert_called_once()
            analyzer.emotion_classifier.create_crowd_emotion.assert_called_once_with(
                sample_features, timestamp, None
            )
    
    def test_classify_audio_stream(self, crowd_analyzer_with_mock_classifier, audio_segment, sample_features, sample_crowd_emotion):
        """Test classification of an audio stream."""
        analyzer = crowd_analyzer_with_mock_classifier
        
        # Configure mocks
        with patch.object(analyzer, 'process_audio_stream') as mock_process_stream:
            with patch.object(analyzer, 'get_average_features') as mock_avg_features:
                segment_features = {
                    "segment_0": {"rms_energy": 0.5},
                    "segment_1": {"rms_energy": 0.6}
                }
                mock_process_stream.return_value = segment_features
                mock_avg_features.return_value = sample_features
                analyzer.emotion_classifier.create_crowd_emotion.return_value = sample_crowd_emotion
                
                # Classify audio stream
                timestamp = 1234.56
                result = analyzer.classify_audio_stream(audio_segment, timestamp)
                
                # Check results
                assert result is sample_crowd_emotion
                mock_process_stream.assert_called_once()
                mock_avg_features.assert_called_once_with(segment_features)
                analyzer.emotion_classifier.create_crowd_emotion.assert_called_once_with(
                    sample_features, timestamp, None
                )
    
    def test_classify_audio_segment_with_context(self, crowd_analyzer_with_mock_classifier, audio_segment, sample_features, sample_crowd_emotion):
        """Test classification of an audio segment with match context."""
        analyzer = crowd_analyzer_with_mock_classifier
        
        # Configure mocks
        with patch.object(analyzer, 'process_audio_segment') as mock_process:
            mock_process.return_value = (audio_segment, sample_features)
            analyzer.emotion_classifier.create_crowd_emotion.return_value = sample_crowd_emotion
            
            # Classify audio segment with context
            timestamp = 1234.56
            match_context = {"last_event_type": "goal"}
            result = analyzer.classify_audio_segment(audio_segment, timestamp, match_context)
            
            # Check results
            assert result is sample_crowd_emotion
            mock_process.assert_called_once()
            analyzer.emotion_classifier.create_crowd_emotion.assert_called_once_with(
                sample_features, timestamp, match_context
            )
    
    def test_error_handling_in_classify_emotion(self, crowd_analyzer_with_mock_classifier, sample_features):
        """Test error handling in classify_emotion."""
        analyzer = crowd_analyzer_with_mock_classifier
        
        # Configure mock to raise an exception
        analyzer.emotion_classifier.create_crowd_emotion.side_effect = AudioProcessingError("Test error")
        
        # Classify emotion should handle the exception
        timestamp = 1234.56
        with pytest.raises(AudioProcessingError):
            analyzer.classify_emotion(sample_features, timestamp) 
   def test_update_context_history(self, crowd_analyzer_with_mock_classifier):
        """Test updating context history."""
        analyzer = crowd_analyzer_with_mock_classifier
        
        # Initial state
        assert len(analyzer.context_history) == 0
        
        # Add first context
        context1 = {"last_event_type": "goal", "match_time": 30}
        analyzer.update_context_history(context1)
        assert len(analyzer.context_history) == 1
        assert analyzer.context_history[0]["last_event_type"] == "goal"
        
        # Add second context
        context2 = {"last_event_type": "near_miss", "match_time": 35}
        analyzer.update_context_history(context2)
        assert len(analyzer.context_history) == 2
        assert analyzer.context_history[1]["last_event_type"] == "near_miss"
        
        # Test timestamp addition
        assert "timestamp" in analyzer.context_history[0]
        assert analyzer.context_history[0]["timestamp"] == 30
        
        # Test history size limit
        for i in range(20):  # Add more than max_history_size
            analyzer.update_context_history({"match_time": 40 + i})
        
        # Should be limited to max_history_size
        assert len(analyzer.context_history) == analyzer.max_history_size
        
        # Test with empty context
        analyzer.update_context_history(None)
        assert len(analyzer.context_history) == analyzer.max_history_size  # Should not change
        
        analyzer.update_context_history({})
        assert len(analyzer.context_history) == analyzer.max_history_size  # Should not change
    
    def test_enrich_match_context(self, crowd_analyzer_with_mock_classifier):
        """Test enriching match context with historical data and derived metrics."""
        analyzer = crowd_analyzer_with_mock_classifier
        
        # Test with empty context
        enriched = analyzer.enrich_match_context(None)
        assert enriched == {}
        
        enriched = analyzer.enrich_match_context({})
        assert enriched == {}
        
        # Test with basic context
        basic_context = {
            "last_event_type": "goal",
            "match_time": 75,
            "score_difference": 1
        }
        
        enriched = analyzer.enrich_match_context(basic_context)
        
        # Should add match phase
        assert "match_phase" in enriched
        assert enriched["match_phase"] == "closing"  # Based on match_time > 75
        
        # Test with historical context
        analyzer.context_history = [
            {"last_event_type": "goal", "timestamp": 30, "event_team": "Team A", "crowd_energy": 0.8},
            {"last_event_type": "yellow_card", "timestamp": 40, "event_team": "Team B", "crowd_energy": 0.6},
            {"last_event_type": "goal", "timestamp": 70, "event_team": "Team B", "crowd_energy": 0.9}
        ]
        
        enriched = analyzer.enrich_match_context({
            "last_event_type": "near_miss",
            "match_time": 75,
            "timestamp": 75,
            "score_difference": 0
        })
        
        # Should detect momentum shift (goals by different teams)
        assert "momentum_shift" in enriched
        assert enriched["momentum_shift"] is True
        
        # Should calculate crowd energy trend
        assert "crowd_energy_trend" in enriched
        assert enriched["crowd_energy_trend"] > 0  # Increasing trend
    
    def test_classify_emotion_with_context_aware_analysis(self, crowd_analyzer_with_mock_classifier, sample_features, sample_crowd_emotion):
        """Test emotion classification with context-aware analysis."""
        analyzer = crowd_analyzer_with_mock_classifier
        
        # Configure mock
        analyzer.emotion_classifier.create_crowd_emotion.return_value = sample_crowd_emotion
        
        # Classify emotion with context
        timestamp = 1234.56
        match_context = {
            "last_event_type": "goal",
            "match_time": 75,
            "score_difference": 1,
            "home_team": "Team A",
            "away_team": "Team B",
            "event_team": "Team A"
        }
        
        # First classification
        result = analyzer.classify_emotion(sample_features, timestamp, match_context)
        
        # Check results
        assert result is sample_crowd_emotion
        
        # Check that context history was updated
        assert len(analyzer.context_history) == 1
        assert analyzer.context_history[0]["last_event_type"] == "goal"
        
        # Check that emotion results were stored in context
        assert "crowd_emotion" in analyzer.context_history[0]
        assert analyzer.context_history[0]["crowd_emotion"] == sample_crowd_emotion.emotion
        assert "crowd_intensity" in analyzer.context_history[0]
        assert analyzer.context_history[0]["crowd_intensity"] == sample_crowd_emotion.intensity
        assert "crowd_energy" in analyzer.context_history[0]
        
        # Second classification should use enriched context
        with patch.object(analyzer, 'enrich_match_context') as mock_enrich:
            mock_enrich.return_value = {"enriched": True}
            
            result2 = analyzer.classify_emotion(sample_features, timestamp + 10, {
                "last_event_type": "near_miss"
            })
            
            assert result2 is sample_crowd_emotion
            mock_enrich.assert_called_once()
            analyzer.emotion_classifier.create_crowd_emotion.assert_called_with(
                sample_features, timestamp + 10, {"enriched": True}
            )