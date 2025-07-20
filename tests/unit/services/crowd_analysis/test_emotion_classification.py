"""Unit tests for emotion classification with context-aware analysis."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from crowd_sentiment_music_generator.exceptions.audio_processing_error import AudioProcessingError
from crowd_sentiment_music_generator.models.data.crowd_emotion import CrowdEmotion
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.services.crowd_analysis.emotion_classification import EmotionClassifier


@pytest.fixture
def emotion_classifier():
    """Create an EmotionClassifier instance for testing."""
    # Use a temporary directory for models during testing
    config = SystemConfig(models_path="./test_models")
    
    # Create the test models directory if it doesn't exist
    os.makedirs(config.models_path, exist_ok=True)
    
    # Create the classifier with mocked model loading
    with patch.object(EmotionClassifier, '_load_models'):
        classifier = EmotionClassifier(config)
        
        # Mock the models and scaler
        classifier.models = {
            "random_forest": MagicMock()
        }
        classifier.scaler = MagicMock()
        classifier.audio_classifier = None
        
        # Configure the random forest mock to return predictable results
        classifier.models["random_forest"].classes_ = np.array([
            "excitement", "joy", "tension", "disappointment", 
            "anger", "anticipation", "neutral"
        ])
        classifier.models["random_forest"].predict_proba = MagicMock(
            return_value=np.array([[0.1, 0.2, 0.3, 0.05, 0.05, 0.2, 0.1]])
        )
        
        # Configure the scaler mock to return the input unchanged
        classifier.scaler.transform = MagicMock(side_effect=lambda x: x)
        
        yield classifier
    
    # Clean up the test models directory after tests
    import shutil
    if os.path.exists("./test_models"):
        shutil.rmtree("./test_models")


@pytest.fixture
def sample_features():
    """Create sample audio features for testing."""
    return {
        "rms_energy": 0.5,
        "spectral_centroid": 2000.0,
        "spectral_rolloff": 4000.0,
        "zero_crossing_rate": 0.1,
        "tempo": 120.0,
        "spectral_contrast": 10.0,
        "mfcc_1": 10.0,
        "mfcc_2": -5.0,
        "mfcc_3": 2.0,
        "isolation_quality": 0.9
    }


def test_emotion_classifier_initialization():
    """Test that EmotionClassifier initializes correctly."""
    with patch.object(EmotionClassifier, '_load_models'):
        classifier = EmotionClassifier()
        assert classifier is not None
        assert classifier.VALID_EMOTIONS == [
            "excitement", "joy", "tension", "disappointment", 
            "anger", "anticipation", "neutral"
        ]


def test_prepare_feature_vector(emotion_classifier, sample_features):
    """Test that feature vectors are prepared correctly."""
    feature_vector = emotion_classifier._prepare_feature_vector(sample_features)
    
    # Check that the feature vector contains the expected features
    assert len(feature_vector) > 0
    assert feature_vector[0] == sample_features["rms_energy"]
    assert feature_vector[1] == sample_features["spectral_centroid"]
    
    # Check that missing features are set to 0.0
    sample_features_missing = sample_features.copy()
    del sample_features_missing["spectral_centroid"]
    feature_vector_missing = emotion_classifier._prepare_feature_vector(sample_features_missing)
    assert feature_vector_missing[1] == 0.0


def test_classify_emotion_basic(emotion_classifier, sample_features):
    """Test basic emotion classification without context."""
    # Configure the mock to return "tension" as the most likely emotion
    emotion_classifier.models["random_forest"].predict_proba = MagicMock(
        return_value=np.array([[0.1, 0.2, 0.5, 0.05, 0.05, 0.05, 0.05]])
    )
    
    emotion, confidence = emotion_classifier.classify_emotion(sample_features)
    
    assert emotion == "tension"
    assert confidence == 0.5
    assert 0.0 <= confidence <= 1.0


def test_classify_emotion_with_context(emotion_classifier, sample_features):
    """Test emotion classification with match context."""
    # Configure the mock to return "neutral" as the most likely emotion
    emotion_classifier.models["random_forest"].predict_proba = MagicMock(
        return_value=np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4]])
    )
    
    # Test with goal event context
    match_context = {
        "last_event_type": "goal",
        "score_difference": 1,
        "time_remaining": 30
    }
    
    emotion, confidence = emotion_classifier.classify_emotion(sample_features, match_context)
    
    # With a goal event and high energy, should override to excitement
    assert emotion == "excitement"
    assert confidence >= 0.4  # Should be at least as high as the original confidence


def test_measure_intensity(emotion_classifier, sample_features):
    """Test intensity measurement."""
    intensity = emotion_classifier.measure_intensity(sample_features, "excitement")
    
    assert 0.0 <= intensity <= 100.0
    
    # Test with different emotions to ensure they affect intensity
    intensity_excitement = emotion_classifier.measure_intensity(sample_features, "excitement")
    intensity_neutral = emotion_classifier.measure_intensity(sample_features, "neutral")
    
    # Excitement should have higher intensity than neutral for the same features
    assert intensity_excitement > intensity_neutral


def test_create_crowd_emotion(emotion_classifier, sample_features):
    """Test creation of CrowdEmotion objects."""
    # Configure the mock to return "joy" as the most likely emotion
    emotion_classifier.models["random_forest"].predict_proba = MagicMock(
        return_value=np.array([[0.1, 0.6, 0.1, 0.05, 0.05, 0.05, 0.05]])
    )
    
    timestamp = 1234.56
    crowd_emotion = emotion_classifier.create_crowd_emotion(sample_features, timestamp)
    
    assert isinstance(crowd_emotion, CrowdEmotion)
    assert crowd_emotion.emotion == "joy"
    assert 0.0 <= crowd_emotion.confidence <= 1.0
    assert 0.0 <= crowd_emotion.intensity <= 100.0
    assert crowd_emotion.timestamp == timestamp
    assert crowd_emotion.audio_features == sample_features


def test_error_handling(emotion_classifier):
    """Test error handling in emotion classification."""
    # Test with empty features
    with pytest.raises(AudioProcessingError):
        emotion_classifier.classify_emotion({})
    
    # Test with invalid features
    with pytest.raises(AudioProcessingError):
        emotion_classifier.models["random_forest"].predict_proba = MagicMock(
            side_effect=Exception("Mock error")
        )
        emotion_classifier.classify_emotion({"rms_energy": 0.5})
de
f test_predict_emotion_from_context(emotion_classifier):
    """Test prediction of emotion from match context."""
    # Test with goal event for home team
    emotion, confidence = emotion_classifier._predict_emotion_from_context(
        event_type="goal",
        score_diff=1,
        time_remaining=30,
        is_home_event=True,
        match_context={"last_event_type": "goal"}
    )
    
    assert emotion == "excitement"
    assert confidence > 0.8
    
    # Test with goal event for away team
    emotion, confidence = emotion_classifier._predict_emotion_from_context(
        event_type="goal",
        score_diff=2,
        time_remaining=30,
        is_home_event=False,
        match_context={"last_event_type": "goal"}
    )
    
    assert emotion == "disappointment"
    assert confidence > 0.5
    
    # Test with near miss late in game
    emotion, confidence = emotion_classifier._predict_emotion_from_context(
        event_type="near_miss",
        score_diff=0,
        time_remaining=5,
        is_home_event=True,
        match_context={"last_event_type": "near_miss"}
    )
    
    assert emotion == "tension"
    assert confidence > 0.7
    
    # Test with red card
    emotion, confidence = emotion_classifier._predict_emotion_from_context(
        event_type="red_card",
        score_diff=0,
        time_remaining=30,
        is_home_event=True,
        match_context={"last_event_type": "red_card"}
    )
    
    assert emotion == "anger"
    assert confidence > 0.7
    
    # Test with no significant event
    emotion, confidence = emotion_classifier._predict_emotion_from_context(
        event_type="throw_in",
        score_diff=0,
        time_remaining=30,
        is_home_event=True,
        match_context={"last_event_type": "throw_in"}
    )
    
    assert emotion is None
    assert confidence == 0.0


def test_calculate_audio_quality_factor(emotion_classifier, sample_features):
    """Test calculation of audio quality factor."""
    # Test with good quality audio
    quality_factor = emotion_classifier._calculate_audio_quality_factor(sample_features)
    assert 0.5 <= quality_factor <= 1.0
    assert quality_factor > 0.9  # Should be high for good quality
    
    # Test with poor isolation quality
    poor_features = sample_features.copy()
    poor_features["isolation_quality"] = 0.3
    quality_factor = emotion_classifier._calculate_audio_quality_factor(poor_features)
    assert 0.5 <= quality_factor <= 1.0
    assert quality_factor < 0.9  # Should be lower for poor quality
    
    # Test with low energy
    low_energy_features = sample_features.copy()
    low_energy_features["rms_energy"] = 0.05
    quality_factor = emotion_classifier._calculate_audio_quality_factor(low_energy_features)
    assert 0.5 <= quality_factor <= 1.0
    assert quality_factor < 0.9  # Should be lower for low energy
    
    # Test with SNR
    snr_features = sample_features.copy()
    snr_features["snr"] = 5.0  # Poor SNR
    quality_factor = emotion_classifier._calculate_audio_quality_factor(snr_features)
    assert 0.5 <= quality_factor <= 1.0
    assert quality_factor < 0.9  # Should be lower for poor SNR


def test_calculate_confidence_score(emotion_classifier, sample_features):
    """Test calculation of comprehensive confidence score."""
    # Test with consistent context
    match_context = {
        "last_event_type": "goal",
        "score_difference": 1,
        "time_remaining": 30,
        "match_importance": 0.8,
        "previous_emotions": ["excitement", "joy"]
    }
    
    confidence = emotion_classifier._calculate_confidence_score(
        emotion="excitement",
        base_confidence=0.7,
        features=sample_features,
        match_context=match_context
    )
    
    assert 0.0 <= confidence <= 1.0
    assert confidence > 0.7  # Should increase for consistent context
    
    # Test with inconsistent context
    match_context = {
        "last_event_type": "goal",
        "score_difference": 1,
        "time_remaining": 30,
        "match_importance": 0.8,
        "previous_emotions": ["tension", "anger", "tension"]
    }
    
    confidence = emotion_classifier._calculate_confidence_score(
        emotion="disappointment",
        base_confidence=0.7,
        features=sample_features,
        match_context=match_context
    )
    
    assert 0.0 <= confidence <= 1.0
    assert confidence < 0.7  # Should decrease for inconsistent context
    
    # Test with multi-modal agreement
    match_context = {
        "last_event_type": "goal",
        "score_difference": 1,
        "time_remaining": 30,
        "audio_classifier_prediction": "excitement"
    }
    
    confidence = emotion_classifier._calculate_confidence_score(
        emotion="excitement",
        base_confidence=0.7,
        features=sample_features,
        match_context=match_context
    )
    
    assert 0.0 <= confidence <= 1.0
    assert confidence > 0.7  # Should increase for multi-modal agreement
    
    # Test with critical moment in important match
    match_context = {
        "last_event_type": "penalty_awarded",
        "score_difference": 0,
        "time_remaining": 5,
        "match_importance": 0.9
    }
    
    confidence = emotion_classifier._calculate_confidence_score(
        emotion="tension",
        base_confidence=0.7,
        features=sample_features,
        match_context=match_context
    )
    
    assert 0.0 <= confidence <= 1.0
    assert confidence > 0.7  # Should increase for critical moment


def test_apply_context_adjustments(emotion_classifier, sample_features):
    """Test context-based adjustments to emotion classification."""
    # Test with strong context influence
    match_context = {
        "last_event_type": "goal",
        "score_difference": 1,
        "time_remaining": 30,
        "home_team": "Team A",
        "away_team": "Team B",
        "event_team": "Team A",
        "match_importance": 0.8
    }
    
    # Initial classification is "neutral"
    emotion, confidence = emotion_classifier._apply_context_adjustments(
        emotion="neutral",
        confidence=0.6,
        features=sample_features,
        match_context=match_context
    )
    
    # Should adjust to excitement or joy for home team goal
    assert emotion in ["excitement", "joy"]
    assert confidence > 0.6  # Confidence should increase
    
    # Test with weak context influence
    match_context = {
        "last_event_type": "throw_in",
        "score_difference": 1,
        "time_remaining": 30,
        "home_team": "Team A",
        "away_team": "Team B",
        "event_team": "Team A",
        "match_importance": 0.3
    }
    
    # Initial classification is "tension"
    initial_emotion = "tension"
    initial_confidence = 0.8
    emotion, confidence = emotion_classifier._apply_context_adjustments(
        emotion=initial_emotion,
        confidence=initial_confidence,
        features=sample_features,
        match_context=match_context
    )
    
    # Should not change for weak context
    assert emotion == initial_emotion
    assert abs(confidence - initial_confidence) < 0.1  # Confidence should not change much