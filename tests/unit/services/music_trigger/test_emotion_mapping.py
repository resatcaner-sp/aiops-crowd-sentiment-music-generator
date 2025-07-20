"""Unit tests for emotion mapping module."""

import pytest
from unittest.mock import MagicMock, patch

from crowd_sentiment_music_generator.exceptions.music_generation_error import MusicGenerationError
from crowd_sentiment_music_generator.models.data.crowd_emotion import CrowdEmotion
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.models.data.user_preferences import UserPreferences
from crowd_sentiment_music_generator.models.music.musical_parameters import MusicalParameters
from crowd_sentiment_music_generator.services.music_trigger.emotion_mapping import EmotionMusicMapper


class TestEmotionMusicMapper:
    """Test cases for EmotionMusicMapper class."""
    
    @pytest.fixture
    def mapper(self) -> EmotionMusicMapper:
        """Create an EmotionMusicMapper instance for testing."""
        config = SystemConfig()
        return EmotionMusicMapper(config)
    
    @pytest.fixture
    def excitement_emotion(self) -> CrowdEmotion:
        """Create an excitement emotion for testing."""
        return CrowdEmotion(
            emotion="excitement",
            intensity=80.0,
            confidence=0.9,
            timestamp=1234.5,
            audio_features={"rms_energy": 0.8, "spectral_centroid": 3000.0}
        )
    
    @pytest.fixture
    def disappointment_emotion(self) -> CrowdEmotion:
        """Create a disappointment emotion for testing."""
        return CrowdEmotion(
            emotion="disappointment",
            intensity=60.0,
            confidence=0.7,
            timestamp=2345.6,
            audio_features={"rms_energy": 0.4, "spectral_centroid": 2000.0}
        )
    
    @pytest.fixture
    def user_preferences(self) -> UserPreferences:
        """Create user preferences for testing."""
        return UserPreferences(
            music_intensity=4,  # 1-5 scale
            preferred_genres=["orchestral", "electronic"],
            music_enabled=True,
            team_preferences={},
            cultural_style="european"
        )
    
    def test_set_base_musical_state(self, mapper: EmotionMusicMapper) -> None:
        """Test setting the base musical state."""
        mapper.set_base_musical_state(120.0, "D Major")
        assert mapper.base_tempo == 120.0
        assert mapper.current_key == "D Major"
    
    def test_set_cultural_style(self, mapper: EmotionMusicMapper) -> None:
        """Test setting the cultural style."""
        mapper.set_cultural_style("latin")
        assert mapper.cultural_style == "latin"
        
        # Test with invalid style
        mapper.set_cultural_style("invalid_style")
        assert mapper.cultural_style == "global"  # Default to global
    
    def test_map_emotion_to_parameters_excitement(
        self, mapper: EmotionMusicMapper, excitement_emotion: CrowdEmotion
    ) -> None:
        """Test mapping an excitement emotion to musical parameters."""
        parameters = mapper.map_emotion_to_parameters(excitement_emotion)
        
        assert parameters is not None
        assert isinstance(parameters, MusicalParameters)
        assert parameters.mood == "energetic"
        assert "brass" in parameters.instrumentation
        assert "percussion" in parameters.instrumentation
        assert parameters.tempo > mapper.base_tempo  # Tempo should increase for excitement
        assert parameters.intensity > 0.7  # High intensity for excitement
    
    def test_map_emotion_to_parameters_disappointment(
        self, mapper: EmotionMusicMapper, disappointment_emotion: CrowdEmotion
    ) -> None:
        """Test mapping a disappointment emotion to musical parameters."""
        parameters = mapper.map_emotion_to_parameters(disappointment_emotion)
        
        assert parameters is not None
        assert isinstance(parameters, MusicalParameters)
        assert parameters.mood == "somber"
        assert "strings" in parameters.instrumentation
        assert "piano" in parameters.instrumentation
        assert parameters.tempo < mapper.base_tempo  # Tempo should decrease for disappointment
        assert parameters.intensity < 0.5  # Lower intensity for disappointment
    
    def test_map_emotion_to_parameters_with_current(
        self, mapper: EmotionMusicMapper, excitement_emotion: CrowdEmotion
    ) -> None:
        """Test mapping an emotion with current parameters."""
        current_parameters = MusicalParameters(
            tempo=120.0,
            key="D Major",
            intensity=0.5,
            instrumentation=["piano", "strings"],
            mood="neutral"
        )
        
        parameters = mapper.map_emotion_to_parameters(excitement_emotion, current_parameters)
        
        assert parameters is not None
        assert isinstance(parameters, MusicalParameters)
        assert parameters.tempo > 120.0  # Tempo should increase for excitement
        assert parameters.intensity > 0.5  # Intensity should increase for excitement
        assert "brass" in parameters.instrumentation
        assert "percussion" in parameters.instrumentation
    
    def test_scale_parameters_with_intensity(self, mapper: EmotionMusicMapper) -> None:
        """Test scaling parameters with intensity."""
        parameters = MusicalParameters(
            tempo=100.0,
            key="C Major",
            intensity=0.5,
            instrumentation=["strings", "piano"],
            mood="neutral",
            transition_duration=5.0
        )
        
        # Test high intensity
        high_intensity = 90.0
        high_params = mapper.scale_parameters_with_intensity(parameters, high_intensity)
        
        assert high_params.tempo > parameters.tempo
        assert high_params.intensity > parameters.intensity
        assert "percussion" in high_params.instrumentation
        assert high_params.transition_duration < parameters.transition_duration
        
        # Test low intensity
        low_intensity = 20.0
        low_params = mapper.scale_parameters_with_intensity(parameters, low_intensity)
        
        assert low_params.tempo < high_params.tempo
        assert low_params.intensity < high_params.intensity
        assert "percussion" not in low_params.instrumentation
        assert "brass" not in low_params.instrumentation
        assert "strings" in low_params.instrumentation
        assert low_params.transition_duration > high_params.transition_duration
    
    def test_apply_user_preferences(
        self, mapper: EmotionMusicMapper, user_preferences: UserPreferences
    ) -> None:
        """Test applying user preferences to parameters."""
        parameters = MusicalParameters(
            tempo=100.0,
            key="C Major",
            intensity=0.5,
            instrumentation=["strings", "piano"],
            mood="neutral",
            transition_duration=5.0
        )
        
        adjusted = mapper.apply_user_preferences(parameters, user_preferences)
        
        assert adjusted is not None
        assert isinstance(adjusted, MusicalParameters)
        
        # Check intensity adjustment (user preference is 4 on 1-5 scale)
        assert adjusted.intensity > parameters.intensity
        
        # Check instrumentation adjustment (user prefers orchestral)
        assert "brass" in adjusted.instrumentation
        assert "woodwinds" in adjusted.instrumentation
        assert "percussion" in adjusted.instrumentation
        
        # Check cultural style was applied
        assert mapper.cultural_style == "european"
    
    def test_select_key_for_emotion(self, mapper: EmotionMusicMapper) -> None:
        """Test key selection based on emotion preference."""
        # Test with major preference
        major_key = mapper._select_key_for_emotion("major", "C Major")
        assert major_key in ["C Major", "G Major", "D Major", "A Major", "E Major", "F Major"]
        
        # Test with minor preference
        minor_key = mapper._select_key_for_emotion("minor", "C Major")
        assert minor_key in ["A Minor", "E Minor", "D Minor", "G Minor", "C Minor", "F Minor"]
        
        # Test with neutral preference (should keep current key)
        neutral_key = mapper._select_key_for_emotion("neutral", "D Major")
        assert neutral_key == "D Major"
    
    def test_adapt_instrumentation_to_culture(self, mapper: EmotionMusicMapper) -> None:
        """Test adapting instrumentation based on cultural style."""
        base_instrumentation = ["strings", "piano"]
        
        # Test with European style
        mapper.set_cultural_style("european")
        european_instr = mapper._adapt_instrumentation_to_culture(base_instrumentation)
        assert "woodwinds" in european_instr or "brass" in european_instr
        
        # Test with Latin style
        mapper.set_cultural_style("latin")
        latin_instr = mapper._adapt_instrumentation_to_culture(base_instrumentation)
        assert "percussion" in latin_instr or "acoustic_guitar" in latin_instr
        
        # Test with global style (should not change)
        mapper.set_cultural_style("global")
        global_instr = mapper._adapt_instrumentation_to_culture(base_instrumentation)
        assert set(global_instr) == set(base_instrumentation)
    
    @patch("crowd_sentiment_music_generator.services.music_trigger.emotion_mapping.with_error_handling")
    def test_error_handling(self, mock_error_handler: MagicMock, mapper: EmotionMusicMapper) -> None:
        """Test that error handling decorator is applied to public methods."""
        # Configure the mock to pass through the original function
        mock_error_handler.side_effect = lambda f: f
        
        # Verify error handling is applied to public methods
        assert hasattr(mapper.map_emotion_to_parameters, "__wrapped__")
        assert hasattr(mapper.scale_parameters_with_intensity, "__wrapped__")
        assert hasattr(mapper.apply_user_preferences, "__wrapped__")