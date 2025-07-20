"""Unit tests for music trigger module."""

import pytest
from unittest.mock import MagicMock, patch

from crowd_sentiment_music_generator.exceptions.music_generation_error import MusicGenerationError
from crowd_sentiment_music_generator.models.data.match_event import MatchEvent
from crowd_sentiment_music_generator.models.data.crowd_emotion import CrowdEmotion
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.models.data.user_preferences import UserPreferences
from crowd_sentiment_music_generator.models.music.musical_parameters import MusicalParameters
from crowd_sentiment_music_generator.services.music_trigger.music_trigger import MusicTrigger


class TestMusicTrigger:
    """Test cases for MusicTrigger class."""
    
    @pytest.fixture
    def trigger(self) -> MusicTrigger:
        """Create a MusicTrigger instance for testing."""
        config = SystemConfig()
        return MusicTrigger(config)
    
    @pytest.fixture
    def goal_event(self) -> MatchEvent:
        """Create a goal event for testing."""
        return MatchEvent(
            id="test_goal",
            type="goal",
            timestamp=1234.5,
            team_id="home_team"
        )
    
    @pytest.fixture
    def yellow_card_event(self) -> MatchEvent:
        """Create a yellow card event for testing."""
        return MatchEvent(
            id="test_yellow",
            type="yellow_card",
            timestamp=2345.6,
            team_id="away_team"
        )
    
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
            preferred_genres=["orchestral"],
            music_enabled=True,
            team_preferences={},
            cultural_style="european"
        )
    
    @pytest.fixture
    def events_list(self) -> list[MatchEvent]:
        """Create a list of events for testing."""
        return [
            MatchEvent(
                id="event1",
                type="goal",
                timestamp=1000.0,
                team_id="home_team"
            ),
            MatchEvent(
                id="event2",
                type="yellow_card",
                timestamp=1010.0,
                team_id="away_team"
            ),
            MatchEvent(
                id="event3",
                type="corner",
                timestamp=1020.0,
                team_id="home_team"
            ),
            MatchEvent(
                id="event4",
                type="near_miss",
                timestamp=1030.0,
                team_id="home_team"
            ),
            MatchEvent(
                id="event5",
                type="red_card",
                timestamp=1040.0,
                team_id="away_team"
            )
        ]
    
    def test_set_music_engine_callback(self, trigger: MusicTrigger) -> None:
        """Test setting the music engine callback."""
        callback = MagicMock()
        trigger.set_music_engine_callback(callback)
        assert trigger.music_engine_callback == callback
    
    def test_set_accent_callback(self, trigger: MusicTrigger) -> None:
        """Test setting the accent callback."""
        callback = MagicMock()
        trigger.set_accent_callback(callback)
        assert trigger.accent_callback == callback
    
    def test_set_base_musical_state(self, trigger: MusicTrigger) -> None:
        """Test setting the base musical state."""
        trigger.set_base_musical_state(120.0, "D Major")
        assert trigger.event_mapper.base_tempo == 120.0
        assert trigger.event_mapper.current_key == "D Major"
        assert trigger.current_parameters is not None
        assert trigger.current_parameters.tempo == 120.0
        assert trigger.current_parameters.key == "D Major"
    
    def test_process_event(self, trigger: MusicTrigger, goal_event: MatchEvent) -> None:
        """Test processing a match event."""
        # Set up callbacks
        music_callback = MagicMock()
        accent_callback = MagicMock()
        trigger.set_music_engine_callback(music_callback)
        trigger.set_accent_callback(accent_callback)
        
        # Process event
        parameters = trigger.process_event(goal_event)
        
        # Verify parameters
        assert parameters is not None
        assert isinstance(parameters, MusicalParameters)
        assert parameters.intensity == 1.0
        assert parameters.mood == "triumphant"
        assert "brass" in parameters.instrumentation
        
        # Verify callbacks were called
        music_callback.assert_called_once()
        accent_callback.assert_called_once()
        
        # Verify current parameters were updated
        assert trigger.current_parameters == parameters
        assert trigger.last_event_time == goal_event.timestamp
    
    def test_process_event_cooldown(
        self, trigger: MusicTrigger, goal_event: MatchEvent, yellow_card_event: MatchEvent
    ) -> None:
        """Test event cooldown prevents processing events too frequently."""
        # Set up initial state
        trigger.last_event_time = goal_event.timestamp
        trigger.current_parameters = MusicalParameters(
            tempo=100.0,
            key="C Major",
            intensity=1.0,
            instrumentation=["brass", "percussion"],
            mood="triumphant"
        )
        
        # Set up callbacks
        music_callback = MagicMock()
        accent_callback = MagicMock()
        trigger.set_music_engine_callback(music_callback)
        trigger.set_accent_callback(accent_callback)
        
        # Set yellow card timestamp to be within cooldown period
        yellow_card_event.timestamp = goal_event.timestamp + 2.0  # Less than cooldown
        
        # Process event
        parameters = trigger.process_event(yellow_card_event)
        
        # Verify parameters are unchanged
        assert parameters == trigger.current_parameters
        
        # Verify callbacks were not called
        music_callback.assert_not_called()
        accent_callback.assert_not_called()
        
        # Verify last event time was not updated
        assert trigger.last_event_time == goal_event.timestamp
    
    def test_process_event_with_context(
        self, trigger: MusicTrigger, goal_event: MatchEvent
    ) -> None:
        """Test processing an event with match context."""
        # Set up match context
        match_context = {
            "score_difference": 1,
            "time_remaining": 5,
            "match_importance": 0.8
        }
        
        # Process event with context
        parameters = trigger.process_event(goal_event, match_context)
        
        # Verify parameters were enriched with context
        assert parameters is not None
        assert parameters.intensity > 0.9  # Should be high due to important match in late game
    
    def test_get_musical_parameters(self, trigger: MusicTrigger) -> None:
        """Test getting musical parameters for event type, emotion, and intensity."""
        # Set up initial state
        trigger.set_base_musical_state(100.0, "C Major")
        
        # Get parameters
        params_dict = trigger.get_musical_parameters("goal", "excitement", 80.0)
        
        # Verify parameters
        assert params_dict is not None
        assert isinstance(params_dict, dict)
        assert params_dict["intensity"] == 0.8  # 80.0 / 100.0
        assert params_dict["mood"] == "triumphant"
        assert "brass" in params_dict["instrumentation"]
        assert params_dict["tempo"] == 120.0  # 100 + 20 BPM increase for goal
    
    def test_process_significant_events(
        self, trigger: MusicTrigger, events_list: list[MatchEvent]
    ) -> None:
        """Test processing multiple events and filtering for significant ones."""
        # Process events
        parameters_list = trigger.process_significant_events(events_list)
        
        # Verify significant events were processed
        assert len(parameters_list) > 0
        
        # Verify goal event was processed (high intensity)
        assert any(p.mood == "triumphant" for p in parameters_list)
        
        # Verify red card event was processed (high intensity)
        assert any(p.mood == "dramatic" for p in parameters_list)
        
        # Verify near miss event was processed (medium intensity)
        assert any(p.mood == "tense" for p in parameters_list)
    
    def test_process_significant_events_empty(self, trigger: MusicTrigger) -> None:
        """Test processing an empty events list."""
        parameters_list = trigger.process_significant_events([])
        assert parameters_list == []
    
    def test_set_user_preferences(self, trigger: MusicTrigger, user_preferences: UserPreferences) -> None:
        """Test setting user preferences."""
        trigger.set_user_preferences(user_preferences)
        assert trigger.user_preferences == user_preferences
        assert trigger.emotion_mapper.cultural_style == "european"
    
    def test_process_emotion(self, trigger: MusicTrigger, excitement_emotion: CrowdEmotion) -> None:
        """Test processing a crowd emotion."""
        # Set up callback
        music_callback = MagicMock()
        trigger.set_music_engine_callback(music_callback)
        
        # Process emotion
        parameters = trigger.process_emotion(excitement_emotion)
        
        # Verify parameters
        assert parameters is not None
        assert isinstance(parameters, MusicalParameters)
        assert parameters.mood == "energetic"
        assert "brass" in parameters.instrumentation
        assert "percussion" in parameters.instrumentation
        assert parameters.tempo > trigger.emotion_mapper.base_tempo  # Tempo should increase for excitement
        
        # Verify callback was called
        music_callback.assert_called_once()
        
        # Verify current parameters were updated
        assert trigger.current_parameters == parameters
        assert trigger.last_emotion_time == excitement_emotion.timestamp
    
    def test_process_emotion_cooldown(
        self, trigger: MusicTrigger, excitement_emotion: CrowdEmotion, disappointment_emotion: CrowdEmotion
    ) -> None:
        """Test emotion cooldown prevents processing emotions too frequently."""
        # Set up initial state
        trigger.last_emotion_time = excitement_emotion.timestamp
        trigger.current_parameters = MusicalParameters(
            tempo=120.0,
            key="C Major",
            intensity=0.8,
            instrumentation=["brass", "percussion", "strings"],
            mood="energetic"
        )
        
        # Set up callback
        music_callback = MagicMock()
        trigger.set_music_engine_callback(music_callback)
        
        # Set disappointment timestamp to be within cooldown period
        disappointment_emotion.timestamp = excitement_emotion.timestamp + 1.0  # Less than cooldown
        
        # Process emotion
        parameters = trigger.process_emotion(disappointment_emotion)
        
        # Verify parameters are unchanged
        assert parameters == trigger.current_parameters
        
        # Verify callback was not called
        music_callback.assert_not_called()
        
        # Verify last emotion time was not updated
        assert trigger.last_emotion_time == excitement_emotion.timestamp
    
    def test_process_emotion_with_preferences(
        self, trigger: MusicTrigger, excitement_emotion: CrowdEmotion, user_preferences: UserPreferences
    ) -> None:
        """Test processing an emotion with user preferences."""
        # Set user preferences
        trigger.set_user_preferences(user_preferences)
        
        # Process emotion
        parameters = trigger.process_emotion(excitement_emotion)
        
        # Verify parameters were adjusted based on preferences
        assert parameters is not None
        assert "woodwinds" in parameters.instrumentation  # Added by orchestral preference
        assert parameters.intensity > 0.7  # High due to excitement and preference level 4
    
    def test_blend_event_and_emotion(
        self, trigger: MusicTrigger, goal_event: MatchEvent, excitement_emotion: CrowdEmotion
    ) -> None:
        """Test blending event and emotion parameters."""
        # Set up callback
        music_callback = MagicMock()
        trigger.set_music_engine_callback(music_callback)
        
        # Blend event and emotion
        parameters = trigger.blend_event_and_emotion(goal_event, excitement_emotion)
        
        # Verify parameters
        assert parameters is not None
        assert isinstance(parameters, MusicalParameters)
        assert parameters.intensity > 0.8  # High intensity from both goal and excitement
        assert "brass" in parameters.instrumentation
        assert "percussion" in parameters.instrumentation
        assert parameters.tempo > trigger.event_mapper.base_tempo  # Tempo should increase
        
        # Verify callback was called
        music_callback.assert_called_once()
        
        # Verify current parameters were updated
        assert trigger.current_parameters == parameters
    
    def test_scale_intensity(self, trigger: MusicTrigger) -> None:
        """Test scaling parameters with intensity."""
        # Set up initial state
        trigger.current_parameters = MusicalParameters(
            tempo=100.0,
            key="C Major",
            intensity=0.5,
            instrumentation=["strings", "piano"],
            mood="neutral",
            transition_duration=5.0
        )
        
        # Set up callback
        music_callback = MagicMock()
        trigger.set_music_engine_callback(music_callback)
        
        # Scale intensity to high level
        high_intensity = 90.0
        high_params = trigger.scale_intensity(high_intensity)
        
        # Verify parameters
        assert high_params.tempo > trigger.current_parameters.tempo
        assert high_params.intensity > trigger.current_parameters.intensity
        assert "percussion" in high_params.instrumentation
        assert high_params.transition_duration < trigger.current_parameters.transition_duration
        
        # Verify callback was called
        music_callback.assert_called_once()
    
    @patch("crowd_sentiment_music_generator.services.music_trigger.music_trigger.with_error_handling")
    def test_error_handling(self, mock_error_handler: MagicMock, trigger: MusicTrigger) -> None:
        """Test that error handling decorator is applied to public methods."""
        # Configure the mock to pass through the original function
        mock_error_handler.side_effect = lambda f: f
        
        # Verify error handling is applied to public methods
        assert hasattr(trigger.process_event, "__wrapped__")
        assert hasattr(trigger.process_emotion, "__wrapped__")
        assert hasattr(trigger.blend_event_and_emotion, "__wrapped__")
        assert hasattr(trigger.get_musical_parameters, "__wrapped__")
        assert hasattr(trigger.process_significant_events, "__wrapped__")
        assert hasattr(trigger.scale_intensity, "__wrapped__")