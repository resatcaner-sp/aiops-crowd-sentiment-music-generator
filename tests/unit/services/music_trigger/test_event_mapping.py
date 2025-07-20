"""Unit tests for event mapping module."""

import pytest
from unittest.mock import MagicMock, patch

from crowd_sentiment_music_generator.exceptions.music_generation_error import MusicGenerationError
from crowd_sentiment_music_generator.models.data.match_event import MatchEvent
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.models.music.musical_parameters import MusicalParameters
from crowd_sentiment_music_generator.services.music_trigger.event_mapping import EventMusicMapper


class TestEventMusicMapper:
    """Test cases for EventMusicMapper class."""
    
    @pytest.fixture
    def mapper(self) -> EventMusicMapper:
        """Create an EventMusicMapper instance for testing."""
        config = SystemConfig()
        return EventMusicMapper(config)
    
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
    def penalty_event(self) -> MatchEvent:
        """Create a penalty event for testing."""
        return MatchEvent(
            id="test_penalty",
            type="penalty_awarded",
            timestamp=2345.6,
            team_id="away_team"
        )
    
    @pytest.fixture
    def unknown_event(self) -> MatchEvent:
        """Create an unknown event type for testing."""
        return MatchEvent(
            id="test_unknown",
            type="unknown_event_type",
            timestamp=3456.7,
            team_id="home_team"
        )
    
    def test_get_event_response_known_event(self, mapper: EventMusicMapper) -> None:
        """Test getting response for a known event type."""
        response = mapper.get_event_response("goal")
        
        assert response is not None
        assert response["immediate"] == "cymbal_crash + brass_fanfare"
        assert response["intensity"] == 1.0
        assert "instrumentation" in response
        assert "mood" in response
    
    def test_get_event_response_unknown_event(self, mapper: EventMusicMapper) -> None:
        """Test getting response for an unknown event type."""
        response = mapper.get_event_response("unknown_event_type")
        
        assert response is not None
        assert response["immediate"] == "subtle_accent"
        assert response["intensity"] == 0.3
        assert "instrumentation" in response
        assert "mood" in response
    
    def test_map_event_to_parameters_goal(
        self, mapper: EventMusicMapper, goal_event: MatchEvent
    ) -> None:
        """Test mapping a goal event to musical parameters."""
        parameters = mapper.map_event_to_parameters(goal_event)
        
        assert parameters is not None
        assert isinstance(parameters, MusicalParameters)
        assert parameters.intensity == 1.0
        assert parameters.mood == "triumphant"
        assert "brass" in parameters.instrumentation
        assert "percussion" in parameters.instrumentation
        assert parameters.tempo == mapper.base_tempo + 20  # BPM increase for goal
    
    def test_map_event_to_parameters_with_current(
        self, mapper: EventMusicMapper, penalty_event: MatchEvent
    ) -> None:
        """Test mapping an event with current parameters."""
        current_parameters = MusicalParameters(
            tempo=120.0,
            key="D Major",
            intensity=0.5,
            instrumentation=["piano", "strings"],
            mood="neutral"
        )
        
        parameters = mapper.map_event_to_parameters(penalty_event, current_parameters)
        
        assert parameters is not None
        assert isinstance(parameters, MusicalParameters)
        assert parameters.intensity == 0.7
        assert parameters.mood == "suspenseful"
        assert "percussion" in parameters.instrumentation
        assert "strings" in parameters.instrumentation
        assert parameters.tempo == 115.0  # 120 - 5 BPM decrease for penalty
    
    def test_map_event_to_parameters_unknown(
        self, mapper: EventMusicMapper, unknown_event: MatchEvent
    ) -> None:
        """Test mapping an unknown event type."""
        parameters = mapper.map_event_to_parameters(unknown_event)
        
        assert parameters is not None
        assert isinstance(parameters, MusicalParameters)
        assert parameters.intensity == 0.3
        assert parameters.mood == "neutral"
        assert "strings" in parameters.instrumentation
        assert parameters.tempo == mapper.base_tempo  # No change for unknown event
    
    def test_get_immediate_trigger(
        self, mapper: EventMusicMapper, goal_event: MatchEvent
    ) -> None:
        """Test getting immediate trigger for an event."""
        trigger = mapper.get_immediate_trigger(goal_event)
        
        assert trigger is not None
        assert trigger["type"] == "cymbal_crash + brass_fanfare"
        assert trigger["intensity"] == 1.0
        assert trigger["duration"] > 0
    
    def test_enrich_parameters_with_context(self, mapper: EventMusicMapper) -> None:
        """Test enriching parameters with match context."""
        parameters = MusicalParameters(
            tempo=100.0,
            key="C Major",
            intensity=0.6,
            instrumentation=["strings", "piano"],
            mood="neutral",
            transition_duration=5.0
        )
        
        match_context = {
            "score_difference": 1,
            "time_remaining": 5,
            "match_importance": 0.8
        }
        
        enriched = mapper.enrich_parameters_with_context(parameters, match_context)
        
        assert enriched is not None
        assert enriched.intensity > parameters.intensity  # Increased for important match in late game
        assert enriched.tempo > parameters.tempo  # Increased for close game in late stages
        assert enriched.transition_duration < parameters.transition_duration  # Faster transitions in late game
    
    def test_select_key(self, mapper: EventMusicMapper) -> None:
        """Test key selection based on preference."""
        major_key = mapper._select_key("major")
        assert major_key in mapper.KEY_MAPPING["major"]
        
        minor_key = mapper._select_key("minor")
        assert minor_key in mapper.KEY_MAPPING["minor"]
        
        diminished_key = mapper._select_key("diminished")
        assert diminished_key in mapper.KEY_MAPPING["diminished"]
        
        neutral_key = mapper._select_key("neutral")
        assert neutral_key in mapper.KEY_MAPPING["neutral"]
        
        unknown_key = mapper._select_key("unknown_preference")
        assert unknown_key in mapper.KEY_MAPPING["neutral"]  # Default to neutral
    
    @patch("crowd_sentiment_music_generator.services.music_trigger.event_mapping.with_error_handling")
    def test_error_handling(self, mock_error_handler: MagicMock, mapper: EventMusicMapper) -> None:
        """Test that error handling decorator is applied to public methods."""
        # Configure the mock to pass through the original function
        mock_error_handler.side_effect = lambda f: f
        
        # Verify error handling is applied to public methods
        assert hasattr(mapper.get_event_response, "__wrapped__")
        assert hasattr(mapper.map_event_to_parameters, "__wrapped__")
        assert hasattr(mapper.get_immediate_trigger, "__wrapped__")
        assert hasattr(mapper.enrich_parameters_with_context, "__wrapped__")