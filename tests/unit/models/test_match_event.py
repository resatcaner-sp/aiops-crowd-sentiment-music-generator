"""Tests for the MatchEvent model."""

import pytest
from pydantic import ValidationError

from crowd_sentiment_music_generator.models.data.match_event import MatchEvent


def test_match_event_creation() -> None:
    """Test creating a valid MatchEvent."""
    event = MatchEvent(
        id="123",
        type="goal",
        timestamp=1625097600.0,
        team_id="team1",
    )
    
    assert event.id == "123"
    assert event.type == "goal"
    assert event.timestamp == 1625097600.0
    assert event.team_id == "team1"
    assert event.player_id is None
    assert event.position is None
    assert event.additional_data is None


def test_match_event_with_optional_fields() -> None:
    """Test creating a MatchEvent with optional fields."""
    event = MatchEvent(
        id="123",
        type="goal",
        timestamp=1625097600.0,
        team_id="team1",
        player_id="player1",
        position={"x": 10.5, "y": 20.3},
        additional_data={"speed": 25.6, "angle": 45.0},
    )
    
    assert event.id == "123"
    assert event.type == "goal"
    assert event.timestamp == 1625097600.0
    assert event.team_id == "team1"
    assert event.player_id == "player1"
    assert event.position == {"x": 10.5, "y": 20.3}
    assert event.additional_data == {"speed": 25.6, "angle": 45.0}


def test_match_event_missing_required_fields() -> None:
    """Test that creating a MatchEvent with missing required fields raises an error."""
    with pytest.raises(ValidationError):
        MatchEvent(
            id="123",
            type="goal",
            # Missing timestamp
            team_id="team1",
        )
    
    with pytest.raises(ValidationError):
        MatchEvent(
            id="123",
            # Missing type
            timestamp=1625097600.0,
            team_id="team1",
        )