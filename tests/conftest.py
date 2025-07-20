"""Common test fixtures for the crowd sentiment music generator."""

import pytest
from fastapi.testclient import TestClient

from crowd_sentiment_music_generator.api.app import create_app
from crowd_sentiment_music_generator.models.data.match_event import MatchEvent
from crowd_sentiment_music_generator.models.data.crowd_emotion import CrowdEmotion
from crowd_sentiment_music_generator.models.music.musical_parameters import MusicalParameters


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI application.
    
    Returns:
        TestClient: The test client
    """
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_match_event() -> MatchEvent:
    """Create a sample match event.
    
    Returns:
        MatchEvent: A sample match event
    """
    return MatchEvent(
        id="123",
        type="goal",
        timestamp=1625097600.0,
        team_id="team1",
        player_id="player1",
        position={"x": 10.5, "y": 20.3},
        additional_data={"speed": 25.6, "angle": 45.0},
    )


@pytest.fixture
def sample_crowd_emotion() -> CrowdEmotion:
    """Create a sample crowd emotion.
    
    Returns:
        CrowdEmotion: A sample crowd emotion
    """
    return CrowdEmotion(
        emotion="excitement",
        intensity=85.5,
        confidence=0.92,
        timestamp=1625097600.0,
        audio_features={
            "rms_energy": 0.75,
            "spectral_centroid": 3500.0,
            "zero_crossing_rate": 0.15,
            "tempo": 120.0,
        },
    )


@pytest.fixture
def sample_musical_parameters() -> MusicalParameters:
    """Create a sample musical parameters.
    
    Returns:
        MusicalParameters: Sample musical parameters
    """
    return MusicalParameters(
        tempo=120.0,
        key="C Major",
        intensity=0.75,
        instrumentation=["piano", "strings", "percussion"],
        mood="excitement",
        transition_duration=3.5,
    )