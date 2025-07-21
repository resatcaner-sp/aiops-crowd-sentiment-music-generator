"""Unit tests for audio analysis router."""

import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from crowd_sentiment_music_generator.api.app import create_app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    app = create_app()
    return TestClient(app)


def test_analyze_audio(client):
    """Test analyzing audio file."""
    # Create a mock audio file
    audio_file = io.BytesIO(b"mock audio data")
    audio_file.name = "test.wav"
    
    # Send request
    response = client.post(
        "/audio/analyze",
        files={"file": ("test.wav", audio_file, "audio/wav")},
        data={
            "match_id": "match-123",
            "segment_id": "segment-456",
            "start_time": 0.0,
            "end_time": 30.0,
        },
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["match_id"] == "match-123"
    assert data["segment_id"] == "segment-456"
    assert data["start_time"] == 0.0
    assert data["end_time"] == 30.0
    assert "emotions" in data
    assert len(data["emotions"]) > 0
    assert "dominant_emotion" in data
    assert "average_intensity" in data


def test_analyze_audio_without_optional_params(client):
    """Test analyzing audio file without optional parameters."""
    # Create a mock audio file
    audio_file = io.BytesIO(b"mock audio data")
    audio_file.name = "test.wav"
    
    # Send request
    response = client.post(
        "/audio/analyze",
        files={"file": ("test.wav", audio_file, "audio/wav")},
        data={"match_id": "match-123"},
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["match_id"] == "match-123"
    assert "segment_id" in data  # Should be auto-generated
    assert "start_time" in data
    assert "end_time" in data
    assert "emotions" in data


def test_get_audio_streams(client):
    """Test getting audio streams."""
    # Send request
    response = client.get("/audio/streams")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "match_id" in data[0]
    assert "stream_url" in data[0]
    assert "stream_type" in data[0]


def test_get_audio_streams_with_match_filter(client):
    """Test getting audio streams filtered by match ID."""
    # Send request
    response = client.get("/audio/streams?match_id=match-123")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert all(stream["match_id"] == "match-123" for stream in data)


def test_add_audio_stream(client):
    """Test adding audio stream."""
    # Prepare request data
    stream_data = {
        "match_id": "match-789",
        "stream_url": "rtmp://example.com/live/match-789-broadcast",
        "stream_type": "broadcast",
        "enabled": True,
        "isolation_level": 0.8,
    }
    
    # Send request
    response = client.post("/audio/streams", json=stream_data)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["match_id"] == "match-789"
    assert data["stream_url"] == "rtmp://example.com/live/match-789-broadcast"
    assert data["stream_type"] == "broadcast"
    assert data["enabled"] is True
    assert data["isolation_level"] == 0.8


def test_delete_audio_stream(client):
    """Test deleting audio stream."""
    # Send request
    response = client.delete("/audio/streams/stream-123")
    
    # Check response
    assert response.status_code == 204


def test_get_match_emotions(client):
    """Test getting match emotions."""
    # Send request
    response = client.get("/audio/emotions/match-123")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "emotion" in data[0]
    assert "intensity" in data[0]
    assert "confidence" in data[0]
    assert "timestamp" in data[0]


def test_get_match_emotions_with_time_filter(client):
    """Test getting match emotions filtered by time range."""
    # Send request
    response = client.get("/audio/emotions/match-123?start_time=30&end_time=120")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert all(30 <= emotion["timestamp"] <= 120 for emotion in data)