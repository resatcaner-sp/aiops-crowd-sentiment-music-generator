"""Unit tests for override router."""

import pytest
from fastapi.testclient import TestClient

from crowd_sentiment_music_generator.api.app import create_app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    app = create_app()
    return TestClient(app)


def test_get_override_options(client):
    """Test getting override options."""
    # Send request
    response = client.get("/override/music")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "id" in data[0]
    assert "match_id" in data[0]
    assert "name" in data[0]
    assert "audio_url" in data[0]
    assert "duration" in data[0]
    assert "tags" in data[0]


def test_get_override_options_with_match_filter(client):
    """Test getting override options filtered by match ID."""
    # Send request
    response = client.get("/override/music?match_id=match-123")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert all(override["match_id"] == "match-123" or override["match_id"] == "" for override in data)


def test_get_override_options_with_tag_filter(client):
    """Test getting override options filtered by tag."""
    # Send request
    response = client.get("/override/music?tag=celebration")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert all("celebration" in override["tags"] for override in data)


def test_create_override_option(client):
    """Test creating override option."""
    # Prepare request data
    override_data = {
        "id": "override-new",
        "match_id": "match-789",
        "name": "Victory Theme",
        "description": "Triumphant music for victory celebrations",
        "audio_url": "https://example.com/overrides/victory-theme.mp3",
        "duration": 60.0,
        "tags": ["victory", "celebration", "orchestral"],
        "created_at": 1626915000.0,
    }
    
    # Send request
    response = client.post("/override/music", json=override_data)
    
    # Check response
    assert response.status_code == 201
    data = response.json()
    assert data["id"] == "override-new"
    assert data["match_id"] == "match-789"
    assert data["name"] == "Victory Theme"
    assert data["audio_url"] == "https://example.com/overrides/victory-theme.mp3"
    assert data["duration"] == 60.0
    assert data["tags"] == ["victory", "celebration", "orchestral"]


def test_apply_override(client):
    """Test applying override."""
    # Prepare request data
    request_data = {
        "match_id": "match-123",
        "segment_id": "segment-456",
        "override_id": "override-1",
        "reason": "Inappropriate music detected",
        "duration": 30.0,
        "fade_in": 0.8,
        "fade_out": 1.2,
    }
    
    # Send request
    response = client.post("/override/apply", json=request_data)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["match_id"] == "match-123"
    assert data["segment_id"] == "segment-456"
    assert data["override_id"] == "override-1"
    assert data["status"] == "applied"
    assert "applied_at" in data
    assert "expires_at" in data


def test_apply_override_without_duration(client):
    """Test applying override without duration."""
    # Prepare request data
    request_data = {
        "match_id": "match-123",
        "segment_id": "segment-456",
        "override_id": "override-1",
        "reason": "Inappropriate music detected",
    }
    
    # Send request
    response = client.post("/override/apply", json=request_data)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["match_id"] == "match-123"
    assert data["segment_id"] == "segment-456"
    assert data["override_id"] == "override-1"
    assert data["status"] == "applied"
    assert "applied_at" in data
    assert data["expires_at"] is None  # No expiration when duration is not provided


def test_get_active_overrides(client):
    """Test getting active overrides."""
    # Send request
    response = client.get("/override/active")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "request_id" in data[0]
    assert "match_id" in data[0]
    assert "override_id" in data[0]
    assert "status" in data[0]
    assert "applied_at" in data[0]


def test_get_active_overrides_with_match_filter(client):
    """Test getting active overrides filtered by match ID."""
    # Send request
    response = client.get("/override/active?match_id=match-123")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert all(override["match_id"] == "match-123" for override in data)


def test_cancel_override(client):
    """Test canceling override."""
    # Send request
    response = client.delete("/override/cancel/request-1")
    
    # Check response
    assert response.status_code == 204