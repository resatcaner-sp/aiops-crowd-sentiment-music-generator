"""Unit tests for preferences router."""

import pytest
from fastapi.testclient import TestClient

from crowd_sentiment_music_generator.api.app import create_app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    app = create_app()
    return TestClient(app)


def test_get_user_preferences(client):
    """Test getting user preferences."""
    # Send request
    response = client.get("/preferences/users/user-123")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "user-123"
    assert "music_enabled" in data
    assert "music_intensity" in data
    assert "preferred_genres" in data
    assert "cultural_style" in data
    assert "team_preferences" in data


def test_get_user_preferences_default(client):
    """Test getting default user preferences for new user."""
    # Send request
    response = client.get("/preferences/users/new-user")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "new-user"
    assert data["music_enabled"] is True
    assert data["music_intensity"] == 3
    assert data["preferred_genres"] == ["orchestral"]
    assert data["cultural_style"] is None
    assert data["team_preferences"] == {}


def test_update_user_preferences(client):
    """Test updating user preferences."""
    # Prepare request data
    preferences_data = {
        "user_id": "user-123",
        "music_enabled": False,
        "music_intensity": 2,
        "preferred_genres": ["electronic", "ambient"],
        "cultural_style": "asian",
        "team_preferences": {},
    }
    
    # Send request
    response = client.put("/preferences/users/user-123", json=preferences_data)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "user-123"
    assert data["music_enabled"] is False
    assert data["music_intensity"] == 2
    assert data["preferred_genres"] == ["electronic", "ambient"]
    assert data["cultural_style"] == "asian"


def test_update_user_preferences_mismatched_id(client):
    """Test updating user preferences with mismatched user ID."""
    # Prepare request data
    preferences_data = {
        "user_id": "wrong-user",
        "music_enabled": False,
        "music_intensity": 2,
        "preferred_genres": ["electronic", "ambient"],
        "cultural_style": "asian",
        "team_preferences": {},
    }
    
    # Send request
    response = client.put("/preferences/users/user-123", json=preferences_data)
    
    # Check response
    assert response.status_code == 400
    assert "user id" in response.json()["detail"].lower()


def test_add_team_preference(client):
    """Test adding team preference."""
    # Prepare request data
    team_preference = {
        "team_id": "team-123",
        "team_name": "Manchester United",
        "musical_style": "orchestral",
        "intensity": 5,
        "theme_song": "manchester_anthem",
    }
    
    # Send request
    response = client.post("/preferences/users/user-123/teams", json=team_preference)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["team_id"] == "team-123"
    assert data["team_name"] == "Manchester United"
    assert data["musical_style"] == "orchestral"
    assert data["intensity"] == 5
    assert data["theme_song"] == "manchester_anthem"


def test_delete_team_preference(client):
    """Test deleting team preference."""
    # Send request
    response = client.delete("/preferences/users/user-123/teams/team-456")
    
    # Check response
    assert response.status_code == 204


def test_get_available_genres(client):
    """Test getting available music genres."""
    # Send request
    response = client.get("/preferences/genres")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "orchestral" in data
    assert "electronic" in data
    assert "cinematic" in data


def test_get_available_cultural_styles(client):
    """Test getting available cultural styles."""
    # Send request
    response = client.get("/preferences/cultural-styles")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "global" in data
    assert "european" in data
    assert "latin" in data


def test_get_preference_sync_status(client):
    """Test getting preference synchronization status."""
    # Send request
    response = client.get("/preferences/sync-status/user-123")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "synced" in data
    assert "last_sync" in data
    assert "devices" in data