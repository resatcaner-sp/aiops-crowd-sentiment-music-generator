"""Unit tests for highlights router."""

import pytest
from fastapi.testclient import TestClient

from crowd_sentiment_music_generator.api.app import create_app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    app = create_app()
    return TestClient(app)


def test_get_highlights(client):
    """Test getting highlights."""
    # Send request
    response = client.get("/highlights")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "id" in data[0]
    assert "match_id" in data[0]
    assert "title" in data[0]
    assert "start_time" in data[0]
    assert "end_time" in data[0]
    assert "key_moment_time" in data[0]


def test_get_highlights_with_match_filter(client):
    """Test getting highlights filtered by match ID."""
    # Send request
    response = client.get("/highlights?match_id=match-123")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert all(highlight["match_id"] == "match-123" for highlight in data)


def test_create_highlight(client):
    """Test creating highlight."""
    # Prepare request data
    highlight_data = {
        "id": "highlight-new",
        "match_id": "match-789",
        "title": "Amazing Save",
        "description": "Goalkeeper makes an incredible save",
        "start_time": 4500.0,
        "end_time": 4530.0,
        "key_moment_time": 4515.0,
        "video_url": "https://example.com/highlights/highlight-new.mp4",
        "thumbnail_url": "https://example.com/thumbnails/highlight-new.jpg",
        "created_at": 1626914000.0,
    }
    
    # Send request
    response = client.post("/highlights", json=highlight_data)
    
    # Check response
    assert response.status_code == 201
    data = response.json()
    assert data["id"] == "highlight-new"
    assert data["match_id"] == "match-789"
    assert data["title"] == "Amazing Save"
    assert data["start_time"] == 4500.0
    assert data["end_time"] == 4530.0
    assert data["key_moment_time"] == 4515.0


def test_get_highlight(client):
    """Test getting highlight by ID."""
    # Send request
    response = client.get("/highlights/highlight-1")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "highlight-1"
    assert data["match_id"] == "match-123"
    assert "title" in data
    assert "start_time" in data
    assert "end_time" in data
    assert "key_moment_time" in data


def test_get_highlight_not_found(client):
    """Test getting non-existent highlight."""
    # Send request
    response = client.get("/highlights/non-existent")
    
    # Check response
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_generate_music_for_highlight(client):
    """Test generating music for highlight."""
    # Prepare request data
    music_settings = {
        "intensity": 0.8,
        "style": "cinematic",
        "mood": "dramatic",
        "transition_speed": 1.2,
        "use_team_themes": True,
    }
    
    # Send request
    response = client.post("/highlights/highlight-1/music", json=music_settings)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["highlight"]["id"] == "highlight-1"
    assert data["music_settings"]["intensity"] == 0.8
    assert data["music_settings"]["style"] == "cinematic"
    assert data["music_settings"]["mood"] == "dramatic"
    assert data["music_settings"]["transition_speed"] == 1.2
    assert data["music_settings"]["use_team_themes"] is True
    assert data["status"] == "processing"


def test_get_highlight_music(client):
    """Test getting music for highlight."""
    # Send request
    response = client.get("/highlights/highlight-1/music")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["highlight"]["id"] == "highlight-1"
    assert "music_settings" in data
    assert "export_settings" in data
    assert "export_url" in data
    assert data["status"] == "completed"


def test_export_highlight(client):
    """Test exporting highlight with music."""
    # Prepare request data
    export_settings = {
        "format": "mp3",
        "quality": "high",
        "include_commentary": False,
        "music_volume": 0.9,
        "add_watermark": True,
    }
    
    # Send request
    response = client.post("/highlights/highlight-1/export", json=export_settings)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["highlight"]["id"] == "highlight-1"
    assert data["export_settings"]["format"] == "mp3"
    assert data["export_settings"]["quality"] == "high"
    assert data["export_settings"]["include_commentary"] is False
    assert data["export_settings"]["music_volume"] == 0.9
    assert data["export_settings"]["add_watermark"] is True
    assert "export_url" in data
    assert data["status"] == "completed"


def test_trim_highlight(client):
    """Test trimming highlight."""
    # Prepare form data
    form_data = {
        "start_time": 1810.0,
        "end_time": 1825.0,
        "key_moment_time": 1815.0,
    }
    
    # Send request
    response = client.put("/highlights/highlight-1/trim", data=form_data)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "highlight-1"
    assert data["start_time"] == 1810.0
    assert data["end_time"] == 1825.0
    assert data["key_moment_time"] == 1815.0