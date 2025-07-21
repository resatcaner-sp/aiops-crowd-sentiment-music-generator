"""Integration tests for API endpoints."""

import io
import pytest
from fastapi.testclient import TestClient

from crowd_sentiment_music_generator.api.app import create_app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    app = create_app()
    return TestClient(app)


@pytest.mark.integration
def test_health_endpoint(client):
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.integration
def test_audio_analysis_flow(client):
    """Test the audio analysis flow."""
    # Step 1: Upload audio for analysis
    audio_file = io.BytesIO(b"mock audio data")
    audio_file.name = "test.wav"
    
    response = client.post(
        "/audio/analyze",
        files={"file": ("test.wav", audio_file, "audio/wav")},
        data={"match_id": "match-integration"},
    )
    
    assert response.status_code == 200
    analysis_result = response.json()
    assert analysis_result["match_id"] == "match-integration"
    assert "emotions" in analysis_result
    
    # Step 2: Get emotions for the match
    response = client.get(f"/audio/emotions/{analysis_result['match_id']}")
    assert response.status_code == 200
    emotions = response.json()
    assert isinstance(emotions, list)
    assert len(emotions) > 0


@pytest.mark.integration
def test_highlight_processing_flow(client):
    """Test the highlight processing flow."""
    # Step 1: Create a highlight
    highlight_data = {
        "id": "highlight-integration",
        "match_id": "match-integration",
        "title": "Integration Test Highlight",
        "description": "A highlight for integration testing",
        "start_time": 1800.0,
        "end_time": 1830.0,
        "key_moment_time": 1815.0,
        "video_url": "https://example.com/highlights/integration.mp4",
        "thumbnail_url": "https://example.com/thumbnails/integration.jpg",
        "created_at": 1626912000.0,
    }
    
    response = client.post("/highlights", json=highlight_data)
    assert response.status_code == 201
    created_highlight = response.json()
    assert created_highlight["id"] == "highlight-integration"
    
    # Step 2: Generate music for the highlight
    music_settings = {
        "intensity": 0.8,
        "style": "cinematic",
        "mood": "dramatic",
        "transition_speed": 1.2,
        "use_team_themes": True,
    }
    
    response = client.post(f"/highlights/{created_highlight['id']}/music", json=music_settings)
    assert response.status_code == 200
    music_result = response.json()
    assert music_result["highlight"]["id"] == created_highlight["id"]
    assert music_result["status"] == "processing"
    
    # Step 3: Export the highlight with music
    export_settings = {
        "format": "mp4",
        "quality": "high",
        "include_commentary": True,
        "music_volume": 0.7,
        "add_watermark": False,
    }
    
    response = client.post(f"/highlights/{created_highlight['id']}/export", json=export_settings)
    assert response.status_code == 200
    export_result = response.json()
    assert export_result["highlight"]["id"] == created_highlight["id"]
    assert export_result["status"] == "completed"
    assert "export_url" in export_result


@pytest.mark.integration
def test_user_preferences_flow(client):
    """Test the user preferences flow."""
    # Step 1: Get default preferences for a new user
    user_id = "user-integration"
    response = client.get(f"/preferences/users/{user_id}")
    assert response.status_code == 200
    default_preferences = response.json()
    assert default_preferences["user_id"] == user_id
    
    # Step 2: Update user preferences
    updated_preferences = default_preferences.copy()
    updated_preferences["music_intensity"] = 4
    updated_preferences["preferred_genres"] = ["electronic", "cinematic"]
    updated_preferences["cultural_style"] = "latin"
    
    response = client.put(f"/preferences/users/{user_id}", json=updated_preferences)
    assert response.status_code == 200
    result = response.json()
    assert result["music_intensity"] == 4
    assert result["preferred_genres"] == ["electronic", "cinematic"]
    assert result["cultural_style"] == "latin"
    
    # Step 3: Add team preference
    team_preference = {
        "team_id": "team-integration",
        "team_name": "Integration FC",
        "musical_style": "electronic",
        "intensity": 5,
        "theme_song": "integration_anthem",
    }
    
    response = client.post(f"/preferences/users/{user_id}/teams", json=team_preference)
    assert response.status_code == 200
    result = response.json()
    assert result["team_id"] == "team-integration"
    assert result["team_name"] == "Integration FC"


@pytest.mark.integration
def test_override_flow(client):
    """Test the override flow."""
    # Step 1: Create an override option
    override_data = {
        "id": "override-integration",
        "match_id": "match-integration",
        "name": "Integration Override",
        "description": "An override for integration testing",
        "audio_url": "https://example.com/overrides/integration.mp3",
        "duration": 30.0,
        "tags": ["integration", "test"],
        "created_at": 1626912000.0,
    }
    
    response = client.post("/override/music", json=override_data)
    assert response.status_code == 201
    created_override = response.json()
    assert created_override["id"] == "override-integration"
    
    # Step 2: Apply the override
    apply_data = {
        "match_id": "match-integration",
        "segment_id": "segment-integration",
        "override_id": created_override["id"],
        "reason": "Integration testing",
        "duration": 30.0,
    }
    
    response = client.post("/override/apply", json=apply_data)
    assert response.status_code == 200
    apply_result = response.json()
    assert apply_result["match_id"] == "match-integration"
    assert apply_result["override_id"] == created_override["id"]
    assert apply_result["status"] == "applied"
    
    # Step 3: Get active overrides
    response = client.get("/override/active")
    assert response.status_code == 200
    active_overrides = response.json()
    assert isinstance(active_overrides, list)


@pytest.mark.integration
def test_scaling_flow(client):
    """Test the scaling flow."""
    # Step 1: Get scaling status
    response = client.get("/scaling/status")
    assert response.status_code == 200
    status = response.json()
    assert "current_instances" in status
    assert "min_instances" in status
    assert "max_instances" in status
    
    # Step 2: Get scaling decision
    response = client.get("/scaling/decision")
    assert response.status_code == 200
    decision = response.json()
    assert "should_scale" in decision
    assert "current_instances" in decision
    assert "target_instances" in decision
    
    # Step 3: Get scaling configuration
    response = client.get("/scaling/config")
    assert response.status_code == 200
    config = response.json()
    assert "enabled" in config
    assert "min_instances" in config
    assert "max_instances" in config
    
    # Step 4: Add match priority
    priority_data = {
        "match_id": "match-integration",
        "priority": 7,
        "resource_allocation": 25.0,
    }
    
    response = client.post("/scaling/priorities", json=priority_data)
    assert response.status_code == 200
    result = response.json()
    assert result["match_id"] == "match-integration"
    assert result["priority"] == 7
    assert result["resource_allocation"] == 25.0