"""Tests for the health check endpoint."""

import pytest
from fastapi.testclient import TestClient

from crowd_sentiment_music_generator.api.app import create_app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI application.
    
    Returns:
        TestClient: The test client
    """
    app = create_app()
    return TestClient(app)


def test_health_check(client: TestClient) -> None:
    """Test the health check endpoint.
    
    Args:
        client: The test client
    """
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}