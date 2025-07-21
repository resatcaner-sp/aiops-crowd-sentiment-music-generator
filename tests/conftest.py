"""Pytest configuration file."""

import os
import pytest
from fastapi.testclient import TestClient

from crowd_sentiment_music_generator.api.app import create_app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    app = create_app()
    return TestClient(app)


def pytest_configure(config):
    """Configure pytest."""
    # Register custom markers
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    
    # Set environment variables for testing
    os.environ["APP_ENVIRONMENT"] = "test"


def pytest_collection_modifyitems(config, items):
    """Modify collected test items."""
    # Skip integration tests unless explicitly requested
    if not config.getoption("--integration"):
        skip_integration = pytest.mark.skip(reason="need --integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
    
    # Skip performance tests unless explicitly requested
    if not config.getoption("--performance"):
        skip_performance = pytest.mark.skip(reason="need --performance option to run")
        for item in items:
            if "performance" in item.keywords:
                item.add_marker(skip_performance)


def pytest_addoption(parser):
    """Add command line options to pytest."""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )
    parser.addoption(
        "--performance",
        action="store_true",
        default=False,
        help="run performance tests",
    )