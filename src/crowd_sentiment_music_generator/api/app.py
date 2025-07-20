"""FastAPI application for crowd sentiment music generator."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from crowd_sentiment_music_generator.api.routers import health, music, events
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for the FastAPI application.
    
    Args:
        app: The FastAPI application
    """
    # Startup
    logger.info("Starting up the application")
    config = SystemConfig()
    app.state.config = config
    
    # Initialize services
    # TODO: Initialize services here
    
    yield
    
    # Shutdown
    logger.info("Shutting down the application")
    # Clean up resources
    # TODO: Clean up resources here


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        The configured FastAPI application
    """
    app = FastAPI(
        title="Crowd Sentiment Music Generator",
        description="AI-powered system that analyzes live crowd noise from sports events and generates real-time musical compositions",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # TODO: Configure this for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health.router)
    app.include_router(music.router)
    app.include_router(events.router)
    
    return app