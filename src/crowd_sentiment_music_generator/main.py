"""Main entry point for the crowd sentiment music generator."""

import logging
import os

import uvicorn

from crowd_sentiment_music_generator.api.app import create_app
from crowd_sentiment_music_generator.log_formatter.custom_json_logger import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Main function to start the FastAPI application."""
    app = create_app()
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "crowd_sentiment_music_generator.api.app:create_app",
        host=host,
        port=port,
        factory=True,
        reload=True,
    )


if __name__ == "__main__":
    setup_logging()
    main()
