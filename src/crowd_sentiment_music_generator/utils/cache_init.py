"""Cache initialization utilities."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from crowd_sentiment_music_generator.config.cache_config import RedisCacheConfig, get_cache_settings
from crowd_sentiment_music_generator.utils.cache import RedisCache

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan_cache_setup(app: FastAPI) -> AsyncGenerator[None, None]:
    """Set up Redis cache during application startup.

    Args:
        app: FastAPI application instance

    Yields:
        None
    """
    # Initialize cache on startup
    cache_settings = get_cache_settings()
    if cache_settings.enabled:
        logger.info("Initializing Redis cache")
        redis_config = RedisCacheConfig.from_env()
        cache = RedisCache(redis_config)
        
        if cache.is_available():
            logger.info("Redis cache is available and ready")
            app.state.cache = cache
            app.state.cache_settings = cache_settings
        else:
            logger.warning("Redis cache is not available, continuing without cache")
            app.state.cache = None
            app.state.cache_settings = None
    else:
        logger.info("Caching is disabled by configuration")
        app.state.cache = None
        app.state.cache_settings = None
    
    yield
    
    # Clean up on shutdown
    if hasattr(app.state, "cache") and app.state.cache is not None:
        logger.info("Cleaning up Redis cache connections")
        # Redis connections are automatically closed when the client is garbage collected
        # but we can explicitly set it to None to ensure it happens now
        app.state.cache = None