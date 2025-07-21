"""Cache configuration for the application."""

import os
from typing import Dict, Optional

from pydantic import BaseModel, Field

from crowd_sentiment_music_generator.utils.cache import CacheConfig


class RedisCacheConfig(CacheConfig):
    """Redis cache configuration with environment variable support."""

    @classmethod
    def from_env(cls) -> "RedisCacheConfig":
        """Create Redis configuration from environment variables.

        Returns:
            RedisCacheConfig instance with values from environment variables
        """
        return cls(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            db=int(os.environ.get("REDIS_DB", "0")),
            password=os.environ.get("REDIS_PASSWORD"),
            socket_timeout=int(os.environ.get("REDIS_SOCKET_TIMEOUT", "5")),
            default_ttl=int(os.environ.get("REDIS_DEFAULT_TTL", "3600")),
        )


class CacheSettings(BaseModel):
    """Global cache settings."""

    enabled: bool = Field(default=True, description="Enable or disable caching globally")
    ttl_settings: Dict[str, int] = Field(
        default_factory=lambda: {
            "event_data": 300,  # 5 minutes
            "audio_features": 1800,  # 30 minutes
            "emotion_classification": 600,  # 10 minutes
            "music_parameters": 300,  # 5 minutes
            "highlight_data": 3600,  # 1 hour
            "user_preferences": 86400,  # 24 hours
        },
        description="TTL settings for different types of data",
    )


def get_cache_settings() -> CacheSettings:
    """Get cache settings from environment or defaults.

    Returns:
        CacheSettings instance
    """
    enabled = os.environ.get("CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
    
    # Default TTL settings
    ttl_settings = {
        "event_data": int(os.environ.get("CACHE_TTL_EVENT_DATA", "300")),
        "audio_features": int(os.environ.get("CACHE_TTL_AUDIO_FEATURES", "1800")),
        "emotion_classification": int(os.environ.get("CACHE_TTL_EMOTION_CLASSIFICATION", "600")),
        "music_parameters": int(os.environ.get("CACHE_TTL_MUSIC_PARAMETERS", "300")),
        "highlight_data": int(os.environ.get("CACHE_TTL_HIGHLIGHT_DATA", "3600")),
        "user_preferences": int(os.environ.get("CACHE_TTL_USER_PREFERENCES", "86400")),
    }
    
    return CacheSettings(enabled=enabled, ttl_settings=ttl_settings)