"""Redis cache implementation for the crowd sentiment music generator.

This module provides a Redis-based caching system with support for:
- Key-value storage
- Cache invalidation strategies
- Efficient data retrieval patterns
"""

import json
import logging
from datetime import timedelta
from functools import wraps
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar, Union, cast

import redis
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class CacheConfig(BaseModel):
    """Configuration for Redis cache."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: int = 5
    default_ttl: int = 3600  # 1 hour default TTL


class RedisCache:
    """Redis cache implementation with invalidation strategies."""

    _instance: Optional["RedisCache"] = None
    _client: Optional[redis.Redis] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "RedisCache":
        """Create a singleton instance of RedisCache."""
        if cls._instance is None:
            cls._instance = super(RedisCache, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        """Initialize Redis cache with configuration.

        Args:
            config: Redis configuration parameters
        """
        if self._client is not None:
            return

        self.config = config or CacheConfig()
        try:
            self._client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                socket_timeout=self.config.socket_timeout,
                decode_responses=True,
            )
            logger.info("Redis cache initialized successfully")
        except redis.RedisError as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            self._client = None

    @property
    def client(self) -> redis.Redis:
        """Get the Redis client instance.

        Returns:
            Redis client instance

        Raises:
            RuntimeError: If Redis client is not initialized
        """
        if self._client is None:
            raise RuntimeError("Redis client is not initialized")
        return self._client

    def is_available(self) -> bool:
        """Check if Redis is available.

        Returns:
            True if Redis is available, False otherwise
        """
        if self._client is None:
            return False

        try:
            return bool(self._client.ping())
        except redis.RedisError:
            return False

    def get(self, key: str) -> Optional[str]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self.is_available():
            logger.warning("Redis cache is not available, skipping get operation")
            return None

        try:
            return self.client.get(key)
        except redis.RedisError as e:
            logger.error(f"Error getting value from cache: {e}")
            return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds, uses default_ttl if None

        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.warning("Redis cache is not available, skipping set operation")
            return False

        ttl = ttl if ttl is not None else self.config.default_ttl

        try:
            return bool(self.client.set(key, value, ex=ttl))
        except redis.RedisError as e:
            logger.error(f"Error setting value in cache: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if key was deleted, False otherwise
        """
        if not self.is_available():
            logger.warning("Redis cache is not available, skipping delete operation")
            return False

        try:
            return bool(self.client.delete(key))
        except redis.RedisError as e:
            logger.error(f"Error deleting key from cache: {e}")
            return False

    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern.

        Args:
            pattern: Pattern to match keys (e.g., "user:*")

        Returns:
            Number of keys deleted
        """
        if not self.is_available():
            logger.warning("Redis cache is not available, skipping delete_pattern operation")
            return 0

        try:
            keys = self.client.keys(pattern)
            if not keys:
                return 0
            return self.client.delete(*keys)
        except redis.RedisError as e:
            logger.error(f"Error deleting keys by pattern: {e}")
            return 0

    def get_model(self, key: str, model_class: Type[T]) -> Optional[T]:
        """Get cached Pydantic model.

        Args:
            key: Cache key
            model_class: Pydantic model class

        Returns:
            Pydantic model instance or None if not found
        """
        data = self.get(key)
        if not data:
            return None

        try:
            return model_class.model_validate_json(data)
        except Exception as e:
            logger.error(f"Error deserializing cached model: {e}")
            return None

    def set_model(self, key: str, model: BaseModel, ttl: Optional[int] = None) -> bool:
        """Cache Pydantic model.

        Args:
            key: Cache key
            model: Pydantic model instance
            ttl: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        try:
            json_data = model.model_dump_json()
            return self.set(key, json_data, ttl)
        except Exception as e:
            logger.error(f"Error serializing model for cache: {e}")
            return False

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Get JSON data from cache.

        Args:
            key: Cache key

        Returns:
            Deserialized JSON data or None if not found
        """
        data = self.get(key)
        if not data:
            return None

        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            logger.error(f"Error deserializing JSON from cache: {e}")
            return None

    def set_json(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache JSON data.

        Args:
            key: Cache key
            data: Dictionary to cache
            ttl: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        try:
            json_data = json.dumps(data)
            return self.set(key, json_data, ttl)
        except (TypeError, ValueError) as e:
            logger.error(f"Error serializing JSON for cache: {e}")
            return False

    def clear_all(self) -> bool:
        """Clear all keys in the current database.

        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.warning("Redis cache is not available, skipping clear_all operation")
            return False

        try:
            return bool(self.client.flushdb())
        except redis.RedisError as e:
            logger.error(f"Error clearing cache: {e}")
            return False


class CacheInvalidationStrategy:
    """Base class for cache invalidation strategies."""

    def __init__(self, cache: RedisCache) -> None:
        """Initialize with a Redis cache instance.

        Args:
            cache: RedisCache instance
        """
        self.cache = cache

    def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache key.

        Args:
            key: Cache key to invalidate

        Returns:
            True if successful, False otherwise
        """
        return self.cache.delete(key)


class TimeBasedInvalidation(CacheInvalidationStrategy):
    """Time-based cache invalidation strategy."""

    def set_with_ttl(self, key: str, value: str, ttl: int) -> bool:
        """Set value with specific TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        return self.cache.set(key, value, ttl)


class PatternBasedInvalidation(CacheInvalidationStrategy):
    """Pattern-based cache invalidation strategy."""

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern.

        Args:
            pattern: Pattern to match keys (e.g., "user:*")

        Returns:
            Number of keys invalidated
        """
        return self.cache.delete_pattern(pattern)


class TagBasedInvalidation(CacheInvalidationStrategy):
    """Tag-based cache invalidation strategy."""

    def add_tag(self, tag: str, key: str) -> bool:
        """Add a tag to a key.

        Args:
            tag: Tag name
            key: Cache key to tag

        Returns:
            True if successful, False otherwise
        """
        tag_key = f"tag:{tag}"
        try:
            return bool(self.cache.client.sadd(tag_key, key))
        except redis.RedisError as e:
            logger.error(f"Error adding tag: {e}")
            return False

    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all keys with specific tag.

        Args:
            tag: Tag name

        Returns:
            Number of keys invalidated
        """
        tag_key = f"tag:{tag}"
        try:
            keys = self.cache.client.smembers(tag_key)
            if not keys:
                return 0

            # Delete all keys in the tag
            deleted = self.cache.client.delete(*keys)

            # Delete the tag itself
            self.cache.client.delete(tag_key)

            return deleted
        except redis.RedisError as e:
            logger.error(f"Error invalidating by tag: {e}")
            return 0


def cached(
    ttl: Optional[int] = None,
    key_prefix: str = "",
    key_builder: Optional[Callable[..., str]] = None,
    tag: Optional[str] = None,
) -> Callable:
    """Decorator for caching function results.

    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        key_builder: Function to build cache key from arguments
        tag: Optional tag for grouped invalidation

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache = RedisCache()
            if not cache.is_available():
                return func(*args, **kwargs)

            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key builder uses function name and arguments
                arg_str = ":".join(str(arg) for arg in args if not isinstance(arg, BaseModel))
                kwarg_str = ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = f"{key_prefix}:{func.__name__}:{arg_str}:{kwarg_str}"

            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    # If not JSON, return as is
                    return result

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            if result is not None:
                try:
                    if isinstance(result, BaseModel):
                        cache.set_model(cache_key, result, ttl)
                    elif isinstance(result, dict):
                        cache.set_json(cache_key, result, ttl)
                    else:
                        try:
                            json_result = json.dumps(result)
                            cache.set(cache_key, json_result, ttl)
                        except (TypeError, ValueError):
                            # If not serializable, don't cache
                            pass

                    # Add tag if specified
                    if tag and cache.is_available():
                        tag_strategy = TagBasedInvalidation(cache)
                        tag_strategy.add_tag(tag, cache_key)
                except Exception as e:
                    logger.error(f"Error caching result: {e}")

            return result

        return wrapper

    return decorator


class CacheManager:
    """Manager for cache operations and invalidation strategies."""

    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        """Initialize cache manager.

        Args:
            config: Redis configuration
        """
        self.cache = RedisCache(config)
        self.time_strategy = TimeBasedInvalidation(self.cache)
        self.pattern_strategy = PatternBasedInvalidation(self.cache)
        self.tag_strategy = TagBasedInvalidation(self.cache)

    def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate cache by pattern.

        Args:
            pattern: Pattern to match keys

        Returns:
            Number of keys invalidated
        """
        return self.pattern_strategy.invalidate_pattern(pattern)

    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate cache by tag.

        Args:
            tag: Tag name

        Returns:
            Number of keys invalidated
        """
        return self.tag_strategy.invalidate_by_tag(tag)

    def set_with_tags(self, key: str, value: Any, tags: list[str], ttl: Optional[int] = None) -> bool:
        """Set value with multiple tags.

        Args:
            key: Cache key
            value: Value to cache
            tags: List of tags to associate with the key
            ttl: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        # Set the value
        if isinstance(value, BaseModel):
            success = self.cache.set_model(key, value, ttl)
        elif isinstance(value, dict):
            success = self.cache.set_json(key, value, ttl)
        else:
            try:
                json_value = json.dumps(value)
                success = self.cache.set(key, json_value, ttl)
            except (TypeError, ValueError):
                # If not serializable, try as string
                success = self.cache.set(key, str(value), ttl)

        if not success:
            return False

        # Add tags
        for tag in tags:
            self.tag_strategy.add_tag(tag, key)

        return True