"""Unit tests for Redis cache implementation."""

import json
import pytest
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from crowd_sentiment_music_generator.utils.cache import (
    CacheConfig,
    RedisCache,
    CacheInvalidationStrategy,
    TimeBasedInvalidation,
    PatternBasedInvalidation,
    TagBasedInvalidation,
    CacheManager,
    cached,
)


class TestModel(BaseModel):
    """Test model for cache tests."""

    id: str
    name: str
    value: int


class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CacheConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password is None
        assert config.socket_timeout == 5
        assert config.default_ttl == 3600

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CacheConfig(
            host="redis.example.com",
            port=6380,
            db=1,
            password="secret",
            socket_timeout=10,
            default_ttl=7200,
        )
        assert config.host == "redis.example.com"
        assert config.port == 6380
        assert config.db == 1
        assert config.password == "secret"
        assert config.socket_timeout == 10
        assert config.default_ttl == 7200


class TestRedisCache:
    """Tests for RedisCache."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        with patch("redis.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_redis.return_value = mock_client
            mock_client.ping.return_value = True
            yield mock_client

    @pytest.fixture
    def redis_cache(self, mock_redis):
        """Create a RedisCache instance with a mock Redis client."""
        cache = RedisCache(CacheConfig())
        cache._client = mock_redis
        return cache

    def test_singleton_pattern(self):
        """Test that RedisCache is a singleton."""
        cache1 = RedisCache()
        cache2 = RedisCache()
        assert cache1 is cache2

    def test_is_available(self, redis_cache, mock_redis):
        """Test is_available method."""
        mock_redis.ping.return_value = True
        assert redis_cache.is_available() is True

        mock_redis.ping.return_value = False
        assert redis_cache.is_available() is False

        mock_redis.ping.side_effect = Exception("Connection error")
        assert redis_cache.is_available() is False

    def test_get(self, redis_cache, mock_redis):
        """Test get method."""
        mock_redis.get.return_value = "test_value"
        assert redis_cache.get("test_key") == "test_value"

        mock_redis.get.side_effect = Exception("Redis error")
        assert redis_cache.get("test_key") is None

    def test_set(self, redis_cache, mock_redis):
        """Test set method."""
        mock_redis.set.return_value = True
        assert redis_cache.set("test_key", "test_value") is True
        mock_redis.set.assert_called_with("test_key", "test_value", ex=3600)

        mock_redis.set.return_value = False
        assert redis_cache.set("test_key", "test_value", ttl=60) is False
        mock_redis.set.assert_called_with("test_key", "test_value", ex=60)

    def test_delete(self, redis_cache, mock_redis):
        """Test delete method."""
        mock_redis.delete.return_value = 1
        assert redis_cache.delete("test_key") is True
        mock_redis.delete.assert_called_with("test_key")

        mock_redis.delete.return_value = 0
        assert redis_cache.delete("test_key") is False

    def test_delete_pattern(self, redis_cache, mock_redis):
        """Test delete_pattern method."""
        mock_redis.keys.return_value = ["key1", "key2"]
        mock_redis.delete.return_value = 2
        assert redis_cache.delete_pattern("test_*") == 2
        mock_redis.keys.assert_called_with("test_*")
        mock_redis.delete.assert_called_with("key1", "key2")

        mock_redis.keys.return_value = []
        assert redis_cache.delete_pattern("test_*") == 0

    def test_get_model(self, redis_cache, mock_redis):
        """Test get_model method."""
        test_model = TestModel(id="1", name="test", value=42)
        mock_redis.get.return_value = test_model.model_dump_json()
        
        result = redis_cache.get_model("test_key", TestModel)
        assert isinstance(result, TestModel)
        assert result.id == "1"
        assert result.name == "test"
        assert result.value == 42

        mock_redis.get.return_value = None
        assert redis_cache.get_model("test_key", TestModel) is None

        mock_redis.get.return_value = "invalid json"
        assert redis_cache.get_model("test_key", TestModel) is None

    def test_set_model(self, redis_cache, mock_redis):
        """Test set_model method."""
        test_model = TestModel(id="1", name="test", value=42)
        mock_redis.set.return_value = True
        
        assert redis_cache.set_model("test_key", test_model) is True
        mock_redis.set.assert_called_with("test_key", test_model.model_dump_json(), ex=3600)

    def test_get_json(self, redis_cache, mock_redis):
        """Test get_json method."""
        test_data = {"id": "1", "name": "test", "value": 42}
        mock_redis.get.return_value = json.dumps(test_data)
        
        result = redis_cache.get_json("test_key")
        assert result == test_data

        mock_redis.get.return_value = None
        assert redis_cache.get_json("test_key") is None

        mock_redis.get.return_value = "invalid json"
        assert redis_cache.get_json("test_key") is None

    def test_set_json(self, redis_cache, mock_redis):
        """Test set_json method."""
        test_data = {"id": "1", "name": "test", "value": 42}
        mock_redis.set.return_value = True
        
        assert redis_cache.set_json("test_key", test_data) is True
        mock_redis.set.assert_called_with("test_key", json.dumps(test_data), ex=3600)

    def test_clear_all(self, redis_cache, mock_redis):
        """Test clear_all method."""
        mock_redis.flushdb.return_value = True
        assert redis_cache.clear_all() is True
        mock_redis.flushdb.assert_called_once()


class TestCacheInvalidationStrategies:
    """Tests for cache invalidation strategies."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock RedisCache."""
        mock = MagicMock(spec=RedisCache)
        mock.delete.return_value = True
        mock.delete_pattern.return_value = 2
        mock.client = MagicMock()
        return mock

    def test_time_based_invalidation(self, mock_cache):
        """Test TimeBasedInvalidation."""
        strategy = TimeBasedInvalidation(mock_cache)
        
        assert strategy.invalidate("test_key") is True
        mock_cache.delete.assert_called_with("test_key")
        
        assert strategy.set_with_ttl("test_key", "test_value", 60) is True
        mock_cache.set.assert_called_with("test_key", "test_value", 60)

    def test_pattern_based_invalidation(self, mock_cache):
        """Test PatternBasedInvalidation."""
        strategy = PatternBasedInvalidation(mock_cache)
        
        assert strategy.invalidate_pattern("test_*") == 2
        mock_cache.delete_pattern.assert_called_with("test_*")

    def test_tag_based_invalidation(self, mock_cache):
        """Test TagBasedInvalidation."""
        strategy = TagBasedInvalidation(mock_cache)
        
        mock_cache.client.sadd.return_value = 1
        assert strategy.add_tag("test_tag", "test_key") is True
        mock_cache.client.sadd.assert_called_with("tag:test_tag", "test_key")
        
        mock_cache.client.smembers.return_value = {"key1", "key2"}
        mock_cache.client.delete.return_value = 2
        assert strategy.invalidate_by_tag("test_tag") == 2
        mock_cache.client.smembers.assert_called_with("tag:test_tag")
        mock_cache.client.delete.assert_any_call("key1", "key2")
        mock_cache.client.delete.assert_any_call("tag:test_tag")


class TestCacheManager:
    """Tests for CacheManager."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock RedisCache."""
        mock = MagicMock(spec=RedisCache)
        mock.is_available.return_value = True
        mock.set_model.return_value = True
        mock.set_json.return_value = True
        mock.set.return_value = True
        return mock

    @pytest.fixture
    def cache_manager(self, mock_cache):
        """Create a CacheManager with a mock cache."""
        manager = CacheManager()
        manager.cache = mock_cache
        manager.pattern_strategy = MagicMock()
        manager.pattern_strategy.invalidate_pattern.return_value = 2
        manager.tag_strategy = MagicMock()
        manager.tag_strategy.invalidate_by_tag.return_value = 3
        manager.tag_strategy.add_tag.return_value = True
        return manager

    def test_invalidate_by_pattern(self, cache_manager):
        """Test invalidate_by_pattern method."""
        assert cache_manager.invalidate_by_pattern("test_*") == 2
        cache_manager.pattern_strategy.invalidate_pattern.assert_called_with("test_*")

    def test_invalidate_by_tag(self, cache_manager):
        """Test invalidate_by_tag method."""
        assert cache_manager.invalidate_by_tag("test_tag") == 3
        cache_manager.tag_strategy.invalidate_by_tag.assert_called_with("test_tag")

    def test_set_with_tags(self, cache_manager, mock_cache):
        """Test set_with_tags method."""
        # Test with BaseModel
        test_model = TestModel(id="1", name="test", value=42)
        assert cache_manager.set_with_tags("test_key", test_model, ["tag1", "tag2"]) is True
        mock_cache.set_model.assert_called_with("test_key", test_model, None)
        cache_manager.tag_strategy.add_tag.assert_any_call("tag1", "test_key")
        cache_manager.tag_strategy.add_tag.assert_any_call("tag2", "test_key")
        
        # Test with dict
        test_dict = {"id": "1", "name": "test", "value": 42}
        assert cache_manager.set_with_tags("test_key", test_dict, ["tag1"], ttl=60) is True
        mock_cache.set_json.assert_called_with("test_key", test_dict, 60)
        
        # Test with serializable value
        assert cache_manager.set_with_tags("test_key", 42, ["tag1"]) is True
        
        # Test with non-serializable value
        mock_cache.set.side_effect = TypeError("Not serializable")
        assert cache_manager.set_with_tags("test_key", object(), ["tag1"]) is True
        mock_cache.set.assert_called_with("test_key", str(object()), None)


class TestCachedDecorator:
    """Tests for cached decorator."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock RedisCache."""
        with patch("crowd_sentiment_music_generator.utils.cache.RedisCache") as mock:
            instance = MagicMock()
            mock.return_value = instance
            instance.is_available.return_value = True
            instance.get.return_value = None
            instance.set.return_value = True
            yield instance

    def test_cached_function(self, mock_cache):
        """Test cached decorator with a simple function."""
        @cached(ttl=60)
        def test_function(arg1, arg2):
            return f"{arg1}:{arg2}"
        
        # First call should execute the function
        result = test_function("test", 123)
        assert result == "test:123"
        mock_cache.get.assert_called()
        mock_cache.set.assert_called()
        
        # Set up cache hit for second call
        mock_cache.get.return_value = json.dumps("test:123")
        
        # Second call should use cached result
        result = test_function("test", 123)
        assert result == "test:123"

    def test_cached_with_model(self, mock_cache):
        """Test cached decorator with a function returning a Pydantic model."""
        @cached(ttl=60)
        def test_function(arg1):
            return TestModel(id=arg1, name="test", value=42)
        
        # First call should execute the function
        result = test_function("test_id")
        assert isinstance(result, TestModel)
        assert result.id == "test_id"
        mock_cache.set_model.assert_called()

    def test_cached_with_key_builder(self, mock_cache):
        """Test cached decorator with a custom key builder."""
        def key_builder(arg1, arg2):
            return f"custom:{arg1}:{arg2}"
        
        @cached(ttl=60, key_builder=key_builder)
        def test_function(arg1, arg2):
            return f"{arg1}:{arg2}"
        
        result = test_function("test", 123)
        assert result == "test:123"
        mock_cache.get.assert_called_with("custom:test:123")

    def test_cached_with_tag(self, mock_cache):
        """Test cached decorator with a tag."""
        @cached(ttl=60, tag="test_tag")
        def test_function(arg):
            return f"result:{arg}"
        
        result = test_function("test")
        assert result == "result:test"
        
        # Check that tag was added
        tag_strategy = TagBasedInvalidation(mock_cache)
        tag_strategy.add_tag.assert_called_with("test_tag", mock_cache.set.call_args[0][0])

    def test_cached_unavailable_cache(self, mock_cache):
        """Test cached decorator when cache is unavailable."""
        mock_cache.is_available.return_value = False
        
        @cached(ttl=60)
        def test_function(arg):
            return f"result:{arg}"
        
        result = test_function("test")
        assert result == "result:test"
        mock_cache.get.assert_not_called()
        mock_cache.set.assert_not_called()