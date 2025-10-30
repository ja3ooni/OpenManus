"""
Tests for the intelligent caching system.
"""

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.performance.cache import (
    CacheEntry,
    CacheLevel,
    CacheStats,
    FileCache,
    IntelligentCache,
    MemoryCache,
    RedisCache,
    cache_result,
    get_cache_stats,
    invalidate_cache_pattern,
)


class TestCacheEntry:
    """Test cache entry functionality."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""

        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc),
            ttl=3600,
        )

        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.ttl == 3600
        assert not entry.is_expired
        assert entry.access_count == 0

    def test_cache_entry_expiration(self):
        """Test cache entry expiration."""

        # Create expired entry
        past_time = datetime.now(timezone.utc) - timedelta(hours=2)
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=past_time,
            last_accessed=past_time,
            ttl=3600,  # 1 hour TTL, but created 2 hours ago
        )

        assert entry.is_expired

    def test_cache_entry_touch(self):
        """Test cache entry touch functionality."""

        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc),
        )

        initial_access_count = entry.access_count
        initial_last_accessed = entry.last_accessed

        # Wait a bit and touch
        import time

        time.sleep(0.01)
        entry.touch()

        assert entry.access_count == initial_access_count + 1
        assert entry.last_accessed > initial_last_accessed


class TestMemoryCache:
    """Test memory cache functionality."""

    @pytest.fixture
    def memory_cache(self):
        """Create memory cache for testing."""
        return MemoryCache(max_size=10, default_ttl=3600)

    @pytest.mark.asyncio
    async def test_memory_cache_set_get(self, memory_cache):
        """Test basic set and get operations."""

        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc),
        )

        # Set entry
        result = await memory_cache.set("test_key", entry)
        assert result is True

        # Get entry
        retrieved_entry = await memory_cache.get("test_key")
        assert retrieved_entry is not None
        assert retrieved_entry.value == "test_value"
        assert retrieved_entry.access_count == 1  # Should be incremented by get

    @pytest.mark.asyncio
    async def test_memory_cache_miss(self, memory_cache):
        """Test cache miss."""

        entry = await memory_cache.get("nonexistent_key")
        assert entry is None

        stats = memory_cache.get_stats()
        assert stats.misses == 1
        assert stats.hits == 0

    @pytest.mark.asyncio
    async def test_memory_cache_expiration(self, memory_cache):
        """Test cache entry expiration."""

        # Create expired entry
        past_time = datetime.now(timezone.utc) - timedelta(hours=2)
        entry = CacheEntry(
            key="expired_key",
            value="expired_value",
            created_at=past_time,
            last_accessed=past_time,
            ttl=3600,  # 1 hour TTL
        )

        await memory_cache.set("expired_key", entry)

        # Try to get expired entry
        retrieved_entry = await memory_cache.get("expired_key")
        assert retrieved_entry is None

        # Should be removed from cache
        assert await memory_cache.size() == 0

    @pytest.mark.asyncio
    async def test_memory_cache_eviction(self, memory_cache):
        """Test LRU eviction when cache is full."""

        # Fill cache to capacity
        for i in range(10):
            entry = CacheEntry(
                key=f"key_{i}",
                value=f"value_{i}",
                created_at=datetime.now(timezone.utc),
                last_accessed=datetime.now(timezone.utc),
            )
            await memory_cache.set(f"key_{i}", entry)

        assert await memory_cache.size() == 10

        # Add one more entry to trigger eviction
        entry = CacheEntry(
            key="key_10",
            value="value_10",
            created_at=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc),
        )
        await memory_cache.set("key_10", entry)

        # Should still be at max size
        assert await memory_cache.size() == 10

        # First entry should be evicted (LRU)
        first_entry = await memory_cache.get("key_0")
        assert first_entry is None

        # Last entry should be present
        last_entry = await memory_cache.get("key_10")
        assert last_entry is not None

    @pytest.mark.asyncio
    async def test_memory_cache_delete(self, memory_cache):
        """Test cache entry deletion."""

        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc),
        )

        await memory_cache.set("test_key", entry)
        assert await memory_cache.size() == 1

        # Delete entry
        result = await memory_cache.delete("test_key")
        assert result is True
        assert await memory_cache.size() == 0

        # Try to delete non-existent entry
        result = await memory_cache.delete("nonexistent_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_memory_cache_clear(self, memory_cache):
        """Test clearing all cache entries."""

        # Add multiple entries
        for i in range(5):
            entry = CacheEntry(
                key=f"key_{i}",
                value=f"value_{i}",
                created_at=datetime.now(timezone.utc),
                last_accessed=datetime.now(timezone.utc),
            )
            await memory_cache.set(f"key_{i}", entry)

        assert await memory_cache.size() == 5

        # Clear cache
        result = await memory_cache.clear()
        assert result is True
        assert await memory_cache.size() == 0

    @pytest.mark.asyncio
    async def test_memory_cache_keys(self, memory_cache):
        """Test getting cache keys."""

        # Add entries
        for i in range(3):
            entry = CacheEntry(
                key=f"test_key_{i}",
                value=f"value_{i}",
                created_at=datetime.now(timezone.utc),
                last_accessed=datetime.now(timezone.utc),
            )
            await memory_cache.set(f"test_key_{i}", entry)

        # Get all keys
        keys = await memory_cache.keys()
        assert len(keys) == 3
        assert "test_key_0" in keys
        assert "test_key_1" in keys
        assert "test_key_2" in keys

        # Get keys with pattern
        pattern_keys = await memory_cache.keys("test_key_[01]")
        assert len(pattern_keys) == 2


class TestFileCache:
    """Test file cache functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for file cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def file_cache(self, temp_dir):
        """Create file cache for testing."""
        return FileCache(cache_dir=temp_dir, max_size_mb=10)

    @pytest.mark.asyncio
    async def test_file_cache_set_get(self, file_cache):
        """Test basic file cache operations."""

        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc),
        )

        # Set entry
        result = await file_cache.set("test_key", entry)
        assert result is True

        # Get entry
        retrieved_entry = await file_cache.get("test_key")
        assert retrieved_entry is not None
        assert retrieved_entry.value == "test_value"
        assert retrieved_entry.key == "test_key"

    @pytest.mark.asyncio
    async def test_file_cache_persistence(self, temp_dir):
        """Test file cache persistence across instances."""

        # Create first cache instance
        cache1 = FileCache(cache_dir=temp_dir)

        entry = CacheEntry(
            key="persistent_key",
            value="persistent_value",
            created_at=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc),
        )

        await cache1.set("persistent_key", entry)

        # Create second cache instance
        cache2 = FileCache(cache_dir=temp_dir)

        # Should be able to retrieve from second instance
        retrieved_entry = await cache2.get("persistent_key")
        assert retrieved_entry is not None
        assert retrieved_entry.value == "persistent_value"

    @pytest.mark.asyncio
    async def test_file_cache_expiration(self, file_cache):
        """Test file cache expiration."""

        # Create expired entry
        past_time = datetime.now(timezone.utc) - timedelta(hours=2)
        entry = CacheEntry(
            key="expired_key",
            value="expired_value",
            created_at=past_time,
            last_accessed=past_time,
            ttl=3600,  # 1 hour TTL
        )

        await file_cache.set("expired_key", entry)

        # Try to get expired entry
        retrieved_entry = await file_cache.get("expired_key")
        assert retrieved_entry is None


class TestRedisCache:
    """Test Redis cache functionality."""

    @pytest.fixture
    def redis_cache(self):
        """Create Redis cache for testing."""
        return RedisCache(host="localhost", port=6379, db=15)  # Use test DB

    @pytest.mark.asyncio
    async def test_redis_cache_connection_failure(self, redis_cache):
        """Test Redis cache behavior when Redis is not available."""

        # Mock Redis to simulate connection failure
        with patch("redis.asyncio.Redis") as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.ping.side_effect = Exception("Connection failed")
            mock_redis_class.return_value = mock_redis

            # Should handle connection failure gracefully
            entry = await redis_cache.get("test_key")
            assert entry is None

            result = await redis_cache.set(
                "test_key",
                CacheEntry(
                    key="test_key",
                    value="test_value",
                    created_at=datetime.now(timezone.utc),
                    last_accessed=datetime.now(timezone.utc),
                ),
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_redis_cache_operations_without_redis(self, redis_cache):
        """Test Redis cache operations when Redis is not installed."""

        # Mock ImportError for redis module
        with patch("app.performance.cache.redis", side_effect=ImportError):
            # Should handle missing Redis gracefully
            entry = await redis_cache.get("test_key")
            assert entry is None


class TestIntelligentCache:
    """Test intelligent multi-level cache."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for file cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def intelligent_cache(self, temp_dir):
        """Create intelligent cache for testing."""
        memory_cache = MemoryCache(max_size=10)
        file_cache = FileCache(cache_dir=temp_dir)

        return IntelligentCache(
            memory_cache=memory_cache,
            redis_cache=None,  # Skip Redis for tests
            file_cache=file_cache,
            enable_cache_warming=False,  # Disable for tests
        )

    @pytest.mark.asyncio
    async def test_intelligent_cache_multi_level(self, intelligent_cache):
        """Test multi-level cache behavior."""

        # Set value in all levels
        await intelligent_cache.set(
            "test_key", "test_value", cache_level=CacheLevel.ALL
        )

        # Should be available from memory (fastest)
        value = await intelligent_cache.get("test_key", cache_level=CacheLevel.MEMORY)
        assert value == "test_value"

        # Should be available from file
        value = await intelligent_cache.get("test_key", cache_level=CacheLevel.FILE)
        assert value == "test_value"

    @pytest.mark.asyncio
    async def test_intelligent_cache_promotion(self, intelligent_cache):
        """Test cache promotion from lower to higher levels."""

        # Set value only in file cache
        await intelligent_cache.set(
            "test_key", "test_value", cache_level=CacheLevel.FILE
        )

        # Clear memory cache to ensure it's not there
        await intelligent_cache.memory_cache.clear()

        # Get value - should promote to memory
        value = await intelligent_cache.get("test_key")
        assert value == "test_value"

        # Should now be in memory cache
        memory_value = await intelligent_cache.get(
            "test_key", cache_level=CacheLevel.MEMORY
        )
        assert memory_value == "test_value"

    @pytest.mark.asyncio
    async def test_intelligent_cache_invalidation(self, intelligent_cache):
        """Test cache invalidation by pattern."""

        # Set multiple values
        await intelligent_cache.set("user:123", "user_data_123")
        await intelligent_cache.set("user:456", "user_data_456")
        await intelligent_cache.set("product:789", "product_data_789")

        # Invalidate user entries
        invalidated = await intelligent_cache.invalidate_by_pattern("user:.*")
        assert invalidated == 2

        # User entries should be gone
        assert await intelligent_cache.get("user:123") is None
        assert await intelligent_cache.get("user:456") is None

        # Product entry should remain
        assert await intelligent_cache.get("product:789") == "product_data_789"

    @pytest.mark.asyncio
    async def test_intelligent_cache_stats(self, intelligent_cache):
        """Test cache statistics."""

        # Perform some operations
        await intelligent_cache.set("key1", "value1")
        await intelligent_cache.set("key2", "value2")

        await intelligent_cache.get("key1")  # Hit
        await intelligent_cache.get("key3")  # Miss

        stats = await intelligent_cache.get_stats()

        assert "memory" in stats
        assert "file" in stats
        assert "combined" in stats

        combined_stats = stats["combined"]
        assert combined_stats.hits > 0
        assert combined_stats.misses > 0

    @pytest.mark.asyncio
    async def test_intelligent_cache_tags(self, intelligent_cache):
        """Test cache invalidation by tags."""

        # Set values with tags
        await intelligent_cache.set("key1", "value1", tags={"user", "profile"})
        await intelligent_cache.set("key2", "value2", tags={"user", "settings"})
        await intelligent_cache.set("key3", "value3", tags={"product"})

        # Invalidate by tag
        invalidated = await intelligent_cache.invalidate_by_tags({"user"})
        assert invalidated == 2

        # Tagged entries should be gone
        assert await intelligent_cache.get("key1") is None
        assert await intelligent_cache.get("key2") is None

        # Untagged entry should remain
        assert await intelligent_cache.get("key3") == "value3"


class TestCacheDecorator:
    """Test cache result decorator."""

    @pytest.mark.asyncio
    async def test_cache_result_decorator(self):
        """Test caching function results."""

        call_count = 0

        @cache_result(ttl=3600)
        async def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        # First call should execute function
        result1 = await expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1

        # Second call should use cache
        result2 = await expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Should not increment

        # Different arguments should execute function
        result3 = await expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cache_result_custom_key(self):
        """Test cache decorator with custom key function."""

        call_count = 0

        def custom_key_func(user_id, **kwargs):
            return f"user_data:{user_id}"

        @cache_result(ttl=3600, key_func=custom_key_func)
        async def get_user_data(user_id, include_profile=True):
            nonlocal call_count
            call_count += 1
            return {"id": user_id, "profile": include_profile}

        # First call
        result1 = await get_user_data(123, include_profile=True)
        assert result1["id"] == 123
        assert call_count == 1

        # Same user_id but different kwargs should use cache (custom key ignores kwargs)
        result2 = await get_user_data(123, include_profile=False)
        assert result2["id"] == 123
        assert result2["profile"] is True  # From cache
        assert call_count == 1


class TestCacheUtilities:
    """Test cache utility functions."""

    @pytest.mark.asyncio
    async def test_get_cache_stats(self):
        """Test getting global cache stats."""

        stats = await get_cache_stats()

        assert isinstance(stats, dict)
        assert "memory" in stats or "file" in stats or "combined" in stats

    @pytest.mark.asyncio
    async def test_invalidate_cache_pattern(self):
        """Test global cache pattern invalidation."""

        from app.performance.cache import intelligent_cache

        # Set some test values
        await intelligent_cache.set("test:pattern:1", "value1")
        await intelligent_cache.set("test:pattern:2", "value2")
        await intelligent_cache.set("other:key", "value3")

        # Invalidate pattern
        invalidated = await invalidate_cache_pattern("test:pattern:.*")
        assert invalidated >= 0  # Should not fail

        # Pattern entries should be gone
        assert await intelligent_cache.get("test:pattern:1") is None
        assert await intelligent_cache.get("test:pattern:2") is None


class TestCachePerformance:
    """Test cache performance characteristics."""

    @pytest.mark.asyncio
    async def test_memory_cache_performance(self):
        """Test memory cache performance with many operations."""

        cache = MemoryCache(max_size=1000)

        # Measure set performance
        import time

        start_time = time.time()

        for i in range(100):
            entry = CacheEntry(
                key=f"perf_key_{i}",
                value=f"perf_value_{i}",
                created_at=datetime.now(timezone.utc),
                last_accessed=datetime.now(timezone.utc),
            )
            await cache.set(f"perf_key_{i}", entry)

        set_time = time.time() - start_time

        # Measure get performance
        start_time = time.time()

        for i in range(100):
            await cache.get(f"perf_key_{i}")

        get_time = time.time() - start_time

        # Performance should be reasonable
        assert set_time < 1.0  # Should set 100 items in under 1 second
        assert get_time < 0.5  # Should get 100 items in under 0.5 seconds

        # Check hit rate
        stats = cache.get_stats()
        assert stats.hit_rate > 0.9  # Should have high hit rate

    @pytest.mark.asyncio
    async def test_cache_memory_usage(self):
        """Test cache memory usage tracking."""

        cache = MemoryCache(max_size=100)

        # Add entries and check memory usage
        for i in range(10):
            large_value = "x" * 1000  # 1KB string
            entry = CacheEntry(
                key=f"large_key_{i}",
                value=large_value,
                created_at=datetime.now(timezone.utc),
                last_accessed=datetime.now(timezone.utc),
            )
            await cache.set(f"large_key_{i}", entry)

        memory_usage = await cache.memory_usage()
        assert memory_usage > 0  # Should track memory usage

        stats = cache.get_stats()
        assert stats.memory_usage_bytes > 0


if __name__ == "__main__":
    pytest.main([__file__])
