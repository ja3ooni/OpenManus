"""
Intelligent caching system with multiple cache levels and advanced features.

This module provides a comprehensive caching solution with:
- Multi-level caching (memory, Redis, file-based)
- TTL management and automatic expiration
- Cache invalidation strategies
- Performance monitoring and optimization
- Cache warming and preloading
"""

import asyncio
import hashlib
import json
import os
import pickle
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from app.logger import logger


class CacheLevel(Enum):
    """Cache levels for multi-tier caching strategy."""
    
    MEMORY = "memory"
    REDIS = "redis"
    FILE = "file"
    ALL = "all"


class CacheStrategy(Enum):
    """Cache strategies for different use cases."""
    
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In, First Out


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[int] = None
    tags: Set[str] = field(default_factory=set)
    size_bytes: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age > self.ttl
    
    def touch(self):
        """Update last accessed time and increment access count."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache performance statistics."""
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    memory_usage_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class MemoryCache:
    """In-memory cache with LRU eviction and TTL support."""
    
    def __init__(
        self,
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.LRU,
        default_ttl: Optional[int] = None,
    ):
        """Initialize memory cache."""
        self.max_size = max_size
        self.strategy = strategy
        self.default_ttl = default_ttl
        
        # Use OrderedDict for LRU behavior
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = CacheStats()
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get a cache entry by key."""
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats.misses += 1
                return None
            
            if entry.is_expired:
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                return None
            
            # Update access information
            entry.touch()
            
            # Move to end for LRU
            if self.strategy == CacheStrategy.LRU:
                self._cache.move_to_end(key)
            
            self._stats.hits += 1
            return entry
    
    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Set a cache entry."""
        async with self._lock:
            # Set default TTL if not specified
            if entry.ttl is None and self.default_ttl is not None:
                entry.ttl = self.default_ttl
            
            # Calculate entry size
            entry.size_bytes = self._calculate_size(entry.value)
            
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats.memory_usage_bytes -= old_entry.size_bytes
                del self._cache[key]
            
            # Add new entry
            self._cache[key] = entry
            self._stats.memory_usage_bytes += entry.size_bytes
            self._stats.size = len(self._cache)
            
            # Evict if necessary
            await self._evict_if_necessary()
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._stats.memory_usage_bytes -= entry.size_bytes
                del self._cache[key]
                self._stats.size = len(self._cache)
                return True
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._stats = CacheStats()
            return True
    
    async def keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get all keys, optionally matching a pattern."""
        async with self._lock:
            keys = list(self._cache.keys())
            
            if pattern:
                import re
                regex = re.compile(pattern)
                keys = [k for k in keys if regex.match(k)]
            
            return keys
    
    async def size(self) -> int:
        """Get the number of entries in the cache."""
        return len(self._cache)
    
    async def memory_usage(self) -> int:
        """Get memory usage in bytes."""
        return self._stats.memory_usage_bytes
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        self._stats.size = len(self._cache)
        return self._stats
    
    async def _evict_if_necessary(self):
        """Evict entries if cache is over capacity."""
        while len(self._cache) > self.max_size:
            if self.strategy == CacheStrategy.LRU:
                # Remove least recently used (first item)
                key, entry = self._cache.popitem(last=False)
            else:
                # Default to LRU
                key, entry = self._cache.popitem(last=False)
            
            self._stats.memory_usage_bytes -= entry.size_bytes
            self._stats.evictions += 1
        
        self._stats.size = len(self._cache)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of a value in bytes."""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8  # Approximate size
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(
                    self._calculate_size(k) + self._calculate_size(v)
                    for k, v in value.items()
                )
            else:
                # Use pickle to estimate size
                return len(pickle.dumps(value))
        except Exception:
            return 100  # Default estimate


class FileCache:
    """File-based cache backend for persistent storage."""
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_size_mb: int = 1000,
    ):
        """Initialize file cache."""
        self.cache_dir = cache_dir or Path.home() / ".openmanus" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_mb = max_size_mb
        self._stats = CacheStats()
        self._lock = asyncio.Lock()
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Hash the key to create a safe filename
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get a cache entry by key."""
        file_path = self._get_file_path(key)
        
        if not file_path.exists():
            self._stats.misses += 1
            return None
        
        try:
            async with self._lock:
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f)
                
                if entry.is_expired:
                    file_path.unlink(missing_ok=True)
                    self._stats.misses += 1
                    self._stats.evictions += 1
                    return None
                
                # Update access information
                entry.touch()
                
                # Write back updated entry
                with open(file_path, 'wb') as f:
                    pickle.dump(entry, f)
                
                self._stats.hits += 1
                return entry
                
        except Exception as e:
            logger.error(f"File cache get error: {e}")
            self._stats.misses += 1
            return None
    
    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Set a cache entry."""
        file_path = self._get_file_path(key)
        
        try:
            async with self._lock:
                with open(file_path, 'wb') as f:
                    pickle.dump(entry, f)
                
                # Update stats
                entry.size_bytes = file_path.stat().st_size
                self._stats.memory_usage_bytes += entry.size_bytes
                
                return True
                
        except Exception as e:
            logger.error(f"File cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        file_path = self._get_file_path(key)
        
        try:
            if file_path.exists():
                size = file_path.stat().st_size
                file_path.unlink()
                self._stats.memory_usage_bytes -= size
                return True
            return False
            
        except Exception as e:
            logger.error(f"File cache delete error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            async with self._lock:
                for file_path in self.cache_dir.glob("*.cache"):
                    file_path.unlink(missing_ok=True)
                
                self._stats = CacheStats()
                return True
                
        except Exception as e:
            logger.error(f"File cache clear error: {e}")
            return False
    
    async def keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get all keys, optionally matching a pattern."""
        keys = []
        
        try:
            for file_path in self.cache_dir.glob("*.cache"):
                try:
                    with open(file_path, 'rb') as f:
                        entry = pickle.load(f)
                        keys.append(entry.key)
                except Exception:
                    continue  # Skip corrupted files
            
            if pattern:
                import re
                regex = re.compile(pattern)
                keys = [k for k in keys if regex.match(k)]
            
            return keys
            
        except Exception as e:
            logger.error(f"File cache keys error: {e}")
            return []
    
    async def size(self) -> int:
        """Get the number of entries in the cache."""
        try:
            return len(list(self.cache_dir.glob("*.cache")))
        except Exception:
            return 0
    
    async def memory_usage(self) -> int:
        """Get memory usage in bytes."""
        try:
            total_size = 0
            for file_path in self.cache_dir.glob("*.cache"):
                total_size += file_path.stat().st_size
            return total_size
        except Exception:
            return 0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


class RedisCache:
    """Redis-based cache backend (placeholder for Redis integration)."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """Initialize Redis cache."""
        self.host = host
        self.port = port
        self.db = db
        self._stats = CacheStats()
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get a cache entry by key."""
        # Redis implementation would go here
        self._stats.misses += 1
        return None
    
    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Set a cache entry."""
        # Redis implementation would go here
        return False
    
    async def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        # Redis implementation would go here
        return False
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        # Redis implementation would go here
        return False
    
    async def keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get all keys, optionally matching a pattern."""
        # Redis implementation would go here
        return []
    
    async def size(self) -> int:
        """Get the number of entries in the cache."""
        return 0
    
    async def memory_usage(self) -> int:
        """Get memory usage in bytes."""
        return 0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


class IntelligentCache:
    """Multi-level intelligent cache with advanced features."""
    
    def __init__(
        self,
        memory_cache: Optional[MemoryCache] = None,
        redis_cache: Optional[RedisCache] = None,
        file_cache: Optional[FileCache] = None,
        default_ttl: Optional[int] = 3600,
        enable_cache_warming: bool = True,
    ):
        """Initialize intelligent cache."""
        self.memory_cache = memory_cache or MemoryCache()
        self.redis_cache = redis_cache
        self.file_cache = file_cache or FileCache()
        self.default_ttl = default_ttl
        self.enable_cache_warming = enable_cache_warming
        
        # Cache warming data
        self._warm_keys: Set[str] = set()
    
    async def get(
        self,
        key: str,
        cache_level: CacheLevel = CacheLevel.ALL,
        default: Any = None,
    ) -> Any:
        """Get value from cache with multi-level fallback."""
        
        # Try memory cache first
        if cache_level in [CacheLevel.MEMORY, CacheLevel.ALL]:
            entry = await self.memory_cache.get(key)
            if entry is not None:
                return entry.value
        
        # Try Redis cache
        if cache_level in [CacheLevel.REDIS, CacheLevel.ALL] and self.redis_cache:
            entry = await self.redis_cache.get(key)
            if entry is not None:
                # Promote to memory cache
                await self.memory_cache.set(key, entry)
                return entry.value
        
        # Try file cache
        if cache_level in [CacheLevel.FILE, CacheLevel.ALL]:
            entry = await self.file_cache.get(key)
            if entry is not None:
                # Promote to higher levels
                await self.memory_cache.set(key, entry)
                if self.redis_cache:
                    await self.redis_cache.set(key, entry)
                return entry.value
        
        return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        cache_level: CacheLevel = CacheLevel.ALL,
        tags: Optional[Set[str]] = None,
    ) -> bool:
        """Set value in cache at specified levels."""
        
        # Use default TTL if not specified
        if ttl is None:
            ttl = self.default_ttl
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc),
            ttl=ttl,
            tags=tags or set(),
        )
        
        success = True
        
        # Set in memory cache
        if cache_level in [CacheLevel.MEMORY, CacheLevel.ALL]:
            success &= await self.memory_cache.set(key, entry)
        
        # Set in Redis cache
        if cache_level in [CacheLevel.REDIS, CacheLevel.ALL] and self.redis_cache:
            success &= await self.redis_cache.set(key, entry)
        
        # Set in file cache
        if cache_level in [CacheLevel.FILE, CacheLevel.ALL]:
            success &= await self.file_cache.set(key, entry)
        
        return success
    
    async def delete(
        self,
        key: str,
        cache_level: CacheLevel = CacheLevel.ALL,
    ) -> bool:
        """Delete key from specified cache levels."""
        
        success = True
        
        if cache_level in [CacheLevel.MEMORY, CacheLevel.ALL]:
            success &= await self.memory_cache.delete(key)
        
        if cache_level in [CacheLevel.REDIS, CacheLevel.ALL] and self.redis_cache:
            success &= await self.redis_cache.delete(key)
        
        if cache_level in [CacheLevel.FILE, CacheLevel.ALL]:
            success &= await self.file_cache.delete(key)
        
        return success
    
    async def clear(self, cache_level: CacheLevel = CacheLevel.ALL) -> bool:
        """Clear specified cache levels."""
        
        success = True
        
        if cache_level in [CacheLevel.MEMORY, CacheLevel.ALL]:
            success &= await self.memory_cache.clear()
        
        if cache_level in [CacheLevel.REDIS, CacheLevel.ALL] and self.redis_cache:
            success &= await self.redis_cache.clear()
        
        if cache_level in [CacheLevel.FILE, CacheLevel.ALL]:
            success &= await self.file_cache.clear()
        
        return success
    
    async def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern."""
        
        invalidated_count = 0
        
        # Get keys matching pattern from all levels
        all_keys = set()
        
        memory_keys = await self.memory_cache.keys(pattern)
        all_keys.update(memory_keys)
        
        if self.redis_cache:
            redis_keys = await self.redis_cache.keys(pattern)
            all_keys.update(redis_keys)
        
        file_keys = await self.file_cache.keys(pattern)
        all_keys.update(file_keys)
        
        # Delete matching keys
        for key in all_keys:
            if await self.delete(key):
                invalidated_count += 1
        
        logger.info(f"Invalidated {invalidated_count} cache entries matching pattern: {pattern}")
        return invalidated_count
    
    async def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate cache entries with specified tags."""
        
        invalidated_count = 0
        
        # Check memory cache
        memory_keys = await self.memory_cache.keys()
        for key in memory_keys:
            entry = await self.memory_cache.get(key)
            if entry and entry.tags.intersection(tags):
                await self.delete(key)
                invalidated_count += 1
        
        logger.info(f"Invalidated {invalidated_count} cache entries with tags: {tags}")
        return invalidated_count
    
    async def warm_cache(self, keys: List[str], loader_func: callable):
        """Warm cache with specified keys using loader function."""
        
        if not self.enable_cache_warming:
            return
        
        self._warm_keys.update(keys)
        
        for key in keys:
            try:
                # Check if already cached
                if await self.get(key) is not None:
                    continue
                
                # Load and cache value
                value = await loader_func(key)
                if value is not None:
                    await self.set(key, value)
                    
            except Exception as e:
                logger.error(f"Cache warming failed for key {key}: {e}")
    
    async def get_stats(self) -> Dict[str, CacheStats]:
        """Get comprehensive cache statistics."""
        
        stats = {}
        
        stats['memory'] = self.memory_cache.get_stats()
        
        if self.redis_cache:
            stats['redis'] = self.redis_cache.get_stats()
        
        stats['file'] = self.file_cache.get_stats()
        
        # Calculate combined stats
        combined = CacheStats()
        for cache_stats in stats.values():
            combined.hits += cache_stats.hits
            combined.misses += cache_stats.misses
            combined.evictions += cache_stats.evictions
            combined.size += cache_stats.size
            combined.memory_usage_bytes += cache_stats.memory_usage_bytes
        
        stats['combined'] = combined
        
        return stats
    
    async def optimize_cache(self):
        """Optimize cache performance based on usage patterns."""
        
        try:
            stats = await self.get_stats()
            
            # Log performance metrics
            for level, cache_stats in stats.items():
                if level != 'combined':
                    logger.info(
                        f"Cache {level} stats: "
                        f"hit_rate={cache_stats.hit_rate:.2%}, "
                        f"size={cache_stats.size}, "
                        f"memory_usage_mb={cache_stats.memory_usage_bytes / (1024*1024):.1f}"
                    )
                    
        except Exception as e:
            logger.error(f"Cache optimization error: {e}")


# Convenience functions and decorators

def cache_result(
    ttl: Optional[int] = None,
    cache_level: CacheLevel = CacheLevel.ALL,
    key_func: Optional[callable] = None,
):
    """Decorator to cache function results."""
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cached_value = await intelligent_cache.get(cache_key, cache_level)
            if cached_value is not None:
                return cached_value
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await intelligent_cache.set(cache_key, result, ttl, cache_level)
            
            return result
        
        return wrapper
    return decorator


# Global cache instance
intelligent_cache = IntelligentCache()


# Cache management functions

async def get_cache_stats() -> Dict[str, Any]:
    """Get comprehensive cache statistics."""
    return await intelligent_cache.get_stats()


async def clear_all_caches():
    """Clear all cache levels."""
    await intelligent_cache.clear()


async def invalidate_cache_pattern(pattern: str) -> int:
    """Invalidate cache entries matching pattern."""
    return await intelligent_cache.invalidate_by_pattern(pattern)


async def warm_cache_keys(keys: List[str], loader_func: callable):
    """Warm cache with specified keys."""
    await intelligent_cache.warm_cache(keys, loader_func)


async def optimize_cache_performance():
    """Optimize cache performance."""
    await intelligent_cache.optimize_cache()