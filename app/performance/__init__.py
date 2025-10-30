"""
Performance optimization module for OpenManus.

This module provides caching, resource management, and performance monitoring
capabilities to ensure optimal system performance and scalability.
"""

from .cache import (
    CacheLevel,
    CacheStats,
    FileCache,
    IntelligentCache,
    MemoryCache,
    RedisCache,
    cache_result,
    clear_all_caches,
    get_cache_stats,
    intelligent_cache,
    invalidate_cache_pattern,
    optimize_cache_performance,
    warm_cache_keys,
)

# from .metrics import MetricsCollector, PerformanceMetrics
# from .pool_manager import AgentPoolManager, RequestQueue
# Temporarily disable resource_manager imports due to file issues
# from .resource_manager import (
#     Priority,
#     ResourceAllocation,
#     ResourceContext,
#     ResourceLimits,
#     ResourceManager,
#     ResourceMonitor,
#     ResourceStatus,
#     ResourceType,
#     ResourceUsage,
#     allocate_connections,
#     allocate_memory,
#     deallocate_resource,
#     force_cleanup,
#     get_resource_statistics,
#     get_resource_usage,
#     global_resource_manager,
#     manage_resources,
# )

__all__ = [
    # Cache system
    "IntelligentCache",
    "MemoryCache",
    "RedisCache",
    "FileCache",
    "CacheLevel",
    "CacheStats",
    "cache_result",
    "get_cache_stats",
    "clear_all_caches",
    "invalidate_cache_pattern",
    "warm_cache_keys",
    "optimize_cache_performance",
    "intelligent_cache",
    # Resource management (temporarily disabled)
    # "ResourceManager",
    # "ResourceUsage",
    # "ResourceAllocation",
    # "ResourceLimits",
    # "ResourceMonitor",
    # "ResourceContext",
    # "ResourceType",
    # "Priority",
    # "ResourceStatus",
    # "manage_resources",
    # "get_resource_usage",
    # "get_resource_statistics",
    # "allocate_memory",
    # "allocate_connections",
    # "deallocate_resource",
    # "force_cleanup",
    # "global_resource_manager",
    # Performance monitoring (not implemented yet)
    # "PerformanceMetrics",
    # "MetricsCollector",
    # Agent management (not implemented yet)
    # "AgentPoolManager",
    # "RequestQueue",
]
