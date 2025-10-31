"""
Configuration package for OpenManus.

This package provides comprehensive configuration management including:
- Configuration validation and schema enforcement
- Environment-specific configuration overlays
- Configuration versioning and rollback capabilities
- Hot reloading with change detection
"""

# Avoid circular imports by not importing modules that depend on app.config
# These can be imported directly when needed

__all__ = [
    "ConfigValidator",
    "ConfigVersionManager",
    "EnvironmentManager",
    "ConfigHotReloader",
    "hot_reloader",
]
