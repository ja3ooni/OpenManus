"""
Dynamic Tool Management System with hot-loading capabilities,
dependency resolution, and performance monitoring.
"""

import asyncio
import importlib
import inspect
import json
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type

from packaging import version

from app.logger import logger
from app.tool.base import BaseTool, ToolResult


class ToolStatus(Enum):
    """Status of a tool in the system."""

    AVAILABLE = "available"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    DISABLED = "disabled"
    DEPRECATED = "deprecated"


@dataclass
class ToolDependency:
    """Represents a tool dependency."""

    name: str
    version_constraint: str = "*"
    optional: bool = False
    description: str = ""


@dataclass
class ToolMetadata:
    """Metadata for a tool."""

    name: str
    version: str
    description: str
    author: str = ""
    dependencies: List[ToolDependency] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    category: str = "general"
    min_python_version: str = "3.8"
    max_python_version: str = ""
    deprecated: bool = False
    deprecation_message: str = ""
    created_at: Optional[float] = None
    updated_at: Optional[float] = None


@dataclass
class ToolPerformanceMetrics:
    """Performance metrics for a tool."""

    tool_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    min_execution_time: float = float("inf")
    max_execution_time: float = 0.0
    last_called: Optional[float] = None
    error_rate: float = 0.0


@dataclass
class ToolRegistration:
    """Registration information for a tool."""

    tool_class: Type[BaseTool]
    metadata: ToolMetadata
    status: ToolStatus = ToolStatus.AVAILABLE
    instance: Optional[BaseTool] = None
    load_time: Optional[float] = None
    error_message: Optional[str] = None
    performance_metrics: ToolPerformanceMetrics = field(
        default_factory=lambda: ToolPerformanceMetrics("")
    )


class DynamicToolManager:
    """
    Dynamic tool management system with hot-loading, dependency resolution,
    and performance monitoring capabilities.
    """

    def __init__(self):
        self.registered_tools: Dict[str, ToolRegistration] = {}
        self.loaded_tools: Dict[str, BaseTool] = {}
        self.tool_directories: List[Path] = []
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.reverse_dependency_graph: Dict[str, Set[str]] = {}
        self.performance_metrics: Dict[str, ToolPerformanceMetrics] = {}
        self.tool_callbacks: List[Callable[[str, ToolStatus], None]] = []
        self.auto_reload: bool = True
        self.file_watchers: Dict[str, Any] = {}

    def add_tool_callback(self, callback: Callable[[str, ToolStatus], None]) -> None:
        """Add a callback for tool status changes."""
        self.tool_callbacks.append(callback)

    def add_tool_directory(self, directory: Path) -> None:
        """Add a directory to scan for tools."""
        if directory.exists() and directory.is_dir():
            self.tool_directories.append(directory)
            logger.info(f"Added tool directory: {directory}")
        else:
            logger.warning(f"Tool directory does not exist: {directory}")

    async def scan_for_tools(self) -> List[str]:
        """Scan registered directories for new tools."""
        discovered_tools = []

        for directory in self.tool_directories:
            try:
                for tool_file in directory.rglob("*.py"):
                    if tool_file.name.startswith("_"):
                        continue

                    tool_name = self._extract_tool_name(tool_file)
                    if tool_name and tool_name not in self.registered_tools:
                        try:
                            await self._discover_tool(tool_file)
                            discovered_tools.append(tool_name)
                        except Exception as e:
                            logger.error(f"Error discovering tool {tool_name}: {e}")

            except Exception as e:
                logger.error(f"Error scanning directory {directory}: {e}")

        return discovered_tools

    def _extract_tool_name(self, tool_file: Path) -> Optional[str]:
        """Extract tool name from file path."""
        # Simple heuristic: use filename without extension
        return tool_file.stem if tool_file.suffix == ".py" else None

    async def _discover_tool(self, tool_file: Path) -> None:
        """Discover and register a tool from a file."""
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(tool_file.stem, tool_file)
            if not spec or not spec.loader:
                return

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find tool classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, BaseTool)
                    and obj != BaseTool
                    and not name.startswith("_")
                ):

                    # Extract metadata
                    metadata = self._extract_metadata(obj, tool_file)

                    # Register the tool
                    registration = ToolRegistration(
                        tool_class=obj, metadata=metadata, status=ToolStatus.AVAILABLE
                    )

                    self.registered_tools[metadata.name] = registration
                    logger.info(f"Discovered tool: {metadata.name}")

        except Exception as e:
            logger.error(f"Error discovering tool from {tool_file}: {e}")

    def _extract_metadata(
        self, tool_class: Type[BaseTool], tool_file: Path
    ) -> ToolMetadata:
        """Extract metadata from a tool class."""
        # Try to get metadata from class attributes or docstring
        name = getattr(tool_class, "name", tool_class.__name__.lower())
        description = getattr(tool_class, "description", tool_class.__doc__ or "")
        version = getattr(tool_class, "version", "1.0.0")
        author = getattr(tool_class, "author", "")
        category = getattr(tool_class, "category", "general")

        # Extract dependencies if defined
        dependencies = []
        if hasattr(tool_class, "dependencies"):
            for dep in tool_class.dependencies:
                if isinstance(dep, dict):
                    dependencies.append(ToolDependency(**dep))
                elif isinstance(dep, str):
                    dependencies.append(ToolDependency(name=dep))

        return ToolMetadata(
            name=name,
            version=version,
            description=description.strip(),
            author=author,
            dependencies=dependencies,
            category=category,
            created_at=time.time(),
        )

    async def register_tool(
        self, tool_class: Type[BaseTool], metadata: Optional[ToolMetadata] = None
    ) -> bool:
        """Register a tool class with optional metadata."""
        try:
            if not metadata:
                metadata = self._extract_metadata(tool_class, Path(""))

            # Check for conflicts
            if metadata.name in self.registered_tools:
                existing = self.registered_tools[metadata.name]
                if not self._is_version_compatible(
                    existing.metadata.version, metadata.version
                ):
                    logger.warning(
                        f"Version conflict for tool {metadata.name}: "
                        f"existing {existing.metadata.version} vs new {metadata.version}"
                    )
                    return False

            # Validate dependencies
            if not await self._validate_dependencies(metadata):
                logger.error(f"Dependency validation failed for tool {metadata.name}")
                return False

            # Register the tool
            registration = ToolRegistration(
                tool_class=tool_class, metadata=metadata, status=ToolStatus.AVAILABLE
            )

            self.registered_tools[metadata.name] = registration
            self._update_dependency_graph(metadata)

            # Initialize performance metrics
            self.performance_metrics[metadata.name] = ToolPerformanceMetrics(
                tool_name=metadata.name
            )

            await self._notify_tool_status_change(metadata.name, ToolStatus.AVAILABLE)
            logger.info(f"Registered tool: {metadata.name} v{metadata.version}")
            return True

        except Exception as e:
            logger.error(
                f"Error registering tool {metadata.name if metadata else 'unknown'}: {e}"
            )
            return False

    async def load_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Load a tool instance."""
        if tool_name not in self.registered_tools:
            logger.error(f"Tool {tool_name} not registered")
            return None

        registration = self.registered_tools[tool_name]

        if registration.status == ToolStatus.LOADED and registration.instance:
            return registration.instance

        if registration.status == ToolStatus.ERROR:
            logger.error(
                f"Tool {tool_name} is in error state: {registration.error_message}"
            )
            return None

        try:
            await self._notify_tool_status_change(tool_name, ToolStatus.LOADING)

            # Load dependencies first
            for dep in registration.metadata.dependencies:
                if not dep.optional:
                    dep_tool = await self.load_tool(dep.name)
                    if not dep_tool:
                        raise Exception(
                            f"Required dependency {dep.name} could not be loaded"
                        )

            # Create tool instance
            start_time = time.time()
            tool_instance = registration.tool_class()

            # Initialize if needed
            if hasattr(tool_instance, "initialize"):
                await tool_instance.initialize()

            registration.instance = tool_instance
            registration.load_time = time.time() - start_time
            registration.status = ToolStatus.LOADED

            self.loaded_tools[tool_name] = tool_instance

            await self._notify_tool_status_change(tool_name, ToolStatus.LOADED)
            logger.info(f"Loaded tool: {tool_name} in {registration.load_time:.3f}s")

            return tool_instance

        except Exception as e:
            error_msg = f"Error loading tool {tool_name}: {e}"
            logger.error(error_msg)

            registration.status = ToolStatus.ERROR
            registration.error_message = str(e)

            await self._notify_tool_status_change(tool_name, ToolStatus.ERROR)
            return None

    async def unload_tool(self, tool_name: str) -> bool:
        """Unload a tool instance."""
        if tool_name not in self.registered_tools:
            return False

        registration = self.registered_tools[tool_name]

        try:
            # Check for dependent tools
            dependents = self.reverse_dependency_graph.get(tool_name, set())
            if dependents:
                logger.warning(f"Tool {tool_name} has dependents: {dependents}")
                # Optionally unload dependents first

            # Cleanup tool instance
            if registration.instance and hasattr(registration.instance, "cleanup"):
                await registration.instance.cleanup()

            registration.instance = None
            registration.status = ToolStatus.AVAILABLE

            self.loaded_tools.pop(tool_name, None)

            await self._notify_tool_status_change(tool_name, ToolStatus.AVAILABLE)
            logger.info(f"Unloaded tool: {tool_name}")
            return True

        except Exception as e:
            logger.error(f"Error unloading tool {tool_name}: {e}")
            return False

    async def reload_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Reload a tool (unload and load again)."""
        await self.unload_tool(tool_name)
        return await self.load_tool(tool_name)

    async def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool with performance monitoring."""
        if tool_name not in self.loaded_tools:
            tool = await self.load_tool(tool_name)
            if not tool:
                return ToolResult(error=f"Tool {tool_name} could not be loaded")
        else:
            tool = self.loaded_tools[tool_name]

        # Performance monitoring
        metrics = self.performance_metrics.get(tool_name)
        if not metrics:
            metrics = ToolPerformanceMetrics(tool_name=tool_name)
            self.performance_metrics[tool_name] = metrics

        start_time = time.time()
        metrics.total_calls += 1
        metrics.last_called = start_time

        try:
            result = await tool.execute(**kwargs)

            # Update success metrics
            execution_time = time.time() - start_time
            metrics.successful_calls += 1
            metrics.total_execution_time += execution_time
            metrics.average_execution_time = (
                metrics.total_execution_time / metrics.total_calls
            )
            metrics.min_execution_time = min(metrics.min_execution_time, execution_time)
            metrics.max_execution_time = max(metrics.max_execution_time, execution_time)
            metrics.error_rate = metrics.failed_calls / metrics.total_calls

            return result

        except Exception as e:
            # Update error metrics
            execution_time = time.time() - start_time
            metrics.failed_calls += 1
            metrics.total_execution_time += execution_time
            metrics.average_execution_time = (
                metrics.total_execution_time / metrics.total_calls
            )
            metrics.error_rate = metrics.failed_calls / metrics.total_calls

            logger.error(f"Error executing tool {tool_name}: {e}")
            return ToolResult(error=str(e))

    async def _validate_dependencies(self, metadata: ToolMetadata) -> bool:
        """Validate tool dependencies."""
        for dep in metadata.dependencies:
            if dep.name not in self.registered_tools:
                if not dep.optional:
                    logger.error(
                        f"Required dependency {dep.name} not found for tool {metadata.name}"
                    )
                    return False
                else:
                    logger.warning(
                        f"Optional dependency {dep.name} not found for tool {metadata.name}"
                    )
                    continue

            # Check version compatibility
            dep_registration = self.registered_tools[dep.name]
            if not self._is_version_compatible(
                dep_registration.metadata.version, dep.version_constraint
            ):
                logger.error(
                    f"Version incompatibility: {dep.name} {dep_registration.metadata.version} "
                    f"does not satisfy {dep.version_constraint}"
                )
                return False

        return True

    def _is_version_compatible(self, current_version: str, constraint: str) -> bool:
        """Check if a version satisfies a constraint."""
        if constraint == "*":
            return True

        try:
            current = version.parse(current_version)
            # Simple constraint parsing - can be enhanced
            if constraint.startswith(">="):
                required = version.parse(constraint[2:])
                return current >= required
            elif constraint.startswith("<="):
                required = version.parse(constraint[2:])
                return current <= required
            elif constraint.startswith(">"):
                required = version.parse(constraint[1:])
                return current > required
            elif constraint.startswith("<"):
                required = version.parse(constraint[1:])
                return current < required
            elif constraint.startswith("=="):
                required = version.parse(constraint[2:])
                return current == required
            else:
                required = version.parse(constraint)
                return current == required
        except Exception:
            return True  # If parsing fails, assume compatible

    def _update_dependency_graph(self, metadata: ToolMetadata) -> None:
        """Update the dependency graph."""
        tool_name = metadata.name
        dependencies = {dep.name for dep in metadata.dependencies}

        self.dependency_graph[tool_name] = dependencies

        # Update reverse dependencies
        for dep_name in dependencies:
            if dep_name not in self.reverse_dependency_graph:
                self.reverse_dependency_graph[dep_name] = set()
            self.reverse_dependency_graph[dep_name].add(tool_name)

    async def _notify_tool_status_change(
        self, tool_name: str, status: ToolStatus
    ) -> None:
        """Notify callbacks about tool status changes."""
        for callback in self.tool_callbacks:
            try:
                callback(tool_name, status)
            except Exception as e:
                logger.error(f"Error in tool status callback: {e}")

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a tool."""
        if tool_name not in self.registered_tools:
            return None

        registration = self.registered_tools[tool_name]
        metrics = self.performance_metrics.get(tool_name)

        return {
            "name": registration.metadata.name,
            "version": registration.metadata.version,
            "description": registration.metadata.description,
            "author": registration.metadata.author,
            "category": registration.metadata.category,
            "status": registration.status.value,
            "dependencies": [
                {
                    "name": dep.name,
                    "version_constraint": dep.version_constraint,
                    "optional": dep.optional,
                }
                for dep in registration.metadata.dependencies
            ],
            "performance": {
                "total_calls": metrics.total_calls if metrics else 0,
                "successful_calls": metrics.successful_calls if metrics else 0,
                "failed_calls": metrics.failed_calls if metrics else 0,
                "error_rate": metrics.error_rate if metrics else 0.0,
                "average_execution_time": (
                    metrics.average_execution_time if metrics else 0.0
                ),
                "last_called": metrics.last_called if metrics else None,
            },
            "load_time": registration.load_time,
            "error_message": registration.error_message,
        }

    def list_tools(
        self,
        status_filter: Optional[ToolStatus] = None,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List tools with optional filtering."""
        tools = []

        for tool_name, registration in self.registered_tools.items():
            if status_filter and registration.status != status_filter:
                continue

            if category_filter and registration.metadata.category != category_filter:
                continue

            tool_info = self.get_tool_info(tool_name)
            if tool_info:
                tools.append(tool_info)

        return sorted(tools, key=lambda x: x["name"])

    def get_performance_report(self) -> Dict[str, Any]:
        """Get a comprehensive performance report."""
        total_calls = sum(m.total_calls for m in self.performance_metrics.values())
        total_errors = sum(m.failed_calls for m in self.performance_metrics.values())

        return {
            "summary": {
                "total_tools": len(self.registered_tools),
                "loaded_tools": len(self.loaded_tools),
                "total_calls": total_calls,
                "total_errors": total_errors,
                "overall_error_rate": (
                    total_errors / total_calls if total_calls > 0 else 0.0
                ),
            },
            "tools": [
                {
                    "name": name,
                    "calls": metrics.total_calls,
                    "success_rate": (
                        (metrics.successful_calls / metrics.total_calls)
                        if metrics.total_calls > 0
                        else 0.0
                    ),
                    "avg_execution_time": metrics.average_execution_time,
                    "last_called": metrics.last_called,
                }
                for name, metrics in self.performance_metrics.items()
                if metrics.total_calls > 0
            ],
        }

    async def cleanup(self) -> None:
        """Cleanup all loaded tools and resources."""
        logger.info("Cleaning up dynamic tool manager")

        # Unload all tools
        for tool_name in list(self.loaded_tools.keys()):
            await self.unload_tool(tool_name)

        # Clear all data structures
        self.registered_tools.clear()
        self.loaded_tools.clear()
        self.dependency_graph.clear()
        self.reverse_dependency_graph.clear()
        self.performance_metrics.clear()

        logger.info("Dynamic tool manager cleanup complete")
