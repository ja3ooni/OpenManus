"""
Tool Marketplace and Discovery Interface for finding, installing, and managing tools.
"""

import asyncio
import hashlib
import json
import tempfile
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

import aiohttp

from app.logger import logger
from app.tool.dynamic_manager import DynamicToolManager, ToolDependency, ToolMetadata


class ToolSource(Enum):
    """Sources for tool discovery."""

    LOCAL = "local"
    REMOTE_REGISTRY = "remote_registry"
    GIT_REPOSITORY = "git_repository"
    FILE_SYSTEM = "file_system"
    PACKAGE_INDEX = "package_index"


@dataclass
class MarketplaceEntry:
    """Entry in the tool marketplace."""

    name: str
    version: str
    description: str
    author: str
    source: ToolSource
    source_url: str
    download_url: Optional[str] = None
    checksum: Optional[str] = None
    size: Optional[int] = None
    downloads: int = 0
    rating: float = 0.0
    tags: List[str] = field(default_factory=list)
    category: str = "general"
    license: str = ""
    homepage: Optional[str] = None
    documentation: Optional[str] = None
    dependencies: List[ToolDependency] = field(default_factory=list)
    compatibility: Dict[str, str] = field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    verified: bool = False


@dataclass
class InstallationResult:
    """Result of a tool installation."""

    success: bool
    tool_name: str
    version: str
    message: str
    installed_files: List[str] = field(default_factory=list)
    error_details: Optional[str] = None


class ToolMarketplace:
    """
    Tool marketplace for discovering, installing, and managing tools
    from various sources including remote registries and repositories.
    """

    def __init__(self, tool_manager: DynamicToolManager):
        self.tool_manager = tool_manager
        self.registries: List[str] = []
        self.cache_dir = Path.home() / ".openmanus" / "tool_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.marketplace_cache: Dict[str, MarketplaceEntry] = {}
        self.installed_tools: Set[str] = set()
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    def add_registry(self, registry_url: str) -> None:
        """Add a tool registry URL."""
        if registry_url not in self.registries:
            self.registries.append(registry_url)
            logger.info(f"Added tool registry: {registry_url}")

    async def refresh_marketplace(self) -> None:
        """Refresh the marketplace cache from all registries."""
        logger.info("Refreshing tool marketplace")
        self.marketplace_cache.clear()

        # Discover from registries
        for registry_url in self.registries:
            try:
                await self._discover_from_registry(registry_url)
            except Exception as e:
                logger.error(f"Error refreshing from registry {registry_url}: {e}")

        # Discover from local directories
        await self._discover_local_tools()

        logger.info(f"Marketplace refreshed with {len(self.marketplace_cache)} tools")

    async def _discover_from_registry(self, registry_url: str) -> None:
        """Discover tools from a remote registry."""
        if not self.session:
            return

        try:
            async with self.session.get(f"{registry_url}/tools") as response:
                if response.status == 200:
                    data = await response.json()

                    for tool_data in data.get("tools", []):
                        entry = self._parse_marketplace_entry(
                            tool_data, ToolSource.REMOTE_REGISTRY
                        )
                        if entry:
                            self.marketplace_cache[entry.name] = entry

        except Exception as e:
            logger.error(f"Error discovering from registry {registry_url}: {e}")

    async def _discover_local_tools(self) -> None:
        """Discover tools from local directories."""
        for directory in self.tool_manager.tool_directories:
            try:
                for tool_file in directory.rglob("*.py"):
                    if tool_file.name.startswith("_"):
                        continue

                    # Try to extract metadata
                    metadata = await self._extract_local_metadata(tool_file)
                    if metadata:
                        entry = MarketplaceEntry(
                            name=metadata.name,
                            version=metadata.version,
                            description=metadata.description,
                            author=metadata.author,
                            source=ToolSource.LOCAL,
                            source_url=str(tool_file),
                            category=metadata.category,
                            tags=metadata.tags,
                            dependencies=metadata.dependencies,
                            verified=True,  # Local tools are considered verified
                        )
                        self.marketplace_cache[entry.name] = entry

            except Exception as e:
                logger.error(f"Error discovering local tools from {directory}: {e}")

    async def _extract_local_metadata(self, tool_file: Path) -> Optional[ToolMetadata]:
        """Extract metadata from a local tool file."""
        try:
            # This would need to parse the Python file to extract metadata
            # For now, return a basic metadata structure
            return ToolMetadata(
                name=tool_file.stem,
                version="1.0.0",
                description=f"Local tool from {tool_file.name}",
                author="Local",
                category="local",
            )
        except Exception as e:
            logger.error(f"Error extracting metadata from {tool_file}: {e}")
            return None

    def _parse_marketplace_entry(
        self, data: Dict[str, Any], source: ToolSource
    ) -> Optional[MarketplaceEntry]:
        """Parse marketplace entry from JSON data."""
        try:
            dependencies = []
            for dep_data in data.get("dependencies", []):
                dependencies.append(
                    ToolDependency(
                        name=dep_data["name"],
                        version_constraint=dep_data.get("version_constraint", "*"),
                        optional=dep_data.get("optional", False),
                        description=dep_data.get("description", ""),
                    )
                )

            return MarketplaceEntry(
                name=data["name"],
                version=data["version"],
                description=data["description"],
                author=data.get("author", ""),
                source=source,
                source_url=data["source_url"],
                download_url=data.get("download_url"),
                checksum=data.get("checksum"),
                size=data.get("size"),
                downloads=data.get("downloads", 0),
                rating=data.get("rating", 0.0),
                tags=data.get("tags", []),
                category=data.get("category", "general"),
                license=data.get("license", ""),
                homepage=data.get("homepage"),
                documentation=data.get("documentation"),
                dependencies=dependencies,
                compatibility=data.get("compatibility", {}),
                created_at=data.get("created_at"),
                updated_at=data.get("updated_at"),
                verified=data.get("verified", False),
            )
        except KeyError as e:
            logger.error(f"Missing required field in marketplace entry: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing marketplace entry: {e}")
            return None

    async def search_tools(
        self,
        query: str = "",
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source: Optional[ToolSource] = None,
    ) -> List[MarketplaceEntry]:
        """Search for tools in the marketplace."""
        if not self.marketplace_cache:
            await self.refresh_marketplace()

        results = []
        query_lower = query.lower()

        for entry in self.marketplace_cache.values():
            # Filter by source
            if source and entry.source != source:
                continue

            # Filter by category
            if category and entry.category != category:
                continue

            # Filter by tags
            if tags and not any(tag in entry.tags for tag in tags):
                continue

            # Filter by query
            if query and not (
                query_lower in entry.name.lower()
                or query_lower in entry.description.lower()
                or any(query_lower in tag.lower() for tag in entry.tags)
            ):
                continue

            results.append(entry)

        # Sort by relevance (downloads, rating, etc.)
        results.sort(key=lambda x: (x.rating, x.downloads, x.name), reverse=True)
        return results

    async def get_tool_details(self, tool_name: str) -> Optional[MarketplaceEntry]:
        """Get detailed information about a specific tool."""
        if not self.marketplace_cache:
            await self.refresh_marketplace()

        return self.marketplace_cache.get(tool_name)

    async def install_tool(
        self, tool_name: str, version: Optional[str] = None
    ) -> InstallationResult:
        """Install a tool from the marketplace."""
        entry = await self.get_tool_details(tool_name)
        if not entry:
            return InstallationResult(
                success=False,
                tool_name=tool_name,
                version=version or "unknown",
                message=f"Tool {tool_name} not found in marketplace",
            )

        if version and entry.version != version:
            return InstallationResult(
                success=False,
                tool_name=tool_name,
                version=version,
                message=f"Version {version} not available for {tool_name}",
            )

        try:
            # Check dependencies
            for dep in entry.dependencies:
                if not dep.optional:
                    dep_entry = await self.get_tool_details(dep.name)
                    if not dep_entry:
                        return InstallationResult(
                            success=False,
                            tool_name=tool_name,
                            version=entry.version,
                            message=f"Required dependency {dep.name} not found",
                        )

            # Install based on source
            if entry.source == ToolSource.LOCAL:
                return await self._install_local_tool(entry)
            elif entry.source == ToolSource.REMOTE_REGISTRY:
                return await self._install_remote_tool(entry)
            else:
                return InstallationResult(
                    success=False,
                    tool_name=tool_name,
                    version=entry.version,
                    message=f"Unsupported source type: {entry.source}",
                )

        except Exception as e:
            logger.error(f"Error installing tool {tool_name}: {e}")
            return InstallationResult(
                success=False,
                tool_name=tool_name,
                version=entry.version,
                message=f"Installation failed: {str(e)}",
                error_details=str(e),
            )

    async def _install_local_tool(self, entry: MarketplaceEntry) -> InstallationResult:
        """Install a local tool."""
        try:
            # For local tools, just register with the tool manager
            tool_file = Path(entry.source_url)
            if not tool_file.exists():
                return InstallationResult(
                    success=False,
                    tool_name=entry.name,
                    version=entry.version,
                    message=f"Local tool file not found: {tool_file}",
                )

            # The tool manager will handle discovery and registration
            await self.tool_manager.scan_for_tools()

            self.installed_tools.add(entry.name)
            return InstallationResult(
                success=True,
                tool_name=entry.name,
                version=entry.version,
                message=f"Local tool {entry.name} registered successfully",
                installed_files=[str(tool_file)],
            )

        except Exception as e:
            return InstallationResult(
                success=False,
                tool_name=entry.name,
                version=entry.version,
                message=f"Failed to install local tool: {str(e)}",
                error_details=str(e),
            )

    async def _install_remote_tool(self, entry: MarketplaceEntry) -> InstallationResult:
        """Install a remote tool."""
        if not entry.download_url or not self.session:
            return InstallationResult(
                success=False,
                tool_name=entry.name,
                version=entry.version,
                message="No download URL available",
            )

        try:
            # Download the tool
            async with self.session.get(entry.download_url) as response:
                if response.status != 200:
                    return InstallationResult(
                        success=False,
                        tool_name=entry.name,
                        version=entry.version,
                        message=f"Download failed with status {response.status}",
                    )

                content = await response.read()

                # Verify checksum if provided
                if entry.checksum:
                    actual_checksum = hashlib.sha256(content).hexdigest()
                    if actual_checksum != entry.checksum:
                        return InstallationResult(
                            success=False,
                            tool_name=entry.name,
                            version=entry.version,
                            message="Checksum verification failed",
                        )

                # Save to cache and extract
                cache_file = self.cache_dir / f"{entry.name}-{entry.version}.zip"
                cache_file.write_bytes(content)

                # Extract and install
                install_dir = self.cache_dir / entry.name / entry.version
                install_dir.mkdir(parents=True, exist_ok=True)

                installed_files = []
                with zipfile.ZipFile(cache_file, "r") as zip_ref:
                    zip_ref.extractall(install_dir)
                    installed_files = [
                        str(install_dir / name) for name in zip_ref.namelist()
                    ]

                # Register with tool manager
                await self.tool_manager.add_tool_directory(install_dir)
                await self.tool_manager.scan_for_tools()

                self.installed_tools.add(entry.name)
                return InstallationResult(
                    success=True,
                    tool_name=entry.name,
                    version=entry.version,
                    message=f"Tool {entry.name} installed successfully",
                    installed_files=installed_files,
                )

        except Exception as e:
            return InstallationResult(
                success=False,
                tool_name=entry.name,
                version=entry.version,
                message=f"Installation failed: {str(e)}",
                error_details=str(e),
            )

    async def uninstall_tool(self, tool_name: str) -> InstallationResult:
        """Uninstall a tool."""
        try:
            # Unload from tool manager
            await self.tool_manager.unload_tool(tool_name)

            # Remove from installed set
            self.installed_tools.discard(tool_name)

            # Clean up cache files
            cache_pattern = self.cache_dir.glob(f"{tool_name}-*")
            removed_files = []
            for cache_file in cache_pattern:
                if cache_file.is_file():
                    cache_file.unlink()
                    removed_files.append(str(cache_file))
                elif cache_file.is_dir():
                    import shutil

                    shutil.rmtree(cache_file)
                    removed_files.append(str(cache_file))

            return InstallationResult(
                success=True,
                tool_name=tool_name,
                version="unknown",
                message=f"Tool {tool_name} uninstalled successfully",
                installed_files=removed_files,
            )

        except Exception as e:
            return InstallationResult(
                success=False,
                tool_name=tool_name,
                version="unknown",
                message=f"Uninstallation failed: {str(e)}",
                error_details=str(e),
            )

    def list_installed_tools(self) -> List[str]:
        """List all installed tools."""
        return list(self.installed_tools)

    def get_categories(self) -> List[str]:
        """Get all available categories."""
        if not self.marketplace_cache:
            return []

        categories = set()
        for entry in self.marketplace_cache.values():
            categories.add(entry.category)

        return sorted(list(categories))

    def get_popular_tools(self, limit: int = 10) -> List[MarketplaceEntry]:
        """Get popular tools based on downloads and ratings."""
        if not self.marketplace_cache:
            return []

        tools = list(self.marketplace_cache.values())
        tools.sort(key=lambda x: (x.downloads, x.rating), reverse=True)
        return tools[:limit]

    async def check_updates(self) -> List[Dict[str, str]]:
        """Check for updates to installed tools."""
        updates = []

        for tool_name in self.installed_tools:
            current_info = self.tool_manager.get_tool_info(tool_name)
            marketplace_entry = await self.get_tool_details(tool_name)

            if current_info and marketplace_entry:
                current_version = current_info["version"]
                latest_version = marketplace_entry.version

                # Simple version comparison - can be enhanced
                if current_version != latest_version:
                    updates.append(
                        {
                            "name": tool_name,
                            "current_version": current_version,
                            "latest_version": latest_version,
                            "description": marketplace_entry.description,
                        }
                    )

        return updates

    async def cleanup(self) -> None:
        """Cleanup marketplace resources."""
        if self.session:
            await self.session.close()
        logger.info("Tool marketplace cleanup complete")
