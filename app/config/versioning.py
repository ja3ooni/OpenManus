"""
Configuration versioning and rollback system for OpenManus.

This module provides configuration versioning, backup, and rollback
capabilities for safe configuration management in production environments.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.logger import logger


class ConfigVersion(BaseModel):
    """Represents a configuration version"""

    version_id: str = Field(..., description="Unique version identifier")
    timestamp: datetime = Field(..., description="Version creation timestamp")
    description: str = Field("", description="Version description")
    author: Optional[str] = Field(None, description="Version author")
    config_hash: str = Field(..., description="Configuration content hash")
    file_path: Path = Field(..., description="Path to versioned config file")
    is_active: bool = Field(False, description="Whether this is the active version")
    tags: List[str] = Field(default_factory=list, description="Version tags")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ConfigVersionManager:
    """Manages configuration versions and rollbacks"""

    def __init__(self, config_dir: Path, versions_dir: Optional[Path] = None):
        self.config_dir = Path(config_dir)
        self.versions_dir = versions_dir or (self.config_dir / "versions")
        self.versions_dir.mkdir(exist_ok=True)

        self.versions_index_file = self.versions_dir / "versions.json"
        self.versions: List[ConfigVersion] = self._load_versions_index()

        # Ensure we have a current version
        self._ensure_current_version()

    def _load_versions_index(self) -> List[ConfigVersion]:
        """Load versions index from file"""
        if not self.versions_index_file.exists():
            return []

        try:
            with open(self.versions_index_file, "r") as f:
                versions_data = json.load(f)

            versions = []
            for version_data in versions_data:
                # Convert timestamp string back to datetime
                if isinstance(version_data["timestamp"], str):
                    version_data["timestamp"] = datetime.fromisoformat(
                        version_data["timestamp"]
                    )

                # Convert file_path string back to Path
                if isinstance(version_data["file_path"], str):
                    version_data["file_path"] = Path(version_data["file_path"])

                versions.append(ConfigVersion(**version_data))

            return versions

        except Exception as e:
            logger.error(f"Failed to load versions index: {e}")
            return []

    def _save_versions_index(self):
        """Save versions index to file"""
        try:
            versions_data = []
            for version in self.versions:
                version_dict = version.dict()
                # Convert datetime to string for JSON serialization
                version_dict["timestamp"] = version.timestamp.isoformat()
                # Convert Path to string for JSON serialization
                version_dict["file_path"] = str(version.file_path)
                versions_data.append(version_dict)

            with open(self.versions_index_file, "w") as f:
                json.dump(versions_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save versions index: {e}")

    def _ensure_current_version(self):
        """Ensure we have a current version of the configuration"""
        current_config_file = self.config_dir / "config.toml"

        if current_config_file.exists() and not any(v.is_active for v in self.versions):
            # Create initial version from current config
            self.create_version(
                description="Initial version", author="system", tags=["initial"]
            )

    def _calculate_config_hash(self, config_file: Path) -> str:
        """Calculate hash of configuration file"""
        import hashlib

        if not config_file.exists():
            return ""

        with open(config_file, "rb") as f:
            content = f.read()

        return hashlib.sha256(content).hexdigest()

    def _generate_version_id(self) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now(timezone.utc)
        return f"v{timestamp.strftime('%Y%m%d_%H%M%S')}"

    def create_version(
        self,
        description: str = "",
        author: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> ConfigVersion:
        """Create a new configuration version"""
        current_config_file = self.config_dir / "config.toml"

        if not current_config_file.exists():
            raise FileNotFoundError("No current configuration file found")

        version_id = self._generate_version_id()
        config_hash = self._calculate_config_hash(current_config_file)

        # Check if this configuration already exists
        existing_version = self._find_version_by_hash(config_hash)
        if existing_version:
            logger.info(
                f"Configuration unchanged, reusing version {existing_version.version_id}"
            )
            return existing_version

        # Create version file
        version_file = self.versions_dir / f"{version_id}.toml"
        shutil.copy2(current_config_file, version_file)

        # Mark previous versions as inactive
        for version in self.versions:
            version.is_active = False

        # Create version record
        version = ConfigVersion(
            version_id=version_id,
            timestamp=datetime.now(timezone.utc),
            description=description,
            author=author,
            config_hash=config_hash,
            file_path=version_file,
            is_active=True,
            tags=tags or [],
            metadata={
                "original_file": str(current_config_file),
                "file_size": current_config_file.stat().st_size,
            },
        )

        self.versions.append(version)
        self._save_versions_index()

        logger.info(f"Created configuration version {version_id}: {description}")
        return version

    def _find_version_by_hash(self, config_hash: str) -> Optional[ConfigVersion]:
        """Find version by configuration hash"""
        for version in self.versions:
            if version.config_hash == config_hash:
                return version
        return None

    def get_version(self, version_id: str) -> Optional[ConfigVersion]:
        """Get version by ID"""
        for version in self.versions:
            if version.version_id == version_id:
                return version
        return None

    def get_active_version(self) -> Optional[ConfigVersion]:
        """Get currently active version"""
        for version in self.versions:
            if version.is_active:
                return version
        return None

    def list_versions(self, limit: Optional[int] = None) -> List[ConfigVersion]:
        """List all versions, most recent first"""
        sorted_versions = sorted(self.versions, key=lambda v: v.timestamp, reverse=True)
        if limit:
            return sorted_versions[:limit]
        return sorted_versions

    def rollback_to_version(self, version_id: str, backup_current: bool = True) -> bool:
        """Rollback to a specific version"""
        target_version = self.get_version(version_id)
        if not target_version:
            logger.error(f"Version {version_id} not found")
            return False

        if not target_version.file_path.exists():
            logger.error(f"Version file {target_version.file_path} not found")
            return False

        current_config_file = self.config_dir / "config.toml"

        try:
            # Backup current configuration if requested
            if backup_current and current_config_file.exists():
                self.create_version(
                    description=f"Backup before rollback to {version_id}",
                    author="system",
                    tags=["rollback-backup"],
                )

            # Copy version file to current config
            shutil.copy2(target_version.file_path, current_config_file)

            # Update active version
            for version in self.versions:
                version.is_active = version.version_id == version_id

            self._save_versions_index()

            logger.info(f"Successfully rolled back to version {version_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to rollback to version {version_id}: {e}")
            return False

    def rollback_to_previous(self) -> bool:
        """Rollback to the previous version"""
        active_version = self.get_active_version()
        if not active_version:
            logger.error("No active version found")
            return False

        # Find previous version
        sorted_versions = sorted(self.versions, key=lambda v: v.timestamp, reverse=True)
        active_index = next(
            (
                i
                for i, v in enumerate(sorted_versions)
                if v.version_id == active_version.version_id
            ),
            -1,
        )

        if active_index == -1 or active_index >= len(sorted_versions) - 1:
            logger.error("No previous version available")
            return False

        previous_version = sorted_versions[active_index + 1]
        return self.rollback_to_version(previous_version.version_id)

    def delete_version(self, version_id: str, force: bool = False) -> bool:
        """Delete a configuration version"""
        version = self.get_version(version_id)
        if not version:
            logger.error(f"Version {version_id} not found")
            return False

        if version.is_active and not force:
            logger.error("Cannot delete active version without force=True")
            return False

        try:
            # Remove version file
            if version.file_path.exists():
                version.file_path.unlink()

            # Remove from versions list
            self.versions = [v for v in self.versions if v.version_id != version_id]
            self._save_versions_index()

            logger.info(f"Deleted version {version_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete version {version_id}: {e}")
            return False

    def cleanup_old_versions(self, keep_count: int = 10) -> int:
        """Clean up old versions, keeping only the most recent ones"""
        if len(self.versions) <= keep_count:
            return 0

        # Sort versions by timestamp, keep most recent
        sorted_versions = sorted(self.versions, key=lambda v: v.timestamp, reverse=True)
        versions_to_delete = sorted_versions[keep_count:]

        deleted_count = 0
        for version in versions_to_delete:
            # Don't delete active version or tagged versions
            if not version.is_active and "keep" not in version.tags:
                if self.delete_version(version.version_id, force=False):
                    deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} old configuration versions")
        return deleted_count

    def tag_version(self, version_id: str, tags: List[str]) -> bool:
        """Add tags to a version"""
        version = self.get_version(version_id)
        if not version:
            return False

        version.tags.extend(tags)
        version.tags = list(set(version.tags))  # Remove duplicates
        self._save_versions_index()

        logger.info(f"Added tags {tags} to version {version_id}")
        return True

    def find_versions_by_tag(self, tag: str) -> List[ConfigVersion]:
        """Find versions by tag"""
        return [v for v in self.versions if tag in v.tags]

    def get_version_diff(self, version1_id: str, version2_id: str) -> Optional[str]:
        """Get diff between two versions"""
        version1 = self.get_version(version1_id)
        version2 = self.get_version(version2_id)

        if not version1 or not version2:
            return None

        if not version1.file_path.exists() or not version2.file_path.exists():
            return None

        try:
            import difflib

            with open(version1.file_path, "r") as f1:
                lines1 = f1.readlines()

            with open(version2.file_path, "r") as f2:
                lines2 = f2.readlines()

            diff = difflib.unified_diff(
                lines1,
                lines2,
                fromfile=f"Version {version1_id}",
                tofile=f"Version {version2_id}",
                lineterm="",
            )

            return "\n".join(diff)

        except Exception as e:
            logger.error(f"Failed to generate diff: {e}")
            return None

    def export_version(self, version_id: str, export_path: Path) -> bool:
        """Export a version to a file"""
        version = self.get_version(version_id)
        if not version or not version.file_path.exists():
            return False

        try:
            shutil.copy2(version.file_path, export_path)
            logger.info(f"Exported version {version_id} to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export version {version_id}: {e}")
            return False

    def import_version(
        self,
        import_path: Path,
        description: str = "Imported version",
        author: Optional[str] = None,
    ) -> Optional[ConfigVersion]:
        """Import a configuration version from a file"""
        if not import_path.exists():
            logger.error(f"Import file {import_path} not found")
            return None

        try:
            # Copy to temporary location first
            temp_config = self.config_dir / "config.toml.temp"
            shutil.copy2(import_path, temp_config)

            # Validate the configuration
            # TODO: Add validation here

            # Replace current config
            current_config = self.config_dir / "config.toml"
            if current_config.exists():
                current_config.unlink()

            temp_config.rename(current_config)

            # Create version
            return self.create_version(
                description=description, author=author, tags=["imported"]
            )

        except Exception as e:
            logger.error(f"Failed to import version: {e}")
            return None

    def get_version_summary(self) -> Dict[str, Any]:
        """Get summary of version management status"""
        active_version = self.get_active_version()

        return {
            "total_versions": len(self.versions),
            "active_version": active_version.version_id if active_version else None,
            "latest_version": self.versions[-1].version_id if self.versions else None,
            "versions_dir": str(self.versions_dir),
            "disk_usage_mb": sum(
                v.file_path.stat().st_size
                for v in self.versions
                if v.file_path.exists()
            )
            / (1024 * 1024),
            "tags_in_use": list(
                set(tag for version in self.versions for tag in version.tags)
            ),
        }
