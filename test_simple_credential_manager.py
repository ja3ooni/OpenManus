#!/usr/bin/env python3
"""Simple test for credential management functionality."""

import asyncio
import os
import tempfile
from pathlib import Path


# Simple credential manager implementation for testing
class SimpleCredentialManager:
    """Simple credential manager for testing."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.credentials = {}

    async def store_credential(self, name: str, value: str) -> bool:
        """Store a credential."""
        try:
            self.credentials[name] = {"value": value, "created": "2024-01-01T00:00:00"}
            return True
        except Exception:
            return False

    async def retrieve_credential(self, name: str) -> str:
        """Retrieve a credential."""
        if name in self.credentials:
            return self.credentials[name]["value"]
        return None

    async def delete_credential(self, name: str) -> bool:
        """Delete a credential."""
        if name in self.credentials:
            del self.credentials[name]
            return True
        return False


async def test_simple_credential_manager():
    """Test simple credential manager."""
    print("Testing simple credential manager...")

    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir)
        manager = SimpleCredentialManager(storage_path)

        # Test storing credential
        result = await manager.store_credential("test_key", "test_value")
        print(f"Store credential: {'✓' if result else '✗'}")

        # Test retrieving credential
        retrieved = await manager.retrieve_credential("test_key")
        print(f"Retrieve credential: {'✓' if retrieved == 'test_value' else '✗'}")

        # Test deleting credential
        deleted = await manager.delete_credential("test_key")
        print(f"Delete credential: {'✓' if deleted else '✗'}")

        # Test retrieving deleted credential
        not_found = await manager.retrieve_credential("test_key")
        print(f"Credential deleted: {'✓' if not_found is None else '✗'}")

        print("Simple credential manager test completed!")


if __name__ == "__main__":
    asyncio.run(test_simple_credential_manager())
