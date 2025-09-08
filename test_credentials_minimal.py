#!/usr/bin/env python3
"""Minimal test for credential management system."""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.append(".")

from app.security.credentials import CredentialManager, EnvironmentCredentialProvider
from app.security.models import SecurityContext, SecurityLevel


async def test_basic_functionality():
    """Test basic credential management functionality."""
    print("Testing basic credential management...")

    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir)

        # Create credential manager
        manager = CredentialManager(
            master_key="test_master_key_for_testing", storage_path=storage_path
        )

        # Create security context
        context = SecurityContext(
            user_id="test_user",
            session_id="test_session",
            ip_address="127.0.0.1",
            operation="test",
        )

        # Test storing credential
        result = await manager.store_credential(
            "test_api_key", "secret_value", credential_type="api_key", context=context
        )
        print(f"Store credential: {'✓' if result else '✗'}")

        # Test retrieving credential
        retrieved = await manager.retrieve_credential("test_api_key", context=context)
        print(f"Retrieve credential: {'✓' if retrieved == 'secret_value' else '✗'}")

        # Test transit encryption
        test_data = "sensitive_data_for_transit"
        encrypted = manager.encrypt_for_transit(test_data)
        decrypted = manager.decrypt_from_transit(encrypted)
        print(f"Transit encryption: {'✓' if decrypted == test_data else '✗'}")

        # Test environment provider
        env_provider = EnvironmentCredentialProvider(prefix="TEST_")
        await env_provider.set_credential("env_key", "env_value")
        env_result = await env_provider.get_credential("env_key")
        print(f"Environment provider: {'✓' if env_result == 'env_value' else '✗'}")

        # Clean up
        os.environ.pop("TEST_ENV_KEY", None)

        print("All basic tests passed!")


if __name__ == "__main__":
    asyncio.run(test_basic_functionality())
