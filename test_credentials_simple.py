#!/usr/bin/env python3
"""Simple test runner for credential management system."""

import asyncio
import os

# Add current directory to path
import sys
import tempfile
from pathlib import Path

sys.path.append(".")

from app.security.credentials import (
    CredentialManager,
    CredentialRotationManager,
    EnvironmentCredentialProvider,
)
from app.security.models import SecurityContext, SecurityLevel


async def test_environment_provider():
    """Test environment credential provider."""
    print("Testing EnvironmentCredentialProvider...")

    provider = EnvironmentCredentialProvider(prefix="TEST_")

    # Test setting and getting credential
    await provider.set_credential("api_key", "test_value")
    result = await provider.get_credential("api_key")
    assert result == "test_value", f"Expected 'test_value', got {result}"

    # Clean up
    os.environ.pop("TEST_API_KEY", None)

    print("✓ EnvironmentCredentialProvider tests passed")


async def test_credential_manager():
    """Test credential manager."""
    print("Testing CredentialManager...")

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
        assert result is True, "Failed to store credential"

        # Test retrieving credential
        retrieved = await manager.retrieve_credential("test_api_key", context=context)
        assert retrieved == "secret_value", f"Expected 'secret_value', got {retrieved}"

        # Test listing credentials
        credentials = await manager.list_credentials(context=context)
        assert len(credentials) == 1, f"Expected 1 credential, got {len(credentials)}"
        assert credentials[0]["name"] == "test_api_key"
        assert credentials[0]["type"] == "api_key"

        # Test credential rotation
        rotation_result = await manager.rotate_credential(
            "test_api_key", "new_secret_value", context=context
        )
        assert rotation_result is True, "Failed to rotate credential"

        # Verify new value
        new_value = await manager.retrieve_credential("test_api_key", context=context)
        assert (
            new_value == "new_secret_value"
        ), f"Expected 'new_secret_value', got {new_value}"

        # Test credential deletion
        delete_result = await manager.delete_credential("test_api_key", context=context)
        assert delete_result is True, "Failed to delete credential"

        # Verify deletion
        deleted_value = await manager.retrieve_credential(
            "test_api_key", context=context
        )
        assert deleted_value is None, "Credential should be deleted"

        print("✓ CredentialManager tests passed")


async def test_transit_encryption():
    """Test transit encryption functionality."""
    print("Testing transit encryption...")

    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir)

        manager = CredentialManager(
            master_key="test_master_key_for_testing", storage_path=storage_path
        )

        # Test data
        test_data = "sensitive_credential_data_for_transit"

        # Encrypt for transit
        encrypted = manager.encrypt_for_transit(test_data)
        assert encrypted != test_data, "Data should be encrypted"
        assert isinstance(encrypted, str), "Encrypted data should be string"

        # Decrypt from transit
        decrypted = manager.decrypt_from_transit(encrypted)
        assert decrypted == test_data, f"Expected '{test_data}', got '{decrypted}'"

        print("✓ Transit encryption tests passed")


async def test_credential_rotation():
    """Test credential rotation manager."""
    print("Testing CredentialRotationManager...")

    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir)

        manager = CredentialManager(
            master_key="test_master_key_for_testing", storage_path=storage_path
        )

        context = SecurityContext(
            user_id="test_user",
            session_id="test_session",
            ip_address="127.0.0.1",
            operation="test",
        )

        # Store initial credential
        await manager.store_credential("test_key", "initial_value", context=context)

        # Setup rotation policy
        result = await manager.setup_credential_rotation(
            "test_key", rotation_interval_days=30, auto_rotate=True, context=context
        )
        assert result is True, "Failed to setup rotation policy"

        # Check rotation status
        status = await manager.rotation_manager.get_rotation_status()
        assert "test_key" in status, "Rotation policy not found"
        assert status["test_key"]["auto_rotate"] is True

        print("✓ CredentialRotationManager tests passed")


async def test_backup_restore():
    """Test credential backup and restore."""
    print("Testing backup and restore...")

    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir)

        manager = CredentialManager(
            master_key="test_master_key_for_testing", storage_path=storage_path
        )

        context = SecurityContext(
            user_id="test_user",
            session_id="test_session",
            ip_address="127.0.0.1",
            operation="test",
        )

        # Store test credentials
        await manager.store_credential("key1", "value1", context=context)
        await manager.store_credential("key2", "value2", context=context)

        # Create backup
        backup_path = storage_path / "test_backup.enc"
        backup_result = await manager.backup_credentials(backup_path, context=context)
        assert backup_result is True, "Failed to create backup"
        assert backup_path.exists(), "Backup file not created"

        # Clear credentials
        manager.credentials.clear()

        # Restore from backup
        restore_result = await manager.restore_credentials(backup_path, context=context)
        assert restore_result is True, "Failed to restore from backup"
        assert len(manager.credentials) == 2, "Not all credentials restored"
        assert "key1" in manager.credentials
        assert "key2" in manager.credentials

        print("✓ Backup and restore tests passed")


async def test_audit_logging():
    """Test audit logging functionality."""
    print("Testing audit logging...")

    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir)

        manager = CredentialManager(
            master_key="test_master_key_for_testing", storage_path=storage_path
        )

        context = SecurityContext(
            user_id="test_user",
            session_id="test_session",
            ip_address="127.0.0.1",
            operation="test",
        )

        # Perform operations that generate audit events
        await manager.store_credential("audit_key", "audit_value", context=context)
        await manager.retrieve_credential("audit_key", context=context)
        await manager.delete_credential("audit_key", context=context)

        # Get audit events
        events = await manager.get_audit_events()
        assert len(events) >= 3, f"Expected at least 3 audit events, got {len(events)}"

        # Check event types
        event_types = [event.event_type for event in events]
        assert "credential_stored" in event_types
        assert "credential_accessed_local" in event_types
        assert "credential_deleted" in event_types

        print("✓ Audit logging tests passed")


async def main():
    """Run all tests."""
    print("Running credential management system tests...\n")

    try:
        await test_environment_provider()
        await test_credential_manager()
        await test_transit_encryption()
        await test_credential_rotation()
        await test_backup_restore()
        await test_audit_logging()

        print("\n🎉 All credential management tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
