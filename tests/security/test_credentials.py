"""Tests for secure credential management system."""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.security.credentials import (
    AWSSecretsManagerProvider,
    CredentialManager,
    CredentialRotationManager,
    EnvironmentCredentialProvider,
    HashiCorpVaultProvider,
)
from app.security.models import SecurityContext, SecurityLevel


@pytest.fixture
def temp_storage_path():
    """Create temporary storage path for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def security_context():
    """Create test security context."""
    return SecurityContext(
        user_id="test_user",
        session_id="test_session",
        ip_address="127.0.0.1",
        operation="test_operation",
    )


@pytest.fixture
def credential_manager(temp_storage_path):
    """Create credential manager for testing."""
    return CredentialManager(
        master_key="test_master_key_for_testing_only", storage_path=temp_storage_path
    )


class TestEnvironmentCredentialProvider:
    """Test environment credential provider."""

    def test_init(self):
        """Test provider initialization."""
        provider = EnvironmentCredentialProvider()
        assert provider.prefix == "OPENMANUS_"

        provider = EnvironmentCredentialProvider(prefix="TEST_")
        assert provider.prefix == "TEST_"

    @pytest.mark.asyncio
    async def test_get_credential_exists(self):
        """Test getting existing credential from environment."""
        provider = EnvironmentCredentialProvider(prefix="TEST_")

        # Set environment variable
        os.environ["TEST_API_KEY"] = "test_value"

        try:
            result = await provider.get_credential("api_key")
            assert result == "test_value"
        finally:
            # Clean up
            os.environ.pop("TEST_API_KEY", None)

    @pytest.mark.asyncio
    async def test_get_credential_not_exists(self):
        """Test getting non-existent credential."""
        provider = EnvironmentCredentialProvider(prefix="TEST_")
        result = await provider.get_credential("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_credential(self):
        """Test setting credential in environment."""
        provider = EnvironmentCredentialProvider(prefix="TEST_")

        success = await provider.set_credential("new_key", "new_value")
        assert success is True
        assert os.environ.get("TEST_NEW_KEY") == "new_value"

        # Clean up
        os.environ.pop("TEST_NEW_KEY", None)

    def test_list_credentials(self):
        """Test listing credentials from environment."""
        provider = EnvironmentCredentialProvider(prefix="TEST_")

        # Set test environment variables
        os.environ["TEST_KEY1"] = "value1"
        os.environ["TEST_KEY2"] = "value2"
        os.environ["OTHER_KEY"] = "other_value"

        try:
            credentials = provider.list_credentials()
            assert "key1" in credentials
            assert "key2" in credentials
            assert "other_key" not in credentials  # Different prefix
        finally:
            # Clean up
            os.environ.pop("TEST_KEY1", None)
            os.environ.pop("TEST_KEY2", None)
            os.environ.pop("OTHER_KEY", None)


class TestAWSSecretsManagerProvider:
    """Test AWS Secrets Manager provider."""

    @pytest.mark.asyncio
    async def test_get_client_missing_boto3(self):
        """Test client creation when boto3 is not available."""
        provider = AWSSecretsManagerProvider()

        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'boto3'")
        ):
            with pytest.raises(ValueError, match="boto3 required"):
                await provider._get_client()

    @pytest.mark.asyncio
    async def test_get_secret_success(self):
        """Test successful secret retrieval."""
        provider = AWSSecretsManagerProvider()

        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {
            "SecretString": "test_secret_value"
        }

        with patch.object(provider, "_get_client", return_value=mock_client):
            result = await provider.get_secret("test_secret")
            assert result == "test_secret_value"
            mock_client.get_secret_value.assert_called_once_with(SecretId="test_secret")

    @pytest.mark.asyncio
    async def test_get_secret_failure(self):
        """Test secret retrieval failure."""
        provider = AWSSecretsManagerProvider()

        mock_client = MagicMock()
        mock_client.get_secret_value.side_effect = Exception("AWS error")

        with patch.object(provider, "_get_client", return_value=mock_client):
            result = await provider.get_secret("test_secret")
            assert result is None

    @pytest.mark.asyncio
    async def test_set_secret_update_existing(self):
        """Test updating existing secret."""
        provider = AWSSecretsManagerProvider()

        mock_client = MagicMock()
        mock_client.update_secret.return_value = {}

        with patch.object(provider, "_get_client", return_value=mock_client):
            result = await provider.set_secret("test_secret", "new_value")
            assert result is True
            mock_client.update_secret.assert_called_once_with(
                SecretId="test_secret", SecretString="new_value"
            )

    @pytest.mark.asyncio
    async def test_set_secret_create_new(self):
        """Test creating new secret."""
        provider = AWSSecretsManagerProvider()

        mock_client = MagicMock()
        mock_client.update_secret.side_effect = (
            mock_client.exceptions.ResourceNotFoundException()
        )
        mock_client.create_secret.return_value = {}
        mock_client.exceptions.ResourceNotFoundException = Exception

        with patch.object(provider, "_get_client", return_value=mock_client):
            result = await provider.set_secret("test_secret", "new_value")
            assert result is True
            mock_client.create_secret.assert_called_once_with(
                Name="test_secret", SecretString="new_value"
            )


class TestHashiCorpVaultProvider:
    """Test HashiCorp Vault provider."""

    def test_init(self):
        """Test provider initialization."""
        provider = HashiCorpVaultProvider("https://vault.example.com", "test_token")
        assert provider.vault_url == "https://vault.example.com"
        assert provider.vault_token == "test_token"

    def test_init_with_env_token(self):
        """Test initialization with environment token."""
        os.environ["VAULT_TOKEN"] = "env_token"
        try:
            provider = HashiCorpVaultProvider("https://vault.example.com")
            assert provider.vault_token == "env_token"
        finally:
            os.environ.pop("VAULT_TOKEN", None)

    @pytest.mark.asyncio
    async def test_make_request_no_token(self):
        """Test request without token."""
        provider = HashiCorpVaultProvider("https://vault.example.com")
        provider.vault_token = None

        result = await provider._make_request("GET", "test/path")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_secret_success(self):
        """Test successful secret retrieval from Vault."""
        provider = HashiCorpVaultProvider("https://vault.example.com", "test_token")

        mock_response = {"data": {"data": {"value": "secret_value"}}}

        with patch.object(provider, "_make_request", return_value=mock_response):
            result = await provider.get_secret("test_secret")
            assert result == "secret_value"

    @pytest.mark.asyncio
    async def test_get_secret_not_found(self):
        """Test secret not found in Vault."""
        provider = HashiCorpVaultProvider("https://vault.example.com", "test_token")

        with patch.object(provider, "_make_request", return_value=None):
            result = await provider.get_secret("nonexistent")
            assert result is None


class TestCredentialRotationManager:
    """Test credential rotation manager."""

    @pytest.fixture
    def rotation_manager(self, credential_manager):
        """Create rotation manager for testing."""
        return CredentialRotationManager(credential_manager)

    @pytest.mark.asyncio
    async def test_set_rotation_policy(self, rotation_manager):
        """Test setting rotation policy."""
        interval = timedelta(days=30)
        callback = AsyncMock()

        await rotation_manager.set_rotation_policy(
            "test_credential", interval, callback, auto_rotate=True
        )

        assert "test_credential" in rotation_manager.rotation_policies
        policy = rotation_manager.rotation_policies["test_credential"]
        assert policy["interval"] == interval
        assert policy["callback"] == callback
        assert policy["auto_rotate"] is True

    @pytest.mark.asyncio
    async def test_check_rotation_needed(self, rotation_manager):
        """Test checking if rotation is needed."""
        # No policy set
        result = await rotation_manager.check_rotation_needed("nonexistent")
        assert result is False

        # Set policy with past rotation time
        past_time = datetime.utcnow() - timedelta(days=1)
        rotation_manager.rotation_policies["test_credential"] = {
            "next_rotation": past_time
        }

        result = await rotation_manager.check_rotation_needed("test_credential")
        assert result is True

    @pytest.mark.asyncio
    async def test_rotate_credential_success(
        self, rotation_manager, credential_manager
    ):
        """Test successful credential rotation."""
        # Store initial credential
        await credential_manager.store_credential("test_credential", "old_value")

        # Set rotation policy
        await rotation_manager.set_rotation_policy(
            "test_credential", timedelta(days=30)
        )

        # Mock the credential manager's rotate_credential method
        with patch.object(credential_manager, "rotate_credential", return_value=True):
            result = await rotation_manager.rotate_credential("test_credential")
            assert result is True

    @pytest.mark.asyncio
    async def test_get_rotation_status(self, rotation_manager):
        """Test getting rotation status."""
        # Set up test policy
        await rotation_manager.set_rotation_policy(
            "test_credential", timedelta(days=30), auto_rotate=True
        )

        status = await rotation_manager.get_rotation_status()
        assert "test_credential" in status
        assert "last_rotation" in status["test_credential"]
        assert "next_rotation" in status["test_credential"]
        assert "needs_rotation" in status["test_credential"]
        assert "auto_rotate" in status["test_credential"]


class TestCredentialManager:
    """Test credential manager."""

    @pytest.mark.asyncio
    async def test_store_credential_local(self, credential_manager, security_context):
        """Test storing credential locally."""
        result = await credential_manager.store_credential(
            "test_api_key",
            "secret_value",
            credential_type="api_key",
            context=security_context,
        )

        assert result is True
        assert "test_api_key" in credential_manager.credentials

        stored_cred = credential_manager.credentials["test_api_key"]
        assert stored_cred["value"] == "secret_value"
        assert stored_cred["type"] == "api_key"
        assert "created" in stored_cred

    @pytest.mark.asyncio
    async def test_store_credential_dict_value(
        self, credential_manager, security_context
    ):
        """Test storing credential with dictionary value."""
        cred_value = {"username": "user", "password": "pass"}

        result = await credential_manager.store_credential(
            "test_creds", cred_value, context=security_context
        )

        assert result is True
        stored_cred = credential_manager.credentials["test_creds"]
        assert stored_cred["value"] == cred_value

    @pytest.mark.asyncio
    async def test_retrieve_credential_local(
        self, credential_manager, security_context
    ):
        """Test retrieving credential from local storage."""
        # Store credential first
        await credential_manager.store_credential(
            "test_key", "test_value", context=security_context
        )

        # Retrieve credential
        result = await credential_manager.retrieve_credential(
            "test_key", context=security_context
        )

        assert result == "test_value"

        # Check access tracking
        stored_cred = credential_manager.credentials["test_key"]
        assert stored_cred["access_count"] == 1
        assert stored_cred["last_accessed"] is not None

    @pytest.mark.asyncio
    async def test_retrieve_credential_not_found(
        self, credential_manager, security_context
    ):
        """Test retrieving non-existent credential."""
        result = await credential_manager.retrieve_credential(
            "nonexistent", context=security_context
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_credential(self, credential_manager, security_context):
        """Test deleting credential."""
        # Store credential first
        await credential_manager.store_credential(
            "test_key", "test_value", context=security_context
        )

        # Delete credential
        result = await credential_manager.delete_credential(
            "test_key", context=security_context
        )

        assert result is True
        assert "test_key" not in credential_manager.credentials

    @pytest.mark.asyncio
    async def test_list_credentials(self, credential_manager, security_context):
        """Test listing credentials."""
        # Store test credentials
        await credential_manager.store_credential(
            "key1", "value1", "api_key", context=security_context
        )
        await credential_manager.store_credential(
            "key2", "value2", "token", context=security_context
        )

        # List credentials
        credentials = await credential_manager.list_credentials(
            context=security_context
        )

        assert len(credentials) == 2

        # Check that sensitive values are not included
        for cred in credentials:
            assert "value" not in cred
            assert "name" in cred
            assert "type" in cred

    @pytest.mark.asyncio
    async def test_rotate_credential(self, credential_manager, security_context):
        """Test credential rotation."""
        # Store initial credential
        await credential_manager.store_credential(
            "test_key", "old_value", context=security_context
        )

        # Rotate credential
        result = await credential_manager.rotate_credential(
            "test_key", "new_value", context=security_context
        )

        assert result is True

        # Verify new value
        new_value = await credential_manager.retrieve_credential(
            "test_key", context=security_context
        )
        assert new_value == "new_value"

        # Check rotation metadata
        stored_cred = credential_manager.credentials["test_key"]
        assert "last_rotated" in stored_cred
        assert stored_cred["rotation_count"] == 1

    @pytest.mark.asyncio
    async def test_backup_and_restore_credentials(
        self, credential_manager, security_context, temp_storage_path
    ):
        """Test credential backup and restore."""
        # Store test credentials
        await credential_manager.store_credential(
            "key1", "value1", context=security_context
        )
        await credential_manager.store_credential(
            "key2", "value2", context=security_context
        )

        # Create backup
        backup_path = temp_storage_path / "test_backup.enc"
        result = await credential_manager.backup_credentials(
            backup_path, context=security_context
        )
        assert result is True
        assert backup_path.exists()

        # Clear credentials
        credential_manager.credentials.clear()

        # Restore from backup
        result = await credential_manager.restore_credentials(
            backup_path, context=security_context
        )
        assert result is True
        assert len(credential_manager.credentials) == 2
        assert "key1" in credential_manager.credentials
        assert "key2" in credential_manager.credentials

    def test_get_credential_stats(self, credential_manager):
        """Test getting credential statistics."""
        # Add test credentials with different types and access patterns
        credential_manager.credentials = {
            "key1": {
                "type": "api_key",
                "access_count": 5,
                "created": "2023-01-01T00:00:00",
            },
            "key2": {
                "type": "token",
                "access_count": 0,
                "created": "2023-06-01T00:00:00",
            },
            "key3": {
                "type": "api_key",
                "access_count": 10,
                "created": "2023-03-01T00:00:00",
            },
        }

        stats = credential_manager.get_credential_stats()

        assert stats["total_credentials"] == 3
        assert stats["by_type"]["api_key"] == 2
        assert stats["by_type"]["token"] == 1
        assert stats["access_stats"]["never_accessed"] == 1
        assert stats["access_stats"]["most_accessed"][1] == 10
        assert stats["access_stats"]["least_accessed"][1] == 0

    @pytest.mark.asyncio
    async def test_setup_credential_rotation(
        self, credential_manager, security_context
    ):
        """Test setting up credential rotation."""
        # Store credential first
        await credential_manager.store_credential(
            "test_key", "test_value", context=security_context
        )

        # Setup rotation
        result = await credential_manager.setup_credential_rotation(
            "test_key",
            rotation_interval_days=30,
            auto_rotate=True,
            context=security_context,
        )

        assert result is True
        assert "test_key" in credential_manager.rotation_manager.rotation_policies

    @pytest.mark.asyncio
    async def test_check_and_rotate_credentials(
        self, credential_manager, security_context
    ):
        """Test checking and rotating credentials."""
        # Store credential
        await credential_manager.store_credential(
            "test_key", "test_value", context=security_context
        )

        # Setup rotation with past due date
        past_time = datetime.utcnow() - timedelta(days=1)
        credential_manager.rotation_manager.rotation_policies["test_key"] = {
            "interval": timedelta(days=30),
            "callback": None,
            "auto_rotate": True,
            "last_rotation": past_time,
            "next_rotation": past_time,
        }

        # Mock the rotation to succeed
        with patch.object(credential_manager, "rotate_credential", return_value=True):
            results = await credential_manager.check_and_rotate_credentials(
                context=security_context
            )

            assert "test_key" in results
            assert results["test_key"] is True

    def test_encrypt_decrypt_transit(self, credential_manager):
        """Test transit encryption and decryption."""
        test_data = "sensitive_credential_data"

        # Encrypt for transit
        encrypted = credential_manager.encrypt_for_transit(test_data)
        assert encrypted != test_data
        assert isinstance(encrypted, str)

        # Decrypt from transit
        decrypted = credential_manager.decrypt_from_transit(encrypted)
        assert decrypted == test_data

    @pytest.mark.asyncio
    async def test_export_credentials_secure(
        self, credential_manager, security_context, temp_storage_path
    ):
        """Test secure credential export."""
        # Store test credentials
        await credential_manager.store_credential(
            "key1", "value1", context=security_context
        )
        await credential_manager.store_credential(
            "key2", {"user": "test", "pass": "secret"}, context=security_context
        )

        # Export without values
        export_path = temp_storage_path / "export_no_values.enc"
        result = await credential_manager.export_credentials_secure(
            export_path, include_values=False, context=security_context
        )
        assert result is True
        assert export_path.exists()

        # Export with values
        export_path_with_values = temp_storage_path / "export_with_values.enc"
        result = await credential_manager.export_credentials_secure(
            export_path_with_values, include_values=True, context=security_context
        )
        assert result is True
        assert export_path_with_values.exists()

    @pytest.mark.asyncio
    async def test_get_credential_health_report(self, credential_manager):
        """Test credential health report generation."""
        # Add test credentials with various states
        old_date = (datetime.utcnow() - timedelta(days=400)).isoformat()
        credential_manager.credentials = {
            "old_key": {"type": "api_key", "created": old_date, "access_count": 0},
            "new_key": {
                "type": "token",
                "created": datetime.utcnow().isoformat(),
                "access_count": 5,
            },
        }

        report = await credential_manager.get_credential_health_report()

        assert "timestamp" in report
        assert "statistics" in report
        assert "security_issues" in report
        assert "health_score" in report
        assert "recommendations" in report

        # Check for detected issues
        issue_types = [issue["type"] for issue in report["security_issues"]]
        assert "old_credential" in issue_types
        assert "unused_credential" in issue_types

    @pytest.mark.asyncio
    async def test_get_audit_events(self, credential_manager, security_context):
        """Test getting audit events."""
        # Perform some operations to generate audit events
        await credential_manager.store_credential(
            "test_key", "test_value", context=security_context
        )
        await credential_manager.retrieve_credential(
            "test_key", context=security_context
        )

        # Get all events
        events = await credential_manager.get_audit_events()
        assert len(events) >= 2

        # Get events with limit
        limited_events = await credential_manager.get_audit_events(limit=1)
        assert len(limited_events) == 1

        # Get events by type
        store_events = await credential_manager.get_audit_events(
            event_types=["credential_stored"]
        )
        assert len(store_events) >= 1
        assert all(event.event_type == "credential_stored" for event in store_events)

    @pytest.mark.asyncio
    async def test_provider_fallback(self, temp_storage_path, security_context):
        """Test provider fallback functionality."""
        # Create credential manager with environment fallback
        env_provider = EnvironmentCredentialProvider(prefix="TEST_")
        credential_manager = CredentialManager(
            master_key="test_key",
            storage_path=temp_storage_path,
            enable_environment_fallback=True,
        )
        credential_manager.env_provider = env_provider

        # Set credential in environment
        os.environ["TEST_FALLBACK_KEY"] = "env_value"

        try:
            # Should find credential in environment when not in local storage
            result = await credential_manager.retrieve_credential(
                "fallback_key", context=security_context
            )
            assert result == "env_value"
        finally:
            os.environ.pop("TEST_FALLBACK_KEY", None)

    def test_master_key_generation(self, temp_storage_path):
        """Test master key generation."""
        # Create credential manager without master key
        with patch.dict(os.environ, {}, clear=True):
            credential_manager = CredentialManager(storage_path=temp_storage_path)

            # Should have generated a master key
            assert credential_manager.master_key is not None

            # Should have created .env file
            env_file = temp_storage_path / ".env"
            assert env_file.exists()

            # File should have restrictive permissions
            assert oct(env_file.stat().st_mode)[-3:] == "600"

    @pytest.mark.asyncio
    async def test_invalid_inputs(self, credential_manager, security_context):
        """Test handling of invalid inputs."""
        # Test storing credential with empty name
        result = await credential_manager.store_credential(
            "", "value", context=security_context
        )
        assert result is False

        # Test storing credential with empty value
        result = await credential_manager.store_credential(
            "name", "", context=security_context
        )
        assert result is False

        # Test deleting non-existent credential
        result = await credential_manager.delete_credential(
            "nonexistent", context=security_context
        )
        assert result is False

    def test_credential_persistence(self, temp_storage_path, security_context):
        """Test credential persistence across manager instances."""
        # Create first manager and store credential
        manager1 = CredentialManager(
            master_key="test_key", storage_path=temp_storage_path
        )

        asyncio.run(
            manager1.store_credential(
                "persistent_key", "persistent_value", context=security_context
            )
        )

        # Create second manager with same storage path
        manager2 = CredentialManager(
            master_key="test_key", storage_path=temp_storage_path
        )

        # Should be able to retrieve credential
        result = asyncio.run(
            manager2.retrieve_credential("persistent_key", context=security_context)
        )
        assert result == "persistent_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
