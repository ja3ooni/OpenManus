"""Secure credential management system."""

import base64
import json
import os
import secrets
import ssl
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from app.logger import define_log_level
from app.security.models import SecurityContext, SecurityEvent, SecurityLevel

logger = define_log_level()


class EnvironmentCredentialProvider:
    """Provides credentials from environment variables with secure fallback."""

    def __init__(self, prefix: str = "OPENMANUS_"):
        """Initialize with environment variable prefix."""
        self.prefix = prefix

    async def get_credential(self, name: str) -> Optional[str]:
        """Get credential from environment variable."""
        env_name = f"{self.prefix}{name.upper()}"
        value = os.getenv(env_name)

        if value:
            logger.info(f"Retrieved credential from environment: {env_name}")
            return value

        return None

    async def set_credential(self, name: str, value: str) -> bool:
        """Set credential in environment (runtime only)."""
        env_name = f"{self.prefix}{name.upper()}"
        os.environ[env_name] = value
        logger.info(f"Set credential in environment: {env_name}")
        return True

    def list_credentials(self) -> List[str]:
        """List available credentials from environment."""
        credentials = []
        for key in os.environ:
            if key.startswith(self.prefix):
                credential_name = key[len(self.prefix) :].lower()
                credentials.append(credential_name)
        return credentials


class KeyVaultProvider:
    """Abstract key vault provider interface."""

    async def get_secret(self, name: str) -> Optional[str]:
        """Get secret from key vault."""
        raise NotImplementedError

    async def set_secret(self, name: str, value: str) -> bool:
        """Set secret in key vault."""
        raise NotImplementedError

    async def delete_secret(self, name: str) -> bool:
        """Delete secret from key vault."""
        raise NotImplementedError

    async def list_secrets(self) -> List[str]:
        """List available secrets."""
        raise NotImplementedError


class AWSSecretsManagerProvider(KeyVaultProvider):
    """AWS Secrets Manager provider."""

    def __init__(self, region: str = "us-east-1"):
        """Initialize AWS Secrets Manager client."""
        self.region = region
        self._client = None

    async def _get_client(self):
        """Get or create AWS Secrets Manager client."""
        if self._client is None:
            try:
                import boto3

                self._client = boto3.client("secretsmanager", region_name=self.region)
            except ImportError:
                logger.error("boto3 not available for AWS Secrets Manager")
                raise ValueError("boto3 required for AWS Secrets Manager")
        return self._client

    async def get_secret(self, name: str) -> Optional[str]:
        """Get secret from AWS Secrets Manager."""
        try:
            client = await self._get_client()
            response = client.get_secret_value(SecretId=name)
            return response["SecretString"]
        except Exception as e:
            logger.error(f"Failed to get secret from AWS: {e}")
            return None

    async def set_secret(self, name: str, value: str) -> bool:
        """Set secret in AWS Secrets Manager."""
        try:
            client = await self._get_client()
            try:
                client.update_secret(SecretId=name, SecretString=value)
            except client.exceptions.ResourceNotFoundException:
                client.create_secret(Name=name, SecretString=value)
            return True
        except Exception as e:
            logger.error(f"Failed to set secret in AWS: {e}")
            return False

    async def delete_secret(self, name: str) -> bool:
        """Delete secret from AWS Secrets Manager."""
        try:
            client = await self._get_client()
            client.delete_secret(SecretId=name, ForceDeleteWithoutRecovery=True)
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret from AWS: {e}")
            return False

    async def list_secrets(self) -> List[str]:
        """List available secrets."""
        try:
            client = await self._get_client()
            response = client.list_secrets()
            return [secret["Name"] for secret in response["SecretList"]]
        except Exception as e:
            logger.error(f"Failed to list secrets from AWS: {e}")
            return []


class HashiCorpVaultProvider(KeyVaultProvider):
    """HashiCorp Vault provider."""

    def __init__(self, vault_url: str, vault_token: Optional[str] = None):
        """Initialize Vault provider."""
        self.vault_url = vault_url.rstrip("/")
        self.vault_token = vault_token or os.getenv("VAULT_TOKEN")
        self.mount_point = "secret"

    async def _make_request(
        self, method: str, path: str, data: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Make authenticated request to Vault."""
        if not self.vault_token:
            logger.error("Vault token not available")
            return None

        url = f"{self.vault_url}/v1/{path}"
        headers = {
            "X-Vault-Token": self.vault_token,
            "Content-Type": "application/json",
        }

        try:
            req_data = json.dumps(data).encode() if data else None
            request = urllib.request.Request(
                url, data=req_data, headers=headers, method=method
            )

            with urllib.request.urlopen(request) as response:
                if response.status == 200:
                    return json.loads(response.read().decode())
                elif response.status == 404:
                    return None
                else:
                    logger.error(f"Vault request failed: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Vault request error: {e}")
            return None

    async def get_secret(self, name: str) -> Optional[str]:
        """Get secret from Vault."""
        response = await self._make_request("GET", f"{self.mount_point}/data/{name}")
        if response and "data" in response and "data" in response["data"]:
            return response["data"]["data"].get("value")
        return None

    async def set_secret(self, name: str, value: str) -> bool:
        """Set secret in Vault."""
        data = {"data": {"value": value}}
        response = await self._make_request(
            "POST", f"{self.mount_point}/data/{name}", data
        )
        return response is not None

    async def delete_secret(self, name: str) -> bool:
        """Delete secret from Vault."""
        response = await self._make_request("DELETE", f"{self.mount_point}/data/{name}")
        return response is not None

    async def list_secrets(self) -> List[str]:
        """List available secrets."""
        response = await self._make_request("LIST", f"{self.mount_point}/metadata")
        if response and "data" in response and "keys" in response["data"]:
            return response["data"]["keys"]
        return []


class CredentialRotationManager:
    """Manages automatic credential rotation."""

    def __init__(self, credential_manager: "CredentialManager"):
        """Initialize rotation manager."""
        self.credential_manager = credential_manager
        self.rotation_policies: Dict[str, Dict[str, Any]] = {}

    async def set_rotation_policy(
        self,
        credential_name: str,
        rotation_interval: timedelta,
        rotation_callback: Optional[callable] = None,
        auto_rotate: bool = True,
    ):
        """Set rotation policy for a credential."""
        self.rotation_policies[credential_name] = {
            "interval": rotation_interval,
            "callback": rotation_callback,
            "auto_rotate": auto_rotate,
            "last_rotation": datetime.utcnow(),
            "next_rotation": datetime.utcnow() + rotation_interval,
        }

    async def check_rotation_needed(self, credential_name: str) -> bool:
        """Check if credential needs rotation."""
        if credential_name not in self.rotation_policies:
            return False

        policy = self.rotation_policies[credential_name]
        return datetime.utcnow() >= policy["next_rotation"]

    async def rotate_credential(
        self, credential_name: str, context: Optional[SecurityContext] = None
    ) -> bool:
        """Rotate a credential."""
        if credential_name not in self.rotation_policies:
            logger.error(f"No rotation policy for credential: {credential_name}")
            return False

        policy = self.rotation_policies[credential_name]

        try:
            # Generate new credential value
            if policy["callback"]:
                new_value = await policy["callback"](credential_name)
            else:
                # Default: generate secure random string
                new_value = secrets.token_urlsafe(32)

            # Rotate the credential
            success = await self.credential_manager.rotate_credential(
                credential_name, new_value, context
            )

            if success:
                # Update rotation tracking
                policy["last_rotation"] = datetime.utcnow()
                policy["next_rotation"] = datetime.utcnow() + policy["interval"]

                logger.info(f"Successfully rotated credential: {credential_name}")
                return True
            else:
                logger.error(f"Failed to rotate credential: {credential_name}")
                return False

        except Exception as e:
            logger.error(f"Error rotating credential {credential_name}: {e}")
            return False

    async def get_rotation_status(self) -> Dict[str, Dict[str, Any]]:
        """Get rotation status for all managed credentials."""
        status = {}
        for name, policy in self.rotation_policies.items():
            status[name] = {
                "last_rotation": policy["last_rotation"],
                "next_rotation": policy["next_rotation"],
                "needs_rotation": await self.check_rotation_needed(name),
                "auto_rotate": policy["auto_rotate"],
            }
        return status


class CredentialManager:
    """Secure credential storage and management with encryption and audit logging."""

    def __init__(
        self,
        master_key: Optional[str] = None,
        storage_path: Optional[Path] = None,
        key_vault_provider: Optional[KeyVaultProvider] = None,
        enable_environment_fallback: bool = True,
    ):
        """Initialize credential manager with encryption and multiple providers."""
        self.storage_path = storage_path or Path.home() / ".openmanus" / "credentials"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize encryption
        self.master_key = master_key or os.getenv("OPENMANUS_MASTER_KEY")
        if not self.master_key:
            self.master_key = self._generate_master_key()

        self.cipher_suite = self._create_cipher_suite(self.master_key)

        # Initialize providers
        self.key_vault_provider = key_vault_provider
        self.env_provider = (
            EnvironmentCredentialProvider() if enable_environment_fallback else None
        )

        # Initialize rotation manager
        self.rotation_manager = CredentialRotationManager(self)

        # Credential storage
        self.credentials_file = self.storage_path / "credentials.enc"
        self.metadata_file = self.storage_path / "metadata.json"
        self.transit_keys_file = self.storage_path / "transit_keys.enc"

        # Audit logging
        self.audit_events: List[SecurityEvent] = []

        # Initialize transit encryption keys
        self._init_transit_encryption()

        # Load existing credentials
        self._load_credentials()

    def _init_transit_encryption(self):
        """Initialize transit encryption keys."""
        try:
            if self.transit_keys_file.exists():
                # Load existing keys
                with open(self.transit_keys_file, "rb") as f:
                    encrypted_keys = f.read()
                decrypted_keys = self.cipher_suite.decrypt(encrypted_keys)
                keys_data = json.loads(decrypted_keys.decode())

                # Load RSA private key
                private_key_pem = keys_data["private_key"].encode()
                self.transit_private_key = serialization.load_pem_private_key(
                    private_key_pem, password=None
                )
                self.transit_public_key = self.transit_private_key.public_key()
            else:
                # Generate new RSA key pair for transit encryption
                self.transit_private_key = rsa.generate_private_key(
                    public_exponent=65537, key_size=2048
                )
                self.transit_public_key = self.transit_private_key.public_key()

                # Save keys
                private_pem = self.transit_private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )

                keys_data = {
                    "private_key": private_pem.decode(),
                    "created": datetime.utcnow().isoformat(),
                }

                encrypted_keys = self.cipher_suite.encrypt(
                    json.dumps(keys_data).encode()
                )

                with open(self.transit_keys_file, "wb") as f:
                    f.write(encrypted_keys)

                # Set restrictive permissions
                os.chmod(self.transit_keys_file, 0o600)

                logger.info("Generated new transit encryption keys")

        except Exception as e:
            logger.error(f"Failed to initialize transit encryption: {e}")
            # Fallback to symmetric encryption only
            self.transit_private_key = None
            self.transit_public_key = None

    def encrypt_for_transit(self, data: str) -> str:
        """Encrypt data for secure transit."""
        try:
            if self.transit_public_key:
                # Use hybrid encryption: RSA + AES
                # Generate random AES key
                aes_key = os.urandom(32)
                iv = os.urandom(16)

                # Encrypt data with AES
                cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
                encryptor = cipher.encryptor()

                # Pad data to AES block size
                padded_data = data.encode()
                padding_length = 16 - (len(padded_data) % 16)
                padded_data += bytes([padding_length] * padding_length)

                encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

                # Encrypt AES key with RSA
                from cryptography.hazmat.primitives import asymmetric

                encrypted_key = self.transit_public_key.encrypt(
                    aes_key,
                    asymmetric.padding.OAEP(
                        mgf=asymmetric.padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None,
                    ),
                )

                # Combine encrypted key, IV, and data
                transit_package = {
                    "encrypted_key": base64.b64encode(encrypted_key).decode(),
                    "iv": base64.b64encode(iv).decode(),
                    "data": base64.b64encode(encrypted_data).decode(),
                    "method": "rsa_aes",
                }

                return base64.b64encode(json.dumps(transit_package).encode()).decode()
            else:
                # Fallback to Fernet encryption
                encrypted = self.cipher_suite.encrypt(data.encode())
                transit_package = {
                    "data": base64.b64encode(encrypted).decode(),
                    "method": "fernet",
                }
                return base64.b64encode(json.dumps(transit_package).encode()).decode()

        except Exception as e:
            logger.error(f"Failed to encrypt for transit: {e}")
            raise

    def decrypt_from_transit(self, encrypted_data: str) -> str:
        """Decrypt data from secure transit."""
        try:
            # Decode the transit package
            package_data = json.loads(base64.b64decode(encrypted_data).decode())
            method = package_data.get("method", "fernet")

            if method == "rsa_aes" and self.transit_private_key:
                # Decrypt with hybrid encryption
                encrypted_key = base64.b64decode(package_data["encrypted_key"])
                iv = base64.b64decode(package_data["iv"])
                encrypted_content = base64.b64decode(package_data["data"])

                # Decrypt AES key with RSA
                from cryptography.hazmat.primitives import asymmetric

                aes_key = self.transit_private_key.decrypt(
                    encrypted_key,
                    asymmetric.padding.OAEP(
                        mgf=asymmetric.padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None,
                    ),
                )

                # Decrypt data with AES
                cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
                decryptor = cipher.decryptor()
                padded_data = decryptor.update(encrypted_content) + decryptor.finalize()

                # Remove padding
                padding_length = padded_data[-1]
                data = padded_data[:-padding_length]

                return data.decode()
            else:
                # Fallback to Fernet decryption
                encrypted_content = base64.b64decode(package_data["data"])
                decrypted = self.cipher_suite.decrypt(encrypted_content)
                return decrypted.decode()

        except Exception as e:
            logger.error(f"Failed to decrypt from transit: {e}")
            raise

    def _generate_master_key(self) -> str:
        """Generate a new master key."""
        key = Fernet.generate_key()
        key_str = base64.urlsafe_b64encode(key).decode()

        # Save to environment file for persistence
        env_file = self.storage_path / ".env"
        with open(env_file, "w") as f:
            f.write(f"OPENMANUS_MASTER_KEY={key_str}\n")

        # Set restrictive permissions
        os.chmod(env_file, 0o600)

        logger.warning(
            "Generated new master key for credential encryption",
            {"key_file": str(env_file), "action": "master_key_generated"},
        )

        return key_str

    def _create_cipher_suite(self, master_key: str) -> Fernet:
        """Create cipher suite from master key."""
        try:
            # If master_key is already a Fernet key
            if len(master_key) == 44 and master_key.endswith("="):
                key = base64.urlsafe_b64decode(master_key.encode())
            else:
                # Derive key from password
                password = master_key.encode()
                salt = b"openmanus_salt_v1"  # In production, use random salt per installation
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = kdf.derive(password)

            return Fernet(base64.urlsafe_b64encode(key))

        except Exception as e:
            logger.error(f"Failed to create cipher suite: {e}")
            raise ValueError("Invalid master key for encryption")

    def _load_credentials(self):
        """Load encrypted credentials from storage."""
        self.credentials = {}

        if self.credentials_file.exists():
            try:
                with open(self.credentials_file, "rb") as f:
                    encrypted_data = f.read()

                decrypted_data = self.cipher_suite.decrypt(encrypted_data)
                self.credentials = json.loads(decrypted_data.decode())

                logger.info(
                    "Loaded encrypted credentials",
                    {"credential_count": len(self.credentials)},
                )

            except Exception as e:
                logger.error(f"Failed to load credentials: {e}")
                self.credentials = {}

    def _save_credentials(self):
        """Save encrypted credentials to storage."""
        try:
            # Serialize credentials
            data = json.dumps(self.credentials, indent=2)
            encrypted_data = self.cipher_suite.encrypt(data.encode())

            # Write to temporary file first, then rename (atomic operation)
            temp_file = self.credentials_file.with_suffix(".tmp")
            with open(temp_file, "wb") as f:
                f.write(encrypted_data)

            # Set restrictive permissions
            os.chmod(temp_file, 0o600)

            # Atomic rename
            temp_file.replace(self.credentials_file)

            # Update metadata
            self._save_metadata()

            logger.info(
                "Saved encrypted credentials",
                {"credential_count": len(self.credentials)},
            )

        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
            raise

    def _save_metadata(self):
        """Save credential metadata (non-sensitive information)."""
        metadata = {
            "last_updated": datetime.utcnow().isoformat(),
            "credential_count": len(self.credentials),
            "credentials": {
                name: {
                    "created": cred.get("created"),
                    "last_accessed": cred.get("last_accessed"),
                    "access_count": cred.get("access_count", 0),
                    "type": cred.get("type", "unknown"),
                }
                for name, cred in self.credentials.items()
            },
        }

        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    async def store_credential(
        self,
        name: str,
        value: Union[str, Dict[str, Any]],
        credential_type: str = "api_key",
        context: Optional[SecurityContext] = None,
        metadata: Optional[Dict[str, Any]] = None,
        provider: str = "local",
        sync_to_vault: bool = False,
    ) -> bool:
        """Store a credential securely in specified provider."""
        try:
            # Validate inputs
            if not name or not value:
                raise ValueError("Credential name and value are required")

            # Convert value to string for external providers
            value_str = json.dumps(value) if isinstance(value, dict) else str(value)

            success = False
            providers_used = []

            # Store in specified provider
            if provider == "vault" and self.key_vault_provider:
                success = await self.key_vault_provider.set_secret(name, value_str)
                if success:
                    providers_used.append("vault")
            elif provider == "environment" and self.env_provider:
                success = await self.env_provider.set_credential(name, value_str)
                if success:
                    providers_used.append("environment")
            elif provider == "local":
                # Prepare credential data
                credential_data = {
                    "value": value,
                    "type": credential_type,
                    "created": datetime.utcnow().isoformat(),
                    "last_accessed": None,
                    "access_count": 0,
                    "metadata": metadata or {},
                }

                # Store credential locally
                self.credentials[name] = credential_data
                self._save_credentials()
                success = True
                providers_used.append("local")

            # Optionally sync to vault
            if (
                success
                and sync_to_vault
                and provider != "vault"
                and self.key_vault_provider
            ):
                vault_success = await self.key_vault_provider.set_secret(
                    name, value_str
                )
                if vault_success:
                    providers_used.append("vault")

            if success:
                # Audit log
                await self._log_audit_event(
                    "credential_stored",
                    SecurityLevel.MEDIUM,
                    context,
                    {
                        "credential_name": name,
                        "credential_type": credential_type,
                        "providers": providers_used,
                        "action": "store",
                    },
                )
                return True
            else:
                raise ValueError(f"Failed to store in provider: {provider}")

        except Exception as e:
            logger.error(f"Failed to store credential '{name}': {e}")
            await self._log_audit_event(
                "credential_store_failed",
                SecurityLevel.HIGH,
                context,
                {
                    "credential_name": name,
                    "provider": provider,
                    "error": str(e),
                    "action": "store_failed",
                },
            )
            return False

    async def retrieve_credential(
        self,
        name: str,
        context: Optional[SecurityContext] = None,
        provider_preference: Optional[str] = None,
    ) -> Optional[Union[str, Dict[str, Any]]]:
        """Retrieve a credential securely with provider fallback."""
        providers_tried = []

        try:
            # Try providers in order of preference
            providers = self._get_provider_order(provider_preference)

            for provider_name in providers:
                try:
                    value = None
                    providers_tried.append(provider_name)

                    if provider_name == "vault" and self.key_vault_provider:
                        value = await self.key_vault_provider.get_secret(name)
                        if value:
                            await self._log_audit_event(
                                "credential_accessed_vault",
                                SecurityLevel.LOW,
                                context,
                                {
                                    "credential_name": name,
                                    "provider": "vault",
                                    "action": "retrieve",
                                },
                            )
                            return value

                    elif provider_name == "environment" and self.env_provider:
                        value = await self.env_provider.get_credential(name)
                        if value:
                            await self._log_audit_event(
                                "credential_accessed_env",
                                SecurityLevel.LOW,
                                context,
                                {
                                    "credential_name": name,
                                    "provider": "environment",
                                    "action": "retrieve",
                                },
                            )
                            return value

                    elif provider_name == "local":
                        if name in self.credentials:
                            # Update access tracking
                            credential = self.credentials[name]
                            credential["last_accessed"] = datetime.utcnow().isoformat()
                            credential["access_count"] = (
                                credential.get("access_count", 0) + 1
                            )

                            # Save updated metadata
                            self._save_credentials()

                            # Audit log
                            await self._log_audit_event(
                                "credential_accessed_local",
                                SecurityLevel.LOW,
                                context,
                                {
                                    "credential_name": name,
                                    "provider": "local",
                                    "action": "retrieve",
                                },
                            )
                            return credential["value"]

                except Exception as e:
                    logger.error(f"Error retrieving from provider {provider_name}: {e}")
                    continue

            # No credential found in any provider
            await self._log_audit_event(
                "credential_not_found",
                SecurityLevel.LOW,
                context,
                {
                    "credential_name": name,
                    "providers_tried": providers_tried,
                    "action": "retrieve_failed",
                },
            )
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve credential '{name}': {e}")
            await self._log_audit_event(
                "credential_retrieve_error",
                SecurityLevel.MEDIUM,
                context,
                {
                    "credential_name": name,
                    "providers_tried": providers_tried,
                    "error": str(e),
                    "action": "retrieve_error",
                },
            )
            return None

    def _get_provider_order(self, preference: Optional[str] = None) -> List[str]:
        """Get provider order based on preference."""
        if preference:
            if preference == "vault" and self.key_vault_provider:
                return ["vault", "local", "environment"]
            elif preference == "environment" and self.env_provider:
                return ["environment", "vault", "local"]
            elif preference == "local":
                return ["local", "vault", "environment"]

        # Default order: vault -> local -> environment
        providers = []
        if self.key_vault_provider:
            providers.append("vault")
        providers.append("local")
        if self.env_provider:
            providers.append("environment")

        return providers

    async def delete_credential(
        self,
        name: str,
        context: Optional[SecurityContext] = None,
        provider: str = "all",
    ) -> bool:
        """Delete a credential from specified provider(s)."""
        try:
            if not name:
                raise ValueError("Credential name is required")

            success = False
            providers_used = []

            # Delete from specified provider(s)
            if provider == "all" or provider == "local":
                if name in self.credentials:
                    del self.credentials[name]
                    self._save_credentials()
                    success = True
                    providers_used.append("local")

            if provider == "all" or provider == "vault":
                if self.key_vault_provider:
                    vault_success = await self.key_vault_provider.delete_secret(name)
                    if vault_success:
                        success = True
                        providers_used.append("vault")

            if provider == "all" or provider == "environment":
                if self.env_provider:
                    # Environment variables can't be truly deleted at runtime
                    # but we can set them to empty
                    env_success = await self.env_provider.set_credential(name, "")
                    if env_success:
                        success = True
                        providers_used.append("environment")

            if success:
                await self._log_audit_event(
                    "credential_deleted",
                    SecurityLevel.MEDIUM,
                    context,
                    {
                        "credential_name": name,
                        "providers": providers_used,
                        "action": "delete",
                    },
                )
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Failed to delete credential '{name}': {e}")
            await self._log_audit_event(
                "credential_delete_failed",
                SecurityLevel.HIGH,
                context,
                {
                    "credential_name": name,
                    "provider": provider,
                    "error": str(e),
                    "action": "delete_failed",
                },
            )
            return False

    async def list_credentials(
        self, context: Optional[SecurityContext] = None, include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """List all credentials (without sensitive values)."""
        try:
            credentials_list = []

            for name, cred in self.credentials.items():
                cred_info = {
                    "name": name,
                    "type": cred.get("type", "unknown"),
                    "created": cred.get("created"),
                    "last_accessed": cred.get("last_accessed"),
                    "access_count": cred.get("access_count", 0),
                }

                if include_metadata:
                    cred_info["metadata"] = cred.get("metadata", {})

                credentials_list.append(cred_info)

            await self._log_audit_event(
                "credentials_listed",
                SecurityLevel.LOW,
                context,
                {
                    "credential_count": len(credentials_list),
                    "action": "list",
                },
            )

            return credentials_list

        except Exception as e:
            logger.error(f"Failed to list credentials: {e}")
            await self._log_audit_event(
                "credentials_list_failed",
                SecurityLevel.MEDIUM,
                context,
                {
                    "error": str(e),
                    "action": "list_failed",
                },
            )
            return []

    async def rotate_credential(
        self, name: str, new_value: str, context: Optional[SecurityContext] = None
    ) -> bool:
        """Rotate a credential with a new value."""
        try:
            if not name or not new_value:
                raise ValueError("Credential name and new value are required")

            if name not in self.credentials:
                raise ValueError(f"Credential '{name}' not found")

            # Store old value for rollback if needed
            old_credential = self.credentials[name].copy()

            # Update credential
            self.credentials[name]["value"] = new_value
            self.credentials[name]["last_rotated"] = datetime.utcnow().isoformat()
            self.credentials[name]["rotation_count"] = (
                self.credentials[name].get("rotation_count", 0) + 1
            )

            # Save changes
            self._save_credentials()

            # Sync to vault if configured
            if self.key_vault_provider:
                vault_success = await self.key_vault_provider.set_secret(
                    name, new_value
                )
                if not vault_success:
                    logger.warning(
                        f"Failed to sync rotated credential to vault: {name}"
                    )

            await self._log_audit_event(
                "credential_rotated",
                SecurityLevel.MEDIUM,
                context,
                {
                    "credential_name": name,
                    "rotation_count": self.credentials[name]["rotation_count"],
                    "action": "rotate",
                },
            )

            return True

        except Exception as e:
            logger.error(f"Failed to rotate credential '{name}': {e}")
            await self._log_audit_event(
                "credential_rotation_failed",
                SecurityLevel.HIGH,
                context,
                {
                    "credential_name": name,
                    "error": str(e),
                    "action": "rotate_failed",
                },
            )
            return False

    async def backup_credentials(
        self,
        backup_path: Optional[Path] = None,
        context: Optional[SecurityContext] = None,
    ) -> bool:
        """Create encrypted backup of credentials."""
        try:
            if backup_path is None:
                backup_path = (
                    self.storage_path
                    / f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.enc"
                )

            # Create backup data
            backup_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "credentials": self.credentials,
                "metadata": {
                    "version": "1.0",
                    "credential_count": len(self.credentials),
                },
            }

            # Encrypt backup
            encrypted_backup = self.cipher_suite.encrypt(
                json.dumps(backup_data, indent=2).encode()
            )

            # Write backup file
            with open(backup_path, "wb") as f:
                f.write(encrypted_backup)

            # Set restrictive permissions
            os.chmod(backup_path, 0o600)

            await self._log_audit_event(
                "credentials_backed_up",
                SecurityLevel.MEDIUM,
                context,
                {
                    "backup_path": str(backup_path),
                    "credential_count": len(self.credentials),
                    "action": "backup",
                },
            )

            logger.info(f"Created credential backup: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create credential backup: {e}")
            await self._log_audit_event(
                "credentials_backup_failed",
                SecurityLevel.HIGH,
                context,
                {
                    "backup_path": str(backup_path) if backup_path else None,
                    "error": str(e),
                    "action": "backup_failed",
                },
            )
            return False

    async def restore_credentials(
        self, backup_path: Path, context: Optional[SecurityContext] = None
    ) -> bool:
        """Restore credentials from encrypted backup."""
        try:
            if not backup_path.exists():
                raise ValueError(f"Backup file not found: {backup_path}")

            # Read and decrypt backup
            with open(backup_path, "rb") as f:
                encrypted_backup = f.read()

            decrypted_data = self.cipher_suite.decrypt(encrypted_backup)
            backup_data = json.loads(decrypted_data.decode())

            # Validate backup format
            if "credentials" not in backup_data:
                raise ValueError("Invalid backup format")

            # Store current credentials for rollback
            original_credentials = self.credentials.copy()

            try:
                # Restore credentials
                self.credentials = backup_data["credentials"]
                self._save_credentials()

                await self._log_audit_event(
                    "credentials_restored",
                    SecurityLevel.HIGH,
                    context,
                    {
                        "backup_path": str(backup_path),
                        "restored_count": len(self.credentials),
                        "backup_timestamp": backup_data.get("timestamp"),
                        "action": "restore",
                    },
                )

                logger.info(f"Restored credentials from backup: {backup_path}")
                return True

            except Exception as e:
                # Rollback on failure
                self.credentials = original_credentials
                self._save_credentials()
                raise e

        except Exception as e:
            logger.error(f"Failed to restore credentials from backup: {e}")
            await self._log_audit_event(
                "credentials_restore_failed",
                SecurityLevel.HIGH,
                context,
                {
                    "backup_path": str(backup_path),
                    "error": str(e),
                    "action": "restore_failed",
                },
            )
            return False

    def get_credential_stats(self) -> Dict[str, Any]:
        """Get statistics about stored credentials."""
        if not self.credentials:
            return {
                "total_credentials": 0,
                "by_type": {},
                "access_stats": {
                    "never_accessed": 0,
                    "most_accessed": ("", 0),
                    "least_accessed": ("", 0),
                },
                "age_stats": {
                    "oldest": None,
                    "newest": None,
                },
            }

        # Count by type
        by_type = {}
        access_counts = []
        creation_dates = []

        for name, cred in self.credentials.items():
            cred_type = cred.get("type", "unknown")
            by_type[cred_type] = by_type.get(cred_type, 0) + 1

            access_count = cred.get("access_count", 0)
            access_counts.append((name, access_count))

            if cred.get("created"):
                creation_dates.append((name, cred["created"]))

        # Access statistics
        never_accessed = sum(1 for _, count in access_counts if count == 0)
        most_accessed = (
            max(access_counts, key=lambda x: x[1]) if access_counts else ("", 0)
        )
        least_accessed = (
            min(access_counts, key=lambda x: x[1]) if access_counts else ("", 0)
        )

        # Age statistics
        oldest = min(creation_dates, key=lambda x: x[1]) if creation_dates else None
        newest = max(creation_dates, key=lambda x: x[1]) if creation_dates else None

        stats = {
            "total_credentials": len(self.credentials),
            "by_type": by_type,
            "access_stats": {
                "never_accessed": never_accessed,
                "most_accessed": most_accessed,
                "least_accessed": least_accessed,
            },
            "age_stats": {
                "oldest": oldest,
                "newest": newest,
            },
        }

        return stats

    async def setup_credential_rotation(
        self,
        credential_name: str,
        rotation_interval_days: int = 30,
        auto_rotate: bool = False,
        rotation_callback: Optional[callable] = None,
        context: Optional[SecurityContext] = None,
    ) -> bool:
        """Setup automatic rotation for a credential."""
        try:
            if credential_name not in self.credentials:
                raise ValueError(f"Credential '{credential_name}' not found")

            rotation_interval = timedelta(days=rotation_interval_days)

            await self.rotation_manager.set_rotation_policy(
                credential_name, rotation_interval, rotation_callback, auto_rotate
            )

            await self._log_audit_event(
                "rotation_policy_set",
                SecurityLevel.MEDIUM,
                context,
                {
                    "credential_name": credential_name,
                    "rotation_interval_days": rotation_interval_days,
                    "auto_rotate": auto_rotate,
                    "action": "setup_rotation",
                },
            )

            return True

        except Exception as e:
            logger.error(f"Failed to setup rotation for '{credential_name}': {e}")
            await self._log_audit_event(
                "rotation_setup_failed",
                SecurityLevel.HIGH,
                context,
                {
                    "credential_name": credential_name,
                    "error": str(e),
                    "action": "setup_rotation_failed",
                },
            )
            return False

    async def check_and_rotate_credentials(
        self, context: Optional[SecurityContext] = None
    ) -> Dict[str, bool]:
        """Check and rotate credentials that need rotation."""
        results = {}

        try:
            for credential_name in self.rotation_manager.rotation_policies:
                if await self.rotation_manager.check_rotation_needed(credential_name):
                    policy = self.rotation_manager.rotation_policies[credential_name]

                    if policy.get("auto_rotate", False):
                        success = await self.rotation_manager.rotate_credential(
                            credential_name, context
                        )
                        results[credential_name] = success
                    else:
                        logger.info(
                            f"Credential '{credential_name}' needs rotation but auto_rotate is disabled"
                        )
                        results[credential_name] = False

            return results

        except Exception as e:
            logger.error(f"Error during credential rotation check: {e}")
            return results

    async def export_credentials_secure(
        self,
        export_path: Path,
        include_values: bool = False,
        context: Optional[SecurityContext] = None,
    ) -> bool:
        """Export credentials in encrypted format."""
        try:
            export_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "include_values": include_values,
                "credentials": {},
            }

            for name, cred in self.credentials.items():
                cred_export = {
                    "type": cred.get("type"),
                    "created": cred.get("created"),
                    "last_accessed": cred.get("last_accessed"),
                    "access_count": cred.get("access_count", 0),
                    "metadata": cred.get("metadata", {}),
                }

                if include_values:
                    cred_export["value"] = cred["value"]

                export_data["credentials"][name] = cred_export

            # Encrypt export data
            encrypted_export = self.cipher_suite.encrypt(
                json.dumps(export_data, indent=2).encode()
            )

            with open(export_path, "wb") as f:
                f.write(encrypted_export)

            os.chmod(export_path, 0o600)

            await self._log_audit_event(
                "credentials_exported",
                SecurityLevel.HIGH if include_values else SecurityLevel.MEDIUM,
                context,
                {
                    "export_path": str(export_path),
                    "include_values": include_values,
                    "credential_count": len(self.credentials),
                    "action": "export",
                },
            )

            return True

        except Exception as e:
            logger.error(f"Failed to export credentials: {e}")
            await self._log_audit_event(
                "credentials_export_failed",
                SecurityLevel.HIGH,
                context,
                {
                    "export_path": str(export_path),
                    "error": str(e),
                    "action": "export_failed",
                },
            )
            return False

    async def get_credential_health_report(self) -> Dict[str, Any]:
        """Generate a health report for all credentials."""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "statistics": self.get_credential_stats(),
            "security_issues": [],
            "health_score": 100.0,
            "recommendations": [],
        }

        issues = []
        recommendations = []
        health_score = 100.0

        # Check for old credentials (> 1 year)
        one_year_ago = datetime.utcnow() - timedelta(days=365)

        for name, cred in self.credentials.items():
            created_str = cred.get("created")
            if created_str:
                try:
                    created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                    if created < one_year_ago:
                        issues.append(
                            {
                                "type": "old_credential",
                                "credential": name,
                                "age_days": (datetime.utcnow() - created).days,
                                "severity": "medium",
                            }
                        )
                        health_score -= 5
                except ValueError:
                    pass

            # Check for unused credentials
            if cred.get("access_count", 0) == 0:
                issues.append(
                    {"type": "unused_credential", "credential": name, "severity": "low"}
                )
                health_score -= 2

            # Check for credentials without rotation policies
            if name not in self.rotation_manager.rotation_policies:
                recommendations.append(
                    {
                        "type": "setup_rotation",
                        "credential": name,
                        "message": f"Consider setting up rotation policy for '{name}'",
                    }
                )

        # Check rotation status
        rotation_status = await self.rotation_manager.get_rotation_status()
        for name, status in rotation_status.items():
            if status.get("needs_rotation", False):
                issues.append(
                    {
                        "type": "rotation_needed",
                        "credential": name,
                        "severity": "medium",
                    }
                )
                health_score -= 10

        report["security_issues"] = issues
        report["health_score"] = max(0, health_score)
        report["recommendations"] = recommendations

        return report

    async def get_audit_events(
        self,
        limit: Optional[int] = None,
        event_types: Optional[List[str]] = None,
        severity_filter: Optional[SecurityLevel] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[SecurityEvent]:
        """Get audit events with optional filtering."""
        events = self.audit_events.copy()

        # Filter by event types
        if event_types:
            events = [e for e in events if e.event_type in event_types]

        # Filter by time range
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        # Filter by severity
        if severity_filter:
            events = [e for e in events if e.severity == severity_filter]

        # Sort by timestamp (newest first)
        events.sort(key=lambda e: e.timestamp, reverse=True)

        if limit:
            events = events[:limit]

        return events

    async def _log_audit_event(
        self,
        event_type: str,
        severity: SecurityLevel,
        context: Optional[SecurityContext],
        details: Dict[str, Any],
    ):
        """Log a credential audit event."""
        event = SecurityEvent(
            event_id=secrets.token_urlsafe(16),
            timestamp=datetime.utcnow(),
            event_type=event_type,
            severity=severity,
            source_ip=context.ip_address if context else None,
            user_id=context.user_id if context else None,
            session_id=context.session_id if context else None,
            operation=context.operation if context else None,
            details=details,
        )

        self.audit_events.append(event)

        # Keep only recent events to prevent memory issues
        if len(self.audit_events) > 1000:
            self.audit_events = self.audit_events[-500:]

        # Log to application logger
        log_level = (
            "error"
            if severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]
            else "info"
        )
        getattr(logger, log_level)(
            f"Credential audit: {event_type} | Severity: {severity.value} | "
            f"User: {context.user_id if context else 'system'} | "
            f"Details: {details}"
        )
