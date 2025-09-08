# Credential Management Implementation (Task 5.2)

## Overview

This document describes the implementation of secure credential management for OpenManus, fulfilling the requirements of task 5.2: "Implement Secure Credential Management".

## Features Implemented

### 1. Secure Credential Storage Using Environment Variables and Key Vaults

#### Environment Variable Provider
- **Class**: `EnvironmentCredentialProvider`
- **Features**:
  - Secure storage and retrieval from environment variables
  - Configurable prefix (default: `OPENMANUS_`)
  - Runtime credential setting and listing
  - Automatic credential name normalization

#### Key Vault Providers
- **AWS Secrets Manager Provider**: `AWSSecretsManagerProvider`
  - Integration with AWS Secrets Manager
  - Automatic secret creation and updates
  - Secure deletion and listing capabilities

- **HashiCorp Vault Provider**: `HashiCorpVaultProvider`
  - Integration with HashiCorp Vault
  - Token-based authentication
  - RESTful API communication

### 2. Encryption for Sensitive Data at Rest and in Transit

#### At Rest Encryption
- **Algorithm**: Fernet (AES 128 in CBC mode with HMAC-SHA256)
- **Key Derivation**: PBKDF2-HMAC-SHA256 with 100,000 iterations
- **Features**:
  - Master key generation and secure storage
  - Encrypted credential files with atomic writes
  - Secure file permissions (0o600)

#### In Transit Encryption
- **Hybrid Encryption**: RSA + AES for optimal security and performance
- **RSA Key Size**: 2048 bits
- **AES Mode**: CBC with random IV
- **Features**:
  - Automatic key pair generation
  - Secure key storage with encryption
  - Fallback to Fernet for compatibility

### 3. Secure API Key Rotation and Management

#### Rotation Manager
- **Class**: `CredentialRotationManager`
- **Features**:
  - Configurable rotation intervals
  - Automatic and manual rotation modes
  - Custom rotation callbacks
  - Rotation status tracking and reporting

#### Rotation Policies
- **Interval-based**: Time-based rotation (days, hours, etc.)
- **Event-driven**: Custom callback functions
- **Automatic**: Background rotation with configurable policies
- **Manual**: On-demand rotation with audit trails

### 4. Audit Logging for Credential Access and Usage

#### Security Events
- **Event Types**:
  - `credential_stored`: When credentials are stored
  - `credential_accessed_*`: When credentials are retrieved
  - `credential_rotated`: When credentials are rotated
  - `credential_deleted`: When credentials are removed
  - `credential_*_failed`: When operations fail

#### Audit Features
- **Structured Logging**: JSON-formatted audit events
- **Context Tracking**: User ID, session ID, IP address
- **Severity Levels**: LOW, MEDIUM, HIGH, CRITICAL
- **Event Filtering**: By type, severity, time range
- **Memory Management**: Automatic cleanup of old events

## Security Features

### 1. Multiple Provider Support
- **Local Storage**: Encrypted file-based storage
- **Environment Variables**: Runtime credential access
- **Key Vaults**: External secure storage (AWS, Vault)
- **Provider Fallback**: Automatic failover between providers

### 2. Access Control
- **Security Context**: User and session tracking
- **Operation Validation**: Input sanitization and validation
- **Permission Checks**: Provider-specific access controls
- **Rate Limiting**: Protection against abuse

### 3. Data Protection
- **Encryption at Rest**: All stored credentials encrypted
- **Encryption in Transit**: Secure data transmission
- **Secure Deletion**: Proper cleanup of sensitive data
- **Memory Protection**: Minimal sensitive data in memory

## Usage Examples

### Basic Credential Management

```python
from app.security.credentials import CredentialManager
from app.security.models import SecurityContext

# Initialize credential manager
manager = CredentialManager(
    master_key="your-master-key",
    enable_environment_fallback=True
)

# Create security context
context = SecurityContext(
    user_id="user123",
    session_id="session456",
    ip_address="192.168.1.1",
    operation="store_api_key"
)

# Store a credential
await manager.store_credential(
    "openai_api_key",
    "sk-1234567890abcdef",
    credential_type="api_key",
    context=context
)

# Retrieve a credential
api_key = await manager.retrieve_credential(
    "openai_api_key",
    context=context
)

# Rotate a credential
await manager.rotate_credential(
    "openai_api_key",
    "sk-new1234567890abcdef",
    context=context
)
```

### Environment Variable Storage

```python
from app.security.credentials import EnvironmentCredentialProvider

# Initialize environment provider
env_provider = EnvironmentCredentialProvider(prefix="MYAPP_")

# Store credential in environment
await env_provider.set_credential("api_key", "secret_value")

# Retrieve credential from environment
value = await env_provider.get_credential("api_key")
# This will look for MYAPP_API_KEY environment variable
```

### Key Vault Integration

```python
from app.security.credentials import AWSSecretsManagerProvider, CredentialManager

# Initialize AWS Secrets Manager provider
aws_provider = AWSSecretsManagerProvider(region="us-east-1")

# Create credential manager with vault support
manager = CredentialManager(
    key_vault_provider=aws_provider,
    enable_environment_fallback=True
)

# Store credential in vault
await manager.store_credential(
    "database_password",
    "super_secret_password",
    provider="vault",
    context=context
)
```

### Credential Rotation Setup

```python
# Setup automatic rotation
await manager.setup_credential_rotation(
    "api_key",
    rotation_interval_days=30,
    auto_rotate=True,
    context=context
)

# Check rotation status
status = await manager.rotation_manager.get_rotation_status()
print(f"Rotation needed: {status['api_key']['needs_rotation']}")

# Manual rotation check and execution
results = await manager.check_and_rotate_credentials(context=context)
```

### Audit Logging

```python
# Get audit events
events = await manager.get_audit_events(
    limit=10,
    event_types=["credential_stored", "credential_accessed_local"]
)

# Filter by severity
high_severity_events = await manager.get_audit_events(
    severity_filter=SecurityLevel.HIGH
)

# Get credential health report
health_report = await manager.get_credential_health_report()
print(f"Health score: {health_report['health_score']}")
```

## Configuration

### Environment Variables

```bash
# Master encryption key
OPENMANUS_MASTER_KEY=your-base64-encoded-key

# AWS credentials (for Secrets Manager)
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-east-1

# Vault configuration
VAULT_TOKEN=your-vault-token
VAULT_ADDR=https://vault.example.com
```

### File Structure

```
~/.openmanus/credentials/
├── credentials.enc          # Encrypted credential storage
├── metadata.json           # Non-sensitive metadata
├── transit_keys.enc        # Transit encryption keys
└── .env                    # Master key storage
```

## Security Considerations

### 1. Key Management
- Master keys are generated securely using Fernet
- Keys are stored with restrictive file permissions
- Key rotation is supported for long-term security

### 2. Provider Security
- Each provider implements its own security model
- Fallback providers ensure availability
- Provider-specific encryption and authentication

### 3. Audit and Monitoring
- All credential operations are logged
- Security events include context and metadata
- Failed operations are tracked and alerted

### 4. Data Protection
- Sensitive data is encrypted at rest and in transit
- Memory exposure is minimized
- Secure deletion prevents data recovery

## Testing

The implementation includes comprehensive tests covering:

- ✅ Environment variable storage and retrieval
- ✅ Encryption at rest using Fernet
- ✅ Encryption in transit using hybrid RSA+AES
- ✅ API key rotation and management
- ✅ Audit logging for all operations
- ✅ Secure file permissions
- ✅ Provider fallback mechanisms
- ✅ Error handling and recovery

## Requirements Compliance

This implementation fulfills all requirements from task 5.2:

- ✅ **Create secure credential storage using environment variables and key vaults**
  - Multiple provider support (environment, AWS, Vault)
  - Secure storage with encryption
  - Provider fallback mechanisms

- ✅ **Implement encryption for sensitive data at rest and in transit**
  - Fernet encryption for at-rest data
  - Hybrid RSA+AES for transit encryption
  - Secure key management

- ✅ **Add secure API key rotation and management**
  - Automated rotation policies
  - Manual rotation capabilities
  - Rotation tracking and reporting

- ✅ **Create audit logging for credential access and usage**
  - Comprehensive event logging
  - Security context tracking
  - Event filtering and reporting

The implementation provides a production-ready credential management system that meets enterprise security standards while maintaining usability and performance.
