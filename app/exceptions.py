import json
import traceback
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Union
from uuid import uuid4


class ErrorClassification(Enum):
    """Classification of different types of errors for handling strategies"""

    TRANSIENT = "transient"  # Network timeouts, temporary API failures
    PERMANENT = "permanent"  # Invalid configuration, authentication failures
    RESOURCE = "resource"  # Memory limits, disk space issues
    SECURITY = "security"  # Permission denied, malicious input detected
    VALIDATION = "validation"  # Input validation failures
    BUSINESS = "business"  # Business logic violations


class ErrorContext:
    """Comprehensive error context for debugging and monitoring"""

    def __init__(
        self,
        operation: str,
        agent_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        request_id: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None,
        system_state: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ):
        self.operation = operation
        self.agent_id = agent_id
        self.tool_name = tool_name
        self.request_id = request_id or str(uuid4())
        self.correlation_id = correlation_id or str(uuid4())
        self.timestamp = datetime.now(timezone.utc)
        self.user_context = user_context or {}
        self.system_state = system_state or {}
        self.stack_trace = traceback.format_stack()

    def to_dict(self) -> Dict[str, Any]:
        """Convert error context to dictionary for serialization"""
        return {
            "operation": self.operation,
            "agent_id": self.agent_id,
            "tool_name": self.tool_name,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "user_context": self.user_context,
            "system_state": self.system_state,
            "stack_trace": self.stack_trace,
        }

    def to_json(self) -> str:
        """Convert error context to JSON string"""
        return json.dumps(self.to_dict(), default=str, indent=2)


class OpenManusError(Exception):
    """Base exception for all OpenManus errors with enhanced context"""

    def __init__(
        self,
        message: str,
        classification: ErrorClassification = ErrorClassification.PERMANENT,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = False,
        retry_after: Optional[int] = None,
    ):
        super().__init__(message)
        self.message = message
        self.classification = classification
        self.context = context
        self.cause = cause
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization"""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "classification": self.classification.value,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context.to_dict() if self.context else None,
            "cause": str(self.cause) if self.cause else None,
        }

    def to_json(self) -> str:
        """Convert exception to JSON string for logging"""
        return json.dumps(self.to_dict(), default=str, indent=2)


# Configuration and Validation Errors
class ConfigurationError(OpenManusError):
    """Raised when configuration is invalid or missing"""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(
            message, classification=ErrorClassification.PERMANENT, **kwargs
        )
        self.field = field


class ValidationError(OpenManusError):
    """Raised when input validation fails"""

    def __init__(
        self, message: str, field: Optional[str] = None, value: Any = None, **kwargs
    ):
        super().__init__(
            message, classification=ErrorClassification.VALIDATION, **kwargs
        )
        self.field = field
        self.value = value


# Tool and Agent Errors
class ToolError(OpenManusError):
    """Raised when a tool encounters an error"""

    def __init__(self, message: str, tool_name: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            classification=ErrorClassification.TRANSIENT,
            recoverable=True,
            **kwargs,
        )
        self.tool_name = tool_name


class AgentError(OpenManusError):
    """Raised when an agent encounters an error"""

    def __init__(self, message: str, agent_id: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.agent_id = agent_id


# LLM and API Errors
class LLMError(OpenManusError):
    """Raised when LLM operations fail"""

    def __init__(self, message: str, provider: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            classification=ErrorClassification.TRANSIENT,
            recoverable=True,
            **kwargs,
        )
        self.provider = provider


class TokenLimitExceeded(LLMError):
    """Exception raised when the token limit is exceeded"""

    def __init__(self, message: str = "Token limit exceeded", **kwargs):
        super().__init__(message, classification=ErrorClassification.RESOURCE, **kwargs)


class APIError(OpenManusError):
    """Raised when external API calls fail"""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        **kwargs,
    ):
        classification = (
            ErrorClassification.TRANSIENT
            if status_code and 500 <= status_code < 600
            else ErrorClassification.PERMANENT
        )
        super().__init__(
            message,
            classification=classification,
            recoverable=classification == ErrorClassification.TRANSIENT,
            **kwargs,
        )
        self.status_code = status_code
        self.endpoint = endpoint


class RateLimitError(APIError):
    """Raised when API rate limits are exceeded"""

    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(
            message,
            classification=ErrorClassification.TRANSIENT,
            recoverable=True,
            retry_after=retry_after,
            **kwargs,
        )


# Resource and System Errors
class ResourceError(OpenManusError):
    """Raised when system resources are exhausted"""

    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        super().__init__(message, classification=ErrorClassification.RESOURCE, **kwargs)
        self.resource_type = resource_type


class MemoryError(ResourceError):
    """Raised when memory limits are exceeded"""

    def __init__(self, message: str = "Memory limit exceeded", **kwargs):
        super().__init__(message, resource_type="memory", **kwargs)


class DiskSpaceError(ResourceError):
    """Raised when disk space is insufficient"""

    def __init__(self, message: str = "Insufficient disk space", **kwargs):
        super().__init__(message, resource_type="disk", **kwargs)


# Security Errors
class SecurityError(OpenManusError):
    """Raised when security violations are detected"""

    def __init__(self, message: str, violation_type: Optional[str] = None, **kwargs):
        super().__init__(message, classification=ErrorClassification.SECURITY, **kwargs)
        self.violation_type = violation_type


class PermissionError(SecurityError):
    """Raised when permission is denied"""

    def __init__(self, message: str, resource: Optional[str] = None, **kwargs):
        super().__init__(message, violation_type="permission", **kwargs)
        self.resource = resource


class SandboxError(SecurityError):
    """Raised when sandbox operations fail"""

    def __init__(self, message: str, sandbox_id: Optional[str] = None, **kwargs):
        super().__init__(message, violation_type="sandbox", **kwargs)
        self.sandbox_id = sandbox_id


# Network and Connectivity Errors
class NetworkError(OpenManusError):
    """Raised when network operations fail"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            classification=ErrorClassification.TRANSIENT,
            recoverable=True,
            **kwargs,
        )


class ConnectionError(NetworkError):
    """Raised when connection to external services fails"""

    def __init__(self, message: str, service: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.service = service


class TimeoutError(NetworkError):
    """Raised when operations timeout"""

    def __init__(
        self, message: str, timeout_duration: Optional[float] = None, **kwargs
    ):
        super().__init__(message, **kwargs)
        self.timeout_duration = timeout_duration


# Circuit Breaker Errors
class CircuitBreakerError(OpenManusError):
    """Raised when circuit breaker is open"""

    def __init__(self, message: str, service: str, **kwargs):
        super().__init__(
            message,
            classification=ErrorClassification.TRANSIENT,
            recoverable=True,
            **kwargs,
        )
        self.service = service


class CircuitBreakerOpenError(CircuitBreakerError):
    """Raised when circuit breaker is in open state"""

    def __init__(self, service: str, **kwargs):
        super().__init__(
            f"Circuit breaker is open for service: {service}", service=service, **kwargs
        )


# Research and Analysis Errors
class ResearchError(OpenManusError):
    """Raised when research operations fail"""

    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            classification=ErrorClassification.TRANSIENT,
            recoverable=True,
            **kwargs,
        )
        self.query = query


class ToolExecutionError(ToolError):
    """Raised when tool execution fails"""

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        exit_code: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, tool_name=tool_name, **kwargs)
        self.exit_code = exit_code
