"""Security data models and enums."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel


class SecurityLevel(Enum):
    """Security levels for operations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats."""

    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    MALICIOUS_CONTENT = "malicious_content"


class ValidationStatus(Enum):
    """Input validation status."""

    VALID = "valid"
    INVALID = "invalid"
    SUSPICIOUS = "suspicious"
    BLOCKED = "blocked"


@dataclass
class SecurityContext:
    """Security context for operations."""

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    operation: Optional[str] = None
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of input validation."""

    status: ValidationStatus
    sanitized_input: Optional[str] = None
    threats_detected: List[ThreatType] = field(default_factory=list)
    risk_score: float = 0.0
    message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    max_requests: int
    time_window: timedelta
    burst_limit: Optional[int] = None
    block_duration: timedelta = field(default_factory=lambda: timedelta(minutes=15))


@dataclass
class RateLimitState:
    """Current rate limit state for a client."""

    request_count: int = 0
    window_start: datetime = field(default_factory=datetime.utcnow)
    blocked_until: Optional[datetime] = None
    burst_count: int = 0


class SecurityEvent(BaseModel):
    """Security event for logging and monitoring."""

    event_id: str
    timestamp: datetime
    event_type: str
    severity: SecurityLevel
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    operation: Optional[str] = None
    threat_type: Optional[ThreatType] = None
    details: Dict[str, Any] = {}
    blocked: bool = False

    class Config:
        use_enum_values = True


@dataclass
class SecurityPolicy:
    """Security policy configuration."""

    max_input_length: int = 10000
    allowed_file_extensions: Set[str] = field(
        default_factory=lambda: {
            ".txt",
            ".md",
            ".py",
            ".js",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
        }
    )
    blocked_patterns: List[str] = field(default_factory=list)
    rate_limits: Dict[str, RateLimitConfig] = field(default_factory=dict)
    enable_xss_protection: bool = True
    enable_sql_injection_protection: bool = True
    enable_command_injection_protection: bool = True
    enable_path_traversal_protection: bool = True
    log_all_events: bool = True
    block_suspicious_requests: bool = True
