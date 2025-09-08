"""Security Manager for comprehensive security controls."""

import asyncio
import hashlib
import hmac
import re
import secrets
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from urllib.parse import unquote

from app.logger import define_log_level
from app.security.models import (
    RateLimitConfig,
    RateLimitState,
    SecurityContext,
    SecurityEvent,
    SecurityLevel,
    SecurityPolicy,
    ThreatType,
    ValidationResult,
    ValidationStatus,
)

logger = define_log_level()


class SecurityManager:
    """Comprehensive security manager for input validation, rate limiting, and threat detection."""

    def __init__(self, policy: Optional[SecurityPolicy] = None):
        """Initialize security manager with policy."""
        self.policy = policy or SecurityPolicy()
        self.rate_limit_states: Dict[str, RateLimitState] = defaultdict(RateLimitState)
        self.blocked_ips: Set[str] = set()
        self.security_events: List[SecurityEvent] = []
        self._lock = asyncio.Lock()

        # Compile regex patterns for performance
        self._sql_patterns = [
            re.compile(
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
                re.IGNORECASE,
            ),
            re.compile(r"(\b(OR|AND)\s+\d+\s*=\s*\d+)", re.IGNORECASE),
            re.compile(r"('|\"|;|--|\*|\/\*|\*\/)", re.IGNORECASE),
        ]

        self._xss_patterns = [
            re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
            re.compile(r"javascript:", re.IGNORECASE),
            re.compile(r"on\w+\s*=", re.IGNORECASE),
            re.compile(r"<iframe[^>]*>", re.IGNORECASE),
            re.compile(r"<object[^>]*>", re.IGNORECASE),
            re.compile(r"<embed[^>]*>", re.IGNORECASE),
        ]

        self._command_patterns = [
            re.compile(r"(\||&|;|`|\$\(|\${)", re.IGNORECASE),
            re.compile(
                r"\b(rm|del|format|shutdown|reboot|kill|ps|ls|cat|grep|find)\b",
                re.IGNORECASE,
            ),
            re.compile(r"(\.\.\/|\.\.\\|\/etc\/|\/bin\/|\/usr\/)", re.IGNORECASE),
        ]

        self._path_traversal_patterns = [
            re.compile(r"(\.\.\/|\.\.\\)", re.IGNORECASE),
            re.compile(r"(\/etc\/|\/bin\/|\/usr\/|\/var\/|\/tmp\/)", re.IGNORECASE),
            re.compile(r"(\\windows\\|\\system32\\|\\program files\\)", re.IGNORECASE),
        ]

    async def validate_input(
        self,
        input_data: Any,
        context: SecurityContext,
        max_length: Optional[int] = None,
    ) -> ValidationResult:
        """Validate and sanitize input data."""
        try:
            # Convert input to string for validation
            input_str = str(input_data) if input_data is not None else ""

            # Check input length
            max_len = max_length or self.policy.max_input_length
            if len(input_str) > max_len:
                await self._log_security_event(
                    "input_validation_failed",
                    SecurityLevel.MEDIUM,
                    context,
                    ThreatType.SUSPICIOUS_PATTERN,
                    {
                        "reason": "input_too_long",
                        "length": len(input_str),
                        "max_length": max_len,
                    },
                )
                return ValidationResult(
                    status=ValidationStatus.INVALID,
                    message=f"Input too long: {len(input_str)} > {max_len}",
                )

            threats_detected = []
            risk_score = 0.0
            sanitized_input = input_str

            # SQL Injection Detection
            if self.policy.enable_sql_injection_protection:
                sql_threats = await self._detect_sql_injection(input_str)
                if sql_threats:
                    threats_detected.append(ThreatType.SQL_INJECTION)
                    risk_score += 0.8
                    sanitized_input = await self._sanitize_sql(sanitized_input)

            # XSS Detection
            if self.policy.enable_xss_protection:
                xss_threats = await self._detect_xss(input_str)
                if xss_threats:
                    threats_detected.append(ThreatType.XSS)
                    risk_score += 0.7
                    sanitized_input = await self._sanitize_xss(sanitized_input)

            # Command Injection Detection
            if self.policy.enable_command_injection_protection:
                cmd_threats = await self._detect_command_injection(input_str)
                if cmd_threats:
                    threats_detected.append(ThreatType.COMMAND_INJECTION)
                    risk_score += 0.9
                    sanitized_input = await self._sanitize_commands(sanitized_input)

            # Path Traversal Detection
            if self.policy.enable_path_traversal_protection:
                path_threats = await self._detect_path_traversal(input_str)
                if path_threats:
                    threats_detected.append(ThreatType.PATH_TRAVERSAL)
                    risk_score += 0.6
                    sanitized_input = await self._sanitize_paths(sanitized_input)

            # Custom Pattern Detection
            custom_threats = await self._detect_custom_patterns(input_str)
            if custom_threats:
                threats_detected.append(ThreatType.SUSPICIOUS_PATTERN)
                risk_score += 0.5

            # Determine validation status
            if risk_score >= 0.8:
                status = (
                    ValidationStatus.BLOCKED
                    if self.policy.block_suspicious_requests
                    else ValidationStatus.SUSPICIOUS
                )
                await self._log_security_event(
                    "high_risk_input_detected",
                    SecurityLevel.HIGH,
                    context,
                    (
                        threats_detected[0]
                        if threats_detected
                        else ThreatType.SUSPICIOUS_PATTERN
                    ),
                    {
                        "risk_score": risk_score,
                        "threats": [t.value for t in threats_detected],
                    },
                    blocked=(status == ValidationStatus.BLOCKED),
                )
            elif risk_score >= 0.5:
                status = ValidationStatus.SUSPICIOUS
                await self._log_security_event(
                    "suspicious_input_detected",
                    SecurityLevel.MEDIUM,
                    context,
                    (
                        threats_detected[0]
                        if threats_detected
                        else ThreatType.SUSPICIOUS_PATTERN
                    ),
                    {
                        "risk_score": risk_score,
                        "threats": [t.value for t in threats_detected],
                    },
                )
            elif threats_detected:
                status = ValidationStatus.SUSPICIOUS
            else:
                status = ValidationStatus.VALID

            return ValidationResult(
                status=status,
                sanitized_input=sanitized_input,
                threats_detected=threats_detected,
                risk_score=risk_score,
                metadata={
                    "original_length": len(input_str),
                    "sanitized_length": len(sanitized_input),
                },
            )

        except Exception as e:
            logger.error(f"Error during input validation: {e}")
            await self._log_security_event(
                "validation_error",
                SecurityLevel.HIGH,
                context,
                ThreatType.SUSPICIOUS_PATTERN,
                {"error": str(e)},
            )
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message=f"Validation error: {e}",
            )

    async def check_rate_limit(
        self,
        client_id: str,
        operation: str,
        context: SecurityContext,
    ) -> bool:
        """Check if client is within rate limits."""
        async with self._lock:
            # Get rate limit config for operation
            rate_config = self.policy.rate_limits.get(operation)
            if not rate_config:
                return True  # No rate limit configured

            # Get current state
            state = self.rate_limit_states[client_id]
            now = datetime.utcnow()

            # Check if client is currently blocked
            if state.blocked_until and now < state.blocked_until:
                await self._log_security_event(
                    "rate_limit_blocked_request",
                    SecurityLevel.MEDIUM,
                    context,
                    ThreatType.RATE_LIMIT_EXCEEDED,
                    {
                        "client_id": client_id,
                        "operation": operation,
                        "blocked_until": state.blocked_until.isoformat(),
                    },
                    blocked=True,
                )
                return False

            # Reset window if expired
            if now - state.window_start >= rate_config.time_window:
                state.request_count = 0
                state.window_start = now
                state.burst_count = 0

            # Check rate limit
            if state.request_count >= rate_config.max_requests:
                # Block client
                state.blocked_until = now + rate_config.block_duration
                await self._log_security_event(
                    "rate_limit_exceeded",
                    SecurityLevel.HIGH,
                    context,
                    ThreatType.RATE_LIMIT_EXCEEDED,
                    {
                        "client_id": client_id,
                        "operation": operation,
                        "request_count": state.request_count,
                        "max_requests": rate_config.max_requests,
                        "blocked_until": state.blocked_until.isoformat(),
                    },
                    blocked=True,
                )
                return False

            # Check burst limit
            if rate_config.burst_limit and state.burst_count >= rate_config.burst_limit:
                await self._log_security_event(
                    "burst_limit_exceeded",
                    SecurityLevel.MEDIUM,
                    context,
                    ThreatType.RATE_LIMIT_EXCEEDED,
                    {
                        "client_id": client_id,
                        "operation": operation,
                        "burst_count": state.burst_count,
                        "burst_limit": rate_config.burst_limit,
                    },
                )
                return False

            # Update counters
            state.request_count += 1
            state.burst_count += 1

            return True

    async def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked."""
        return ip_address in self.blocked_ips

    async def block_ip(
        self,
        ip_address: str,
        context: SecurityContext,
        reason: str = "Security violation",
    ):
        """Block an IP address."""
        async with self._lock:
            self.blocked_ips.add(ip_address)
            await self._log_security_event(
                "ip_blocked",
                SecurityLevel.HIGH,
                context,
                ThreatType.SUSPICIOUS_PATTERN,
                {"ip_address": ip_address, "reason": reason},
                blocked=True,
            )

    async def unblock_ip(self, ip_address: str, context: SecurityContext):
        """Unblock an IP address."""
        async with self._lock:
            self.blocked_ips.discard(ip_address)
            await self._log_security_event(
                "ip_unblocked",
                SecurityLevel.LOW,
                context,
                None,
                {"ip_address": ip_address},
            )

    async def generate_secure_token(self, length: int = 32) -> str:
        """Generate a cryptographically secure token."""
        return secrets.token_urlsafe(length)

    async def hash_password(
        self, password: str, salt: Optional[str] = None
    ) -> tuple[str, str]:
        """Hash a password with salt."""
        if salt is None:
            salt = secrets.token_hex(16)

        # Use PBKDF2 with SHA-256
        password_hash = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), 100000
        )
        return password_hash.hex(), salt

    async def verify_password(
        self, password: str, password_hash: str, salt: str
    ) -> bool:
        """Verify a password against its hash."""
        computed_hash, _ = await self.hash_password(password, salt)
        return hmac.compare_digest(computed_hash, password_hash)

    async def get_security_events(
        self,
        limit: Optional[int] = None,
        severity: Optional[SecurityLevel] = None,
        since: Optional[datetime] = None,
    ) -> List[SecurityEvent]:
        """Get security events with optional filtering."""
        events = self.security_events.copy()

        if severity:
            events = [e for e in events if e.severity == severity]

        if since:
            events = [e for e in events if e.timestamp >= since]

        # Sort by timestamp (newest first)
        events.sort(key=lambda e: e.timestamp, reverse=True)

        if limit:
            events = events[:limit]

        return events

    async def _detect_sql_injection(self, input_str: str) -> List[str]:
        """Detect SQL injection patterns."""
        threats = []
        for pattern in self._sql_patterns:
            if pattern.search(input_str):
                threats.append(pattern.pattern)
        return threats

    async def _detect_xss(self, input_str: str) -> List[str]:
        """Detect XSS patterns."""
        threats = []
        decoded_input = unquote(input_str)  # URL decode to catch encoded attacks
        for pattern in self._xss_patterns:
            if pattern.search(decoded_input):
                threats.append(pattern.pattern)
        return threats

    async def _detect_command_injection(self, input_str: str) -> List[str]:
        """Detect command injection patterns."""
        threats = []
        for pattern in self._command_patterns:
            if pattern.search(input_str):
                threats.append(pattern.pattern)
        return threats

    async def _detect_path_traversal(self, input_str: str) -> List[str]:
        """Detect path traversal patterns."""
        threats = []
        for pattern in self._path_traversal_patterns:
            if pattern.search(input_str):
                threats.append(pattern.pattern)
        return threats

    async def _detect_custom_patterns(self, input_str: str) -> List[str]:
        """Detect custom blocked patterns."""
        threats = []
        for pattern_str in self.policy.blocked_patterns:
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                if pattern.search(input_str):
                    threats.append(pattern_str)
            except re.error:
                logger.warning(
                    f"Invalid regex pattern in security policy: {pattern_str}"
                )
        return threats

    async def _sanitize_sql(self, input_str: str) -> str:
        """Sanitize SQL injection attempts."""
        # Remove common SQL keywords and characters
        sanitized = re.sub(
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            "",
            input_str,
            flags=re.IGNORECASE,
        )
        sanitized = re.sub(r"('|\"|;|--|\*|\/\*|\*\/)", "", sanitized)
        return sanitized.strip()

    async def _sanitize_xss(self, input_str: str) -> str:
        """Sanitize XSS attempts."""
        # Remove script tags and event handlers
        sanitized = re.sub(
            r"<script[^>]*>.*?</script>", "", input_str, flags=re.IGNORECASE | re.DOTALL
        )
        sanitized = re.sub(r"javascript:", "", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(
            r"on\w+\s*=\s*[\"'][^\"']*[\"']", "", sanitized, flags=re.IGNORECASE
        )
        sanitized = re.sub(
            r"<(iframe|object|embed)[^>]*>", "", sanitized, flags=re.IGNORECASE
        )
        return sanitized.strip()

    async def _sanitize_commands(self, input_str: str) -> str:
        """Sanitize command injection attempts."""
        # Remove command separators and dangerous commands
        sanitized = re.sub(r"(\||&|;|`|\$\(|\${)", "", input_str)
        sanitized = re.sub(
            r"\b(rm|del|format|shutdown|reboot|kill|ps|ls|cat|grep|find)\b",
            "",
            sanitized,
            flags=re.IGNORECASE,
        )
        return sanitized.strip()

    async def _sanitize_paths(self, input_str: str) -> str:
        """Sanitize path traversal attempts."""
        # Remove path traversal sequences
        sanitized = re.sub(r"(\.\.\/|\.\.\\)", "", input_str)
        sanitized = re.sub(r"(\/etc\/|\/bin\/|\/usr\/|\/var\/|\/tmp\/)", "", sanitized)
        sanitized = re.sub(
            r"(\\windows\\|\\system32\\|\\program files\\)",
            "",
            sanitized,
            flags=re.IGNORECASE,
        )
        return sanitized.strip()

    async def _log_security_event(
        self,
        event_type: str,
        severity: SecurityLevel,
        context: SecurityContext,
        threat_type: Optional[ThreatType] = None,
        details: Optional[Dict[str, Any]] = None,
        blocked: bool = False,
    ):
        """Log a security event."""
        event = SecurityEvent(
            event_id=await self.generate_secure_token(16),
            timestamp=datetime.utcnow(),
            event_type=event_type,
            severity=severity,
            source_ip=context.ip_address,
            user_id=context.user_id,
            session_id=context.session_id,
            operation=context.operation,
            threat_type=threat_type,
            details=details or {},
            blocked=blocked,
        )

        self.security_events.append(event)

        # Keep only recent events to prevent memory issues
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-5000:]

        # Log to application logger
        log_level = (
            "error"
            if severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]
            else "warning"
        )
        getattr(logger, log_level)(
            f"Security event: {event_type} | Severity: {severity.value} | "
            f"IP: {context.ip_address} | User: {context.user_id} | "
            f"Threat: {threat_type.value if threat_type else 'None'} | "
            f"Blocked: {blocked}"
        )
