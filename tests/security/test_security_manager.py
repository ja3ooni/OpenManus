"""Tests for SecurityManager."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from app.security.manager import SecurityManager
from app.security.models import (
    RateLimitConfig,
    SecurityContext,
    SecurityLevel,
    SecurityPolicy,
    ThreatType,
    ValidationStatus,
)


class TestSecurityManager:
    """Test cases for SecurityManager."""

    @pytest.fixture
    def security_policy(self):
        """Create test security policy."""
        return SecurityPolicy(
            max_input_length=1000,
            rate_limits={
                "test_operation": RateLimitConfig(
                    max_requests=5,
                    time_window=timedelta(minutes=1),
                    burst_limit=2,
                )
            },
            enable_xss_protection=True,
            enable_sql_injection_protection=True,
            enable_command_injection_protection=True,
            enable_path_traversal_protection=True,
        )

    @pytest.fixture
    def security_manager(self, security_policy):
        """Create SecurityManager instance."""
        return SecurityManager(security_policy)

    @pytest.fixture
    def security_context(self):
        """Create test security context."""
        return SecurityContext(
            user_id="test_user",
            session_id="test_session",
            ip_address="192.168.1.1",
            operation="test_operation",
        )

    @pytest.mark.asyncio
    async def test_validate_input_clean(self, security_manager, security_context):
        """Test validation of clean input."""
        result = await security_manager.validate_input(
            "This is a clean input string", security_context
        )

        assert result.status == ValidationStatus.VALID
        assert result.sanitized_input == "This is a clean input string"
        assert not result.threats_detected
        assert result.risk_score == 0.0

    @pytest.mark.asyncio
    async def test_validate_input_sql_injection(
        self, security_manager, security_context
    ):
        """Test detection of SQL injection."""
        malicious_input = "'; DROP TABLE users; --"

        result = await security_manager.validate_input(
            malicious_input, security_context
        )

        assert result.status in [ValidationStatus.SUSPICIOUS, ValidationStatus.BLOCKED]
        assert ThreatType.SQL_INJECTION in result.threats_detected
        assert result.risk_score > 0.5
        assert result.sanitized_input != malicious_input

    @pytest.mark.asyncio
    async def test_validate_input_xss(self, security_manager, security_context):
        """Test detection of XSS."""
        malicious_input = "<script>alert('xss')</script>"

        result = await security_manager.validate_input(
            malicious_input, security_context
        )

        assert result.status in [ValidationStatus.SUSPICIOUS, ValidationStatus.BLOCKED]
        assert ThreatType.XSS in result.threats_detected
        assert result.risk_score > 0.5
        assert "<script>" not in result.sanitized_input

    @pytest.mark.asyncio
    async def test_validate_input_command_injection(
        self, security_manager, security_context
    ):
        """Test detection of command injection."""
        malicious_input = "test; rm -rf /"

        result = await security_manager.validate_input(
            malicious_input, security_context
        )

        assert result.status in [ValidationStatus.SUSPICIOUS, ValidationStatus.BLOCKED]
        assert ThreatType.COMMAND_INJECTION in result.threats_detected
        assert result.risk_score > 0.5

    @pytest.mark.asyncio
    async def test_validate_input_path_traversal(
        self, security_manager, security_context
    ):
        """Test detection of path traversal."""
        malicious_input = "../../etc/passwd"

        result = await security_manager.validate_input(
            malicious_input, security_context
        )

        assert result.status in [ValidationStatus.SUSPICIOUS, ValidationStatus.BLOCKED]
        assert ThreatType.PATH_TRAVERSAL in result.threats_detected
        assert result.risk_score > 0.5

    @pytest.mark.asyncio
    async def test_validate_input_too_long(self, security_manager, security_context):
        """Test input length validation."""
        long_input = "a" * 2000  # Exceeds max_input_length of 1000

        result = await security_manager.validate_input(long_input, security_context)

        assert result.status == ValidationStatus.INVALID
        assert "too long" in result.message.lower()

    @pytest.mark.asyncio
    async def test_rate_limiting_normal_usage(self, security_manager, security_context):
        """Test normal rate limiting behavior."""
        client_id = "test_client"

        # Should allow requests within limit
        for i in range(5):
            allowed = await security_manager.check_rate_limit(
                client_id, "test_operation", security_context
            )
            assert allowed

        # Should block after limit exceeded
        blocked = await security_manager.check_rate_limit(
            client_id, "test_operation", security_context
        )
        assert not blocked

    @pytest.mark.asyncio
    async def test_rate_limiting_window_reset(self, security_manager, security_context):
        """Test rate limit window reset."""
        client_id = "test_client"

        # Exhaust rate limit
        for i in range(6):
            await security_manager.check_rate_limit(
                client_id, "test_operation", security_context
            )

        # Manually reset window by modifying state
        state = security_manager.rate_limit_states[client_id]
        state.window_start = datetime.utcnow() - timedelta(minutes=2)

        # Should allow requests again
        allowed = await security_manager.check_rate_limit(
            client_id, "test_operation", security_context
        )
        assert allowed

    @pytest.mark.asyncio
    async def test_ip_blocking(self, security_manager, security_context):
        """Test IP blocking functionality."""
        ip_address = "192.168.1.100"

        # IP should not be blocked initially
        assert not await security_manager.is_ip_blocked(ip_address)

        # Block IP
        await security_manager.block_ip(ip_address, security_context, "Test block")

        # IP should now be blocked
        assert await security_manager.is_ip_blocked(ip_address)

        # Unblock IP
        await security_manager.unblock_ip(ip_address, security_context)

        # IP should not be blocked anymore
        assert not await security_manager.is_ip_blocked(ip_address)

    @pytest.mark.asyncio
    async def test_secure_token_generation(self, security_manager):
        """Test secure token generation."""
        token1 = await security_manager.generate_secure_token()
        token2 = await security_manager.generate_secure_token()

        # Tokens should be different
        assert token1 != token2

        # Tokens should have reasonable length
        assert len(token1) > 20
        assert len(token2) > 20

    @pytest.mark.asyncio
    async def test_password_hashing(self, security_manager):
        """Test password hashing and verification."""
        password = "test_password_123"

        # Hash password
        password_hash, salt = await security_manager.hash_password(password)

        # Verify correct password
        assert await security_manager.verify_password(password, password_hash, salt)

        # Verify incorrect password
        assert not await security_manager.verify_password(
            "wrong_password", password_hash, salt
        )

    @pytest.mark.asyncio
    async def test_security_events_logging(self, security_manager, security_context):
        """Test security event logging."""
        # Trigger a security event
        malicious_input = "'; DROP TABLE users; --"
        await security_manager.validate_input(malicious_input, security_context)

        # Check that event was logged
        events = await security_manager.get_security_events(limit=10)
        assert len(events) > 0

        # Check event details
        event = events[0]
        assert event.event_type in [
            "high_risk_input_detected",
            "suspicious_input_detected",
        ]
        assert event.source_ip == security_context.ip_address
        assert event.user_id == security_context.user_id

    @pytest.mark.asyncio
    async def test_security_events_filtering(self, security_manager, security_context):
        """Test security event filtering."""
        # Generate multiple events with different severities
        await security_manager._log_security_event(
            "test_event_1", SecurityLevel.LOW, security_context
        )
        await security_manager._log_security_event(
            "test_event_2", SecurityLevel.HIGH, security_context
        )

        # Filter by severity
        high_events = await security_manager.get_security_events(
            severity=SecurityLevel.HIGH
        )
        low_events = await security_manager.get_security_events(
            severity=SecurityLevel.LOW
        )

        assert len(high_events) >= 1
        assert len(low_events) >= 1
        assert all(e.severity == SecurityLevel.HIGH for e in high_events)
        assert all(e.severity == SecurityLevel.LOW for e in low_events)

    @pytest.mark.asyncio
    async def test_multiple_threats_detection(self, security_manager, security_context):
        """Test detection of multiple threats in single input."""
        # Input with both SQL injection and XSS
        malicious_input = "'; DROP TABLE users; --<script>alert('xss')</script>"

        result = await security_manager.validate_input(
            malicious_input, security_context
        )

        assert result.status in [ValidationStatus.SUSPICIOUS, ValidationStatus.BLOCKED]
        assert ThreatType.SQL_INJECTION in result.threats_detected
        assert ThreatType.XSS in result.threats_detected
        assert result.risk_score > 1.0  # Should be high due to multiple threats

    @pytest.mark.asyncio
    async def test_concurrent_rate_limiting(self, security_manager, security_context):
        """Test rate limiting under concurrent access."""
        client_id = "concurrent_client"

        async def make_request():
            return await security_manager.check_rate_limit(
                client_id, "test_operation", security_context
            )

        # Make concurrent requests
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # Should have some allowed and some blocked
        allowed_count = sum(results)
        assert allowed_count <= 5  # Rate limit is 5
        assert allowed_count > 0  # At least some should be allowed

    @pytest.mark.asyncio
    async def test_error_handling_in_validation(
        self, security_manager, security_context
    ):
        """Test error handling during validation."""
        # Test with None input
        result = await security_manager.validate_input(None, security_context)
        assert result.status == ValidationStatus.VALID
        assert result.sanitized_input == ""

    @pytest.mark.asyncio
    async def test_custom_pattern_detection(self, security_manager, security_context):
        """Test custom pattern detection."""
        # Add custom blocked pattern
        security_manager.policy.blocked_patterns = [r"\bcustom_threat\b"]

        # Test input with custom pattern
        result = await security_manager.validate_input(
            "This contains custom_threat pattern", security_context
        )

        assert result.status in [ValidationStatus.SUSPICIOUS, ValidationStatus.BLOCKED]
        assert ThreatType.SUSPICIOUS_PATTERN in result.threats_detected
