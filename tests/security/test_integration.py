"""
Tests for security monitoring integration.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.security.integration import (
    SecurityEventProcessor,
    SecurityMonitoringContext,
    SecurityMonitoringMiddleware,
    monitor_security,
)
from app.security.models import SecurityContext, SecurityLevel


class TestSecurityMonitoringMiddleware:
    """Test security monitoring middleware functionality."""

    @pytest.fixture
    def middleware(self):
        """Create security monitoring middleware for testing."""
        return SecurityMonitoringMiddleware()

    @pytest.fixture
    def security_context(self):
        """Create security context for testing."""
        return SecurityContext(
            user_id="test_user",
            session_id="test_session",
            ip_address="192.168.1.100",
            operation="test_operation",
        )

    @pytest.mark.asyncio
    async def test_monitor_operation_success(self, middleware, security_context):
        """Test monitoring successful operation."""

        async def test_operation():
            return "success_result"

        with patch("app.security.integration.log_audit_entry") as mock_audit, patch(
            "app.security.integration.analyze_access_attempt"
        ) as mock_analyze:

            mock_analyze.return_value = []  # No anomalies

            result = await middleware.monitor_operation(
                "credential_access", security_context, test_operation
            )

            assert result == "success_result"

            # Verify audit logging was called
            assert mock_audit.call_count == 2  # start and complete

            # Verify anomaly analysis was called
            mock_analyze.assert_called_once_with(
                security_context, True, "credential_access"
            )

    @pytest.mark.asyncio
    async def test_monitor_operation_failure(self, middleware, security_context):
        """Test monitoring failed operation."""

        async def failing_operation():
            raise ValueError("Test error")

        with patch("app.security.integration.log_audit_entry") as mock_audit, patch(
            "app.security.integration.log_security_event"
        ) as mock_security:

            with pytest.raises(ValueError, match="Test error"):
                await middleware.monitor_operation(
                    "credential_access", security_context, failing_operation
                )

            # Verify security event was logged for failure
            mock_security.assert_called_once()
            call_args = mock_security.call_args
            assert call_args[0][0] == "operation_failed"
            assert call_args[0][1] == SecurityLevel.MEDIUM

    @pytest.mark.asyncio
    async def test_monitor_operation_with_anomalies(self, middleware, security_context):
        """Test monitoring operation that triggers anomalies."""

        async def test_operation():
            return "result"

        from app.security.monitoring import AnomalyType

        with patch("app.security.integration.log_audit_entry"), patch(
            "app.security.integration.analyze_access_attempt"
        ) as mock_analyze, patch(
            "app.security.integration.log_security_event"
        ) as mock_security:

            # Mock anomalies detected
            mock_analyze.return_value = [AnomalyType.UNUSUAL_ACCESS_PATTERN]

            await middleware.monitor_operation(
                "credential_access", security_context, test_operation
            )

            # Verify security event was logged for anomalies
            mock_security.assert_called_once()
            call_args = mock_security.call_args
            assert call_args[0][0] == "access_anomalies_detected"

    @pytest.mark.asyncio
    async def test_monitor_authentication_success(self, middleware, security_context):
        """Test monitoring successful authentication."""

        with patch(
            "app.security.integration.log_security_event"
        ) as mock_security, patch(
            "app.security.integration.log_audit_entry"
        ) as mock_audit, patch(
            "app.security.integration.analyze_access_attempt"
        ) as mock_analyze:

            mock_analyze.return_value = []

            await middleware.monitor_authentication(
                security_context, "password", True, {"method": "form"}
            )

            # Verify security event logged
            mock_security.assert_called_once()
            call_args = mock_security.call_args
            assert call_args[0][0] == "authentication_success"
            assert call_args[0][1] == SecurityLevel.LOW

            # Verify audit entry logged
            mock_audit.assert_called_once()

    @pytest.mark.asyncio
    async def test_monitor_authentication_failure(self, middleware, security_context):
        """Test monitoring failed authentication."""

        with patch(
            "app.security.integration.log_security_event"
        ) as mock_security, patch(
            "app.security.integration.log_audit_entry"
        ) as mock_audit, patch(
            "app.security.integration.analyze_access_attempt"
        ) as mock_analyze:

            mock_analyze.return_value = []

            await middleware.monitor_authentication(
                security_context, "password", False, {"reason": "invalid_password"}
            )

            # Verify security event logged with higher severity
            mock_security.assert_called()
            call_args = mock_security.call_args_list[0]
            assert call_args[0][0] == "authentication_failure"
            assert call_args[0][1] == SecurityLevel.MEDIUM

    @pytest.mark.asyncio
    async def test_monitor_file_operation_high_risk(self, middleware, security_context):
        """Test monitoring high-risk file operations."""

        with patch("app.security.integration.log_audit_entry") as mock_audit, patch(
            "app.security.integration.log_security_event"
        ) as mock_security:

            await middleware.monitor_file_operation(
                security_context, "delete", "/etc/passwd", True
            )

            # Verify high risk level was assigned
            mock_audit.assert_called_once()
            call_args = mock_audit.call_args
            assert call_args[1]["risk_level"] == SecurityLevel.HIGH

            # Verify security event was logged
            mock_security.assert_called_once()

    @pytest.mark.asyncio
    async def test_monitor_network_request_external(self, middleware, security_context):
        """Test monitoring external network requests."""

        with patch("app.security.integration.log_audit_entry") as mock_audit:

            await middleware.monitor_network_request(
                security_context, "https://external-api.com/data", "GET", True, 200
            )

            # Verify audit entry was logged for external request
            mock_audit.assert_called_once()
            call_args = mock_audit.call_args
            assert call_args[0][0] == "network_request"
            assert call_args[1]["risk_level"] == SecurityLevel.MEDIUM

    @pytest.mark.asyncio
    async def test_monitor_command_execution_high_risk(
        self, middleware, security_context
    ):
        """Test monitoring high-risk command execution."""

        with patch("app.security.integration.log_audit_entry") as mock_audit, patch(
            "app.security.integration.log_security_event"
        ) as mock_security:

            await middleware.monitor_command_execution(
                security_context, "rm -rf /important/data", True, 0
            )

            # Verify high risk assessment
            mock_audit.assert_called_once()
            call_args = mock_audit.call_args
            assert call_args[1]["risk_level"] == SecurityLevel.HIGH

            # Verify security event was logged
            mock_security.assert_called_once()

    def test_assess_command_risk(self, middleware):
        """Test command risk assessment."""

        # High-risk commands
        assert middleware._assess_command_risk("rm -rf /") == SecurityLevel.HIGH
        assert middleware._assess_command_risk("sudo passwd root") == SecurityLevel.HIGH
        assert (
            middleware._assess_command_risk("chmod 777 /etc/passwd")
            == SecurityLevel.HIGH
        )

        # Medium-risk commands
        assert (
            middleware._assess_command_risk("wget http://example.com")
            == SecurityLevel.MEDIUM
        )
        assert middleware._assess_command_risk("git clone repo") == SecurityLevel.MEDIUM
        assert (
            middleware._assess_command_risk("docker run image") == SecurityLevel.MEDIUM
        )

        # Low-risk commands
        assert middleware._assess_command_risk("ls -la") == SecurityLevel.LOW
        assert middleware._assess_command_risk("echo hello") == SecurityLevel.LOW
        assert middleware._assess_command_risk("cat file.txt") == SecurityLevel.LOW

    def test_is_internal_url(self, middleware):
        """Test internal URL detection."""

        assert middleware._is_internal_url("http://localhost:8080") is True
        assert middleware._is_internal_url("https://127.0.0.1:3000") is True
        assert middleware._is_internal_url("http://openmanus.local") is True
        assert middleware._is_internal_url("https://external-api.com") is False
        assert middleware._is_internal_url("http://google.com") is False


class TestSecurityEventProcessor:
    """Test security event processor functionality."""

    @pytest.fixture
    def processor(self):
        """Create security event processor for testing."""
        return SecurityEventProcessor()

    @pytest.mark.asyncio
    async def test_process_rapid_failed_attempts(self, processor):
        """Test processing rapid failed attempts event."""

        event_data = {
            "event_type": "rapid_failed_attempts_detected",
            "severity": "high",
            "source_ip": "192.168.1.100",
            "user_id": "test_user",
        }

        with patch("app.security.integration.logger") as mock_logger:
            await processor.process_event(event_data)

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert "Rapid failed attempts detected" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_process_high_risk_command(self, processor):
        """Test processing high-risk command event."""

        event_data = {
            "event_type": "command_execution_monitored",
            "severity": "high",
            "user_id": "test_user",
            "details": {"command": "rm -rf /important"},
        }

        with patch("app.security.integration.logger") as mock_logger:
            await processor.process_event(event_data)

            # Verify error was logged
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            assert "High-risk command executed" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_process_threat_indicator_match(self, processor):
        """Test processing threat indicator match event."""

        event_data = {
            "event_type": "threat_indicator_match_detected",
            "severity": "critical",
            "source_ip": "192.168.1.100",
            "user_id": "test_user",
        }

        with patch("app.security.integration.logger") as mock_logger:
            await processor.process_event(event_data)

            # Verify critical log was made
            mock_logger.critical.assert_called_once()
            call_args = mock_logger.critical.call_args
            assert "Known threat indicator detected" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_auto_response_disabled(self, processor):
        """Test that auto-response is disabled by default."""

        assert processor.auto_response_enabled is False

        event_data = {
            "event_type": "rapid_failed_attempts_detected",
            "severity": "high",
            "source_ip": "192.168.1.100",
        }

        with patch("app.security.integration.logger") as mock_logger:
            await processor.process_event(event_data)

            # Should log but not take automatic action
            mock_logger.warning.assert_called_once()
            # No auto-blocking should occur
            assert not any(
                "Auto-blocking" in str(call)
                for call in mock_logger.warning.call_args_list
            )


class TestSecurityDecorator:
    """Test security monitoring decorator."""

    @pytest.mark.asyncio
    async def test_monitor_security_decorator(self):
        """Test security monitoring decorator."""

        @monitor_security("test_operation")
        async def test_function(value):
            return f"processed_{value}"

        with patch("app.security.integration.security_middleware") as mock_middleware:
            mock_middleware.monitor_operation = AsyncMock(return_value="mocked_result")

            result = await test_function("test_value")

            assert result == "mocked_result"
            mock_middleware.monitor_operation.assert_called_once()


class TestSecurityMonitoringContext:
    """Test security monitoring context manager."""

    @pytest.fixture
    def security_context(self):
        """Create security context for testing."""
        return SecurityContext(
            user_id="test_user",
            operation="test_operation",
        )

    @pytest.mark.asyncio
    async def test_context_manager_success(self, security_context):
        """Test security monitoring context manager with successful operation."""

        with patch("app.security.integration.log_audit_entry") as mock_audit, patch(
            "app.security.integration.analyze_access_attempt"
        ) as mock_analyze:

            mock_analyze.return_value = []

            async with SecurityMonitoringContext("test_op", security_context):
                # Simulate some work
                await asyncio.sleep(0.01)

            # Verify audit entries were logged
            assert mock_audit.call_count == 2  # start and complete

            # Verify start audit entry
            start_call = mock_audit.call_args_list[0]
            assert start_call[1]["operation"] == "test_op"
            assert start_call[1]["action"] == "start"

            # Verify complete audit entry
            complete_call = mock_audit.call_args_list[1]
            assert complete_call[1]["action"] == "complete"
            assert complete_call[1]["result"] == "success"

    @pytest.mark.asyncio
    async def test_context_manager_failure(self, security_context):
        """Test security monitoring context manager with failed operation."""

        with patch("app.security.integration.log_audit_entry") as mock_audit, patch(
            "app.security.integration.analyze_access_attempt"
        ) as mock_analyze:

            mock_analyze.return_value = []

            try:
                async with SecurityMonitoringContext("test_op", security_context):
                    raise ValueError("Test error")
            except ValueError:
                pass

            # Verify complete audit entry shows failure
            complete_call = mock_audit.call_args_list[1]
            assert complete_call[1]["result"] == "failure"
            assert "Test error" in complete_call[1]["details"]["error"]

    @pytest.mark.asyncio
    async def test_context_manager_with_anomalies(self, security_context):
        """Test security monitoring context manager with detected anomalies."""

        from app.security.monitoring import AnomalyType

        with patch("app.security.integration.log_audit_entry"), patch(
            "app.security.integration.analyze_access_attempt"
        ) as mock_analyze, patch(
            "app.security.integration.log_security_event"
        ) as mock_security:

            # Mock anomalies detected
            mock_analyze.return_value = [AnomalyType.BEHAVIORAL_ANOMALY]

            async with SecurityMonitoringContext("test_op", security_context):
                pass

            # Verify security event was logged for anomalies
            mock_security.assert_called_once()
            call_args = mock_security.call_args
            assert call_args[0][0] == "operation_anomalies_detected"

    @pytest.mark.asyncio
    async def test_context_manager_auto_log_disabled(self, security_context):
        """Test security monitoring context manager with auto-logging disabled."""

        with patch("app.security.integration.log_audit_entry") as mock_audit, patch(
            "app.security.integration.analyze_access_attempt"
        ) as mock_analyze:

            mock_analyze.return_value = []

            async with SecurityMonitoringContext(
                "test_op", security_context, auto_log=False
            ):
                pass

            # Verify no audit entries were logged
            mock_audit.assert_not_called()


class TestSecurityIntegrationWorkflow:
    """Test complete security integration workflows."""

    @pytest.mark.asyncio
    async def test_complete_monitoring_workflow(self):
        """Test complete security monitoring workflow."""

        from app.security.integration import security_middleware

        security_context = SecurityContext(
            user_id="test_user",
            ip_address="192.168.1.100",
            operation="sensitive_operation",
        )

        async def sensitive_operation():
            return "operation_result"

        with patch("app.security.integration.log_audit_entry") as mock_audit, patch(
            "app.security.integration.log_security_event"
        ) as mock_security, patch(
            "app.security.integration.analyze_access_attempt"
        ) as mock_analyze:

            mock_analyze.return_value = []

            result = await security_middleware.monitor_operation(
                "credential_access", security_context, sensitive_operation
            )

            assert result == "operation_result"

            # Verify all monitoring components were called
            assert mock_audit.call_count >= 1
            mock_analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_authentication_monitoring_workflow(self):
        """Test authentication monitoring workflow."""

        from app.security.integration import security_middleware

        security_context = SecurityContext(
            user_id="test_user",
            ip_address="192.168.1.100",
        )

        with patch(
            "app.security.integration.log_security_event"
        ) as mock_security, patch(
            "app.security.integration.log_audit_entry"
        ) as mock_audit, patch(
            "app.security.integration.analyze_access_attempt"
        ) as mock_analyze:

            mock_analyze.return_value = []

            # Test successful authentication
            await security_middleware.monitor_authentication(
                security_context, "password", True
            )

            # Test failed authentication
            await security_middleware.monitor_authentication(
                security_context, "password", False
            )

            # Verify both events were logged
            assert mock_security.call_count == 2
            assert mock_audit.call_count == 2
            assert mock_analyze.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__])
