"""
Tests for security monitoring and audit logging system.
"""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.security.models import SecurityContext, SecurityLevel, ThreatType
from app.security.monitoring import (
    AlertSeverity,
    AnomalyType,
    IntrusionDetectionSystem,
    SecurityAlertSystem,
    SecurityEventLogger,
    ThreatIndicator,
)


class TestSecurityEventLogger:
    """Test security event logging functionality."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def event_logger(self, temp_storage):
        """Create security event logger for testing."""
        return SecurityEventLogger(storage_path=temp_storage)

    @pytest.fixture
    def security_context(self):
        """Create security context for testing."""
        return SecurityContext(
            user_id="test_user",
            session_id="test_session",
            ip_address="192.168.1.100",
            user_agent="TestAgent/1.0",
            operation="test_operation",
        )

    @pytest.mark.asyncio
    async def test_log_security_event(self, event_logger, security_context):
        """Test logging security events."""

        event = await event_logger.log_security_event(
            event_type="test_event",
            severity=SecurityLevel.MEDIUM,
            context=security_context,
            threat_type=ThreatType.SUSPICIOUS_PATTERN,
            details={"test": "data"},
            blocked=False,
        )

        assert event.event_type == "test_event"
        assert event.severity == SecurityLevel.MEDIUM
        assert event.source_ip == "192.168.1.100"
        assert event.user_id == "test_user"
        assert event.threat_type == ThreatType.SUSPICIOUS_PATTERN
        assert event.details == {"test": "data"}
        assert not event.blocked

        # Check event is stored in memory
        assert len(event_logger.recent_events) == 1
        assert event_logger.recent_events[0] == event

        # Check event statistics
        stats = event_logger.get_event_statistics()
        assert stats["total_events"] == 1
        assert stats["event_types"]["test_event"] == 1

    @pytest.mark.asyncio
    async def test_log_audit_entry(self, event_logger, security_context):
        """Test logging audit entries."""

        entry = await event_logger.log_audit_entry(
            operation="test_operation",
            action="test_action",
            context=security_context,
            resource="test_resource",
            result="success",
            details={"audit": "data"},
            risk_level=SecurityLevel.HIGH,
        )

        assert entry.operation == "test_operation"
        assert entry.action == "test_action"
        assert entry.user_id == "test_user"
        assert entry.source_ip == "192.168.1.100"
        assert entry.resource == "test_resource"
        assert entry.result == "success"
        assert entry.details == {"audit": "data"}
        assert entry.risk_level == SecurityLevel.HIGH

        # Check entry is stored in memory
        assert len(event_logger.recent_audit_entries) == 1
        assert event_logger.recent_audit_entries[0] == entry

    def test_add_threat_indicator(self, event_logger):
        """Test adding threat indicators."""

        indicator = event_logger.add_threat_indicator(
            indicator_type="ip",
            value="192.168.1.200",
            threat_level=SecurityLevel.HIGH,
            description="Known malicious IP",
            source="threat_intel",
        )

        assert indicator.indicator_type == "ip"
        assert indicator.value == "192.168.1.200"
        assert indicator.threat_level == SecurityLevel.HIGH
        assert indicator.description == "Known malicious IP"
        assert indicator.source == "threat_intel"

        # Check indicator is stored
        assert len(event_logger.threat_indicators) == 1
        assert indicator.indicator_id in event_logger.threat_indicators

    def test_remove_threat_indicator(self, event_logger):
        """Test removing threat indicators."""

        indicator = event_logger.add_threat_indicator(
            indicator_type="ip",
            value="192.168.1.200",
            threat_level=SecurityLevel.HIGH,
            description="Test indicator",
        )

        # Remove indicator
        result = event_logger.remove_threat_indicator(indicator.indicator_id)
        assert result is True
        assert len(event_logger.threat_indicators) == 0

        # Try to remove non-existent indicator
        result = event_logger.remove_threat_indicator("non_existent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_events_filtering(self, event_logger, security_context):
        """Test filtering security events."""

        # Create multiple events
        await event_logger.log_security_event(
            "event1", SecurityLevel.LOW, security_context
        )
        await event_logger.log_security_event(
            "event2", SecurityLevel.HIGH, security_context
        )
        await event_logger.log_security_event(
            "event1", SecurityLevel.MEDIUM, security_context
        )

        # Test filtering by event type
        events = await event_logger.get_events(event_type="event1")
        assert len(events) == 2
        assert all(e.event_type == "event1" for e in events)

        # Test filtering by severity
        events = await event_logger.get_events(severity=SecurityLevel.HIGH)
        assert len(events) == 1
        assert events[0].severity == SecurityLevel.HIGH

        # Test filtering by user
        events = await event_logger.get_events(user_id="test_user")
        assert len(events) == 3

        # Test limit
        events = await event_logger.get_events(limit=2)
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_threat_indicator_matching(self, event_logger, security_context):
        """Test threat indicator matching."""

        # Add threat indicator
        event_logger.add_threat_indicator(
            indicator_type="ip",
            value="192.168.1.100",  # Same as security_context IP
            threat_level=SecurityLevel.HIGH,
            description="Test threat IP",
        )

        with patch.object(event_logger, "_check_threat_indicators") as mock_check:
            await event_logger.log_security_event(
                "test_event", SecurityLevel.LOW, security_context
            )

            # Verify threat indicator check was called
            mock_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_event_persistence(
        self, event_logger, security_context, temp_storage
    ):
        """Test event persistence to storage."""

        await event_logger.log_security_event(
            "test_event", SecurityLevel.MEDIUM, security_context
        )

        # Check that events file was created and contains data
        events_file = temp_storage / "security_events.jsonl"
        assert events_file.exists()

        with open(events_file, "r") as f:
            line = f.readline().strip()
            event_data = json.loads(line)

            assert event_data["event_type"] == "test_event"
            assert event_data["severity"] == "medium"
            assert event_data["source_ip"] == "192.168.1.100"


class TestIntrusionDetectionSystem:
    """Test intrusion detection system functionality."""

    @pytest.fixture
    def event_logger(self):
        """Create mock event logger."""
        return MagicMock(spec=SecurityEventLogger)

    @pytest.fixture
    def ids(self, event_logger):
        """Create intrusion detection system for testing."""
        return IntrusionDetectionSystem(event_logger)

    @pytest.fixture
    def security_context(self):
        """Create security context for testing."""
        return SecurityContext(
            user_id="test_user",
            session_id="test_session",
            ip_address="192.168.1.100",
            user_agent="TestAgent/1.0",
            operation="login",
        )

    @pytest.mark.asyncio
    async def test_analyze_access_attempt_new_user(self, ids, security_context):
        """Test analyzing access attempt for new user."""

        anomalies = await ids.analyze_access_attempt(
            security_context, success=True, operation="login"
        )

        # New user should not trigger anomalies
        assert len(anomalies) == 0

        # Check that access pattern was created
        assert "test_user" in ids.access_patterns
        pattern = ids.access_patterns["test_user"]
        assert pattern.user_id == "test_user"
        assert pattern.success_count == 1
        assert pattern.failure_count == 0
        assert "192.168.1.100" in pattern.ip_addresses

    @pytest.mark.asyncio
    async def test_detect_failed_login_anomaly(self, ids, security_context):
        """Test detection of rapid failed login attempts."""

        # Simulate multiple failed attempts
        for _ in range(6):  # Above threshold of 5
            await ids.analyze_access_attempt(
                security_context, success=False, operation="login"
            )

        # Should detect rapid failed attempts anomaly
        pattern = ids.access_patterns["test_user"]
        assert pattern.failure_count == 6

        # Verify security event was logged
        ids.event_logger.log_security_event.assert_called()
        call_args = ids.event_logger.log_security_event.call_args
        assert call_args[0][0] == "rapid_failed_attempts_detected"
        assert call_args[0][1] == SecurityLevel.HIGH

    @pytest.mark.asyncio
    async def test_detect_rapid_requests_anomaly(self, ids, security_context):
        """Test detection of rapid request volume."""

        # Simulate rapid requests (above threshold)
        for _ in range(101):  # Above threshold of 100
            await ids.analyze_access_attempt(
                security_context, success=True, operation="api_call"
            )

        # Should detect volume anomaly
        ids.event_logger.log_security_event.assert_called()

        # Check for volume anomaly in calls
        calls = ids.event_logger.log_security_event.call_args_list
        volume_anomaly_calls = [
            call for call in calls if call[0][0] == "high_request_volume_detected"
        ]
        assert len(volume_anomaly_calls) > 0

    @pytest.mark.asyncio
    async def test_detect_time_based_anomaly(self, ids, security_context):
        """Test detection of unusual access times."""

        # First, establish normal hours pattern
        for _ in range(15):  # Above threshold for normal hours
            await ids.analyze_access_attempt(
                security_context, success=True, operation="login"
            )

        # Mock current time to be outside normal hours
        with patch("app.security.monitoring.datetime") as mock_datetime:
            mock_now = datetime.now(timezone.utc).replace(hour=2)  # 2 AM
            mock_datetime.now.return_value = mock_now
            mock_datetime.utc = timezone.utc

            await ids.analyze_access_attempt(
                security_context, success=True, operation="login"
            )

        # Should detect time-based anomaly
        pattern = ids.access_patterns["test_user"]
        assert len(pattern.access_times) > 15

    @pytest.mark.asyncio
    async def test_detect_geographic_anomaly(self, ids, security_context):
        """Test detection of geographic anomalies."""

        # Establish pattern with private IP
        private_context = SecurityContext(
            user_id="test_user",
            ip_address="192.168.1.100",  # Private IP
        )

        await ids.analyze_access_attempt(
            private_context, success=True, operation="login"
        )

        # Access from public IP
        public_context = SecurityContext(
            user_id="test_user",
            ip_address="8.8.8.8",  # Public IP
        )

        await ids.analyze_access_attempt(
            public_context, success=True, operation="login"
        )

        # Should detect geographic anomaly
        pattern = ids.access_patterns["test_user"]
        assert len(pattern.ip_addresses) == 2
        assert "192.168.1.100" in pattern.ip_addresses
        assert "8.8.8.8" in pattern.ip_addresses

    def test_is_private_ip(self, ids):
        """Test private IP detection."""

        assert ids._is_private_ip("192.168.1.1") is True
        assert ids._is_private_ip("10.0.0.1") is True
        assert ids._is_private_ip("172.16.0.1") is True
        assert ids._is_private_ip("8.8.8.8") is False
        assert ids._is_private_ip("1.1.1.1") is False
        assert ids._is_private_ip("invalid.ip") is False

    def test_get_access_pattern_summary(self, ids, security_context):
        """Test getting access pattern summary."""

        # No pattern exists
        summary = ids.get_access_pattern_summary("nonexistent_user")
        assert summary is None

        # Create pattern and get summary
        asyncio.run(
            ids.analyze_access_attempt(
                security_context, success=True, operation="login"
            )
        )

        summary = ids.get_access_pattern_summary("test_user")
        assert summary is not None
        assert summary["user_id"] == "test_user"
        assert summary["success_count"] == 1
        assert summary["failure_count"] == 0
        assert summary["unique_ips"] == 1


class TestSecurityAlertSystem:
    """Test security alert system functionality."""

    @pytest.fixture
    def event_logger(self):
        """Create mock event logger."""
        mock_logger = MagicMock(spec=SecurityEventLogger)
        mock_logger.get_events = AsyncMock(return_value=[])
        return mock_logger

    @pytest.fixture
    def alert_system(self, event_logger):
        """Create security alert system for testing."""
        return SecurityAlertSystem(event_logger)

    @pytest.fixture
    def security_event(self):
        """Create security event for testing."""
        from app.security.models import SecurityEvent

        return SecurityEvent(
            event_id="test_event_id",
            timestamp=datetime.now(timezone.utc),
            event_type="authentication_failure",
            severity=SecurityLevel.MEDIUM,
            source_ip="192.168.1.100",
            user_id="test_user",
            session_id="test_session",
            operation="login",
            threat_type=ThreatType.SUSPICIOUS_PATTERN,
            details={"reason": "invalid_password"},
            blocked=False,
        )

    @pytest.mark.asyncio
    async def test_process_security_event_no_alert(self, alert_system, security_event):
        """Test processing security event that doesn't trigger alerts."""

        # Modify event to not match any alert rules
        security_event.event_type = "normal_operation"
        security_event.severity = SecurityLevel.LOW

        await alert_system.process_security_event(security_event)

        # No alerts should be created
        assert len(alert_system.active_alerts) == 0

    @pytest.mark.asyncio
    async def test_process_high_risk_event_alert(self, alert_system, security_event):
        """Test processing high-risk security event that triggers alert."""

        # Set high severity to trigger alert
        security_event.severity = SecurityLevel.HIGH

        await alert_system.process_security_event(security_event)

        # Should create alert
        assert len(alert_system.active_alerts) == 1

        alert = list(alert_system.active_alerts.values())[0]
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.alert_type == "high_risk_security_event"
        assert alert.source_ip == "192.168.1.100"
        assert alert.user_id == "test_user"

    @pytest.mark.asyncio
    async def test_alert_rate_limiting(self, alert_system, security_event):
        """Test alert rate limiting to prevent spam."""

        security_event.severity = SecurityLevel.HIGH

        # Process same event twice quickly
        await alert_system.process_security_event(security_event)
        await alert_system.process_security_event(security_event)

        # Should only create one alert due to rate limiting
        assert len(alert_system.active_alerts) == 1

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, alert_system, security_event):
        """Test acknowledging alerts."""

        security_event.severity = SecurityLevel.HIGH
        await alert_system.process_security_event(security_event)

        alert_id = list(alert_system.active_alerts.keys())[0]

        # Acknowledge alert
        result = await alert_system.acknowledge_alert(alert_id, "admin_user")
        assert result is True

        alert = alert_system.active_alerts[alert_id]
        assert alert.acknowledged is True
        assert alert.acknowledged_by == "admin_user"
        assert alert.acknowledged_at is not None

    @pytest.mark.asyncio
    async def test_resolve_alert(self, alert_system, security_event):
        """Test resolving alerts."""

        security_event.severity = SecurityLevel.HIGH
        await alert_system.process_security_event(security_event)

        alert_id = list(alert_system.active_alerts.keys())[0]

        # Resolve alert
        result = await alert_system.resolve_alert(alert_id, "admin_user")
        assert result is True

        # Alert should be removed from active alerts
        assert len(alert_system.active_alerts) == 0

        # Alert should be in history
        assert len(alert_system.alert_history) == 1
        resolved_alert = alert_system.alert_history[0]
        assert resolved_alert.resolved is True
        assert resolved_alert.resolved_by == "admin_user"

    def test_get_recommended_actions(self, alert_system, security_event):
        """Test getting recommended actions for alerts."""

        actions = alert_system._get_recommended_actions(
            "multiple_failed_logins", security_event
        )

        assert len(actions) > 0
        assert any("account" in action.lower() for action in actions)
        assert any("review" in action.lower() for action in actions)

    def test_get_alert_statistics(self, alert_system):
        """Test getting alert statistics."""

        stats = alert_system.get_alert_statistics()

        assert "active_alerts_count" in stats
        assert "total_alerts_24h" in stats
        assert "active_by_severity" in stats
        assert "recent_by_severity" in stats
        assert "alert_rules_count" in stats

        assert stats["active_alerts_count"] == 0
        assert stats["alert_rules_count"] == len(alert_system.alert_rules)


class TestSecurityIntegration:
    """Test security monitoring integration functionality."""

    @pytest.mark.asyncio
    async def test_security_monitoring_workflow(self):
        """Test complete security monitoring workflow."""

        from app.security.monitoring import (
            log_security_event,
            security_alert_system,
            security_event_logger,
        )

        # Create security context
        context = SecurityContext(
            user_id="test_user",
            ip_address="192.168.1.100",
            operation="test_operation",
        )

        # Log security event
        event = await log_security_event(
            "test_security_event",
            SecurityLevel.HIGH,
            context,
            ThreatType.SUSPICIOUS_PATTERN,
            {"test": "data"},
        )

        # Verify event was logged
        assert event.event_type == "test_security_event"
        assert event.severity == SecurityLevel.HIGH

        # Verify event is in recent events
        recent_events = await security_event_logger.get_events(limit=10)
        assert len(recent_events) >= 1
        assert any(e.event_id == event.event_id for e in recent_events)

    @pytest.mark.asyncio
    async def test_threat_indicator_workflow(self):
        """Test threat indicator workflow."""

        from app.security.monitoring import add_threat_indicator, security_event_logger

        # Add threat indicator
        indicator = add_threat_indicator(
            indicator_type="ip",
            value="192.168.1.200",
            threat_level=SecurityLevel.HIGH,
            description="Test malicious IP",
        )

        # Verify indicator was added
        assert indicator.indicator_type == "ip"
        assert indicator.value == "192.168.1.200"

        # Verify indicator is stored
        assert len(security_event_logger.threat_indicators) >= 1
        assert any(
            t.value == "192.168.1.200"
            for t in security_event_logger.threat_indicators.values()
        )

    @pytest.mark.asyncio
    async def test_dashboard_data_generation(self):
        """Test security dashboard data generation."""

        from app.security.monitoring import get_security_dashboard

        dashboard_data = await get_security_dashboard()

        # Verify dashboard structure
        assert "timestamp" in dashboard_data
        assert "event_statistics" in dashboard_data
        assert "alert_statistics" in dashboard_data
        assert "active_alerts" in dashboard_data
        assert "recent_events" in dashboard_data
        assert "threat_indicators" in dashboard_data

        # Verify data types
        assert isinstance(dashboard_data["event_statistics"], dict)
        assert isinstance(dashboard_data["alert_statistics"], dict)
        assert isinstance(dashboard_data["active_alerts"], list)
        assert isinstance(dashboard_data["recent_events"], list)
        assert isinstance(dashboard_data["threat_indicators"], int)


if __name__ == "__main__":
    pytest.main([__file__])
