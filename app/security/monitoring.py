"""
Security monitoring and audit logging system.

This module provides comprehensive security monitoring capabilities including:
- Security event logging with threat detection
- Audit trail for security-sensitive operations
- Intrusion detection and anomaly monitoring
- Security alert system with notification capabilities
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from app.config import config
from app.logger import logger
from app.security.models import (
    SecurityContext,
    SecurityEvent,
    SecurityLevel,
    ThreatType,
)


class AnomalyType(Enum):
    """Types of security anomalies that can be detected."""

    UNUSUAL_ACCESS_PATTERN = "unusual_access_pattern"
    RAPID_FAILED_ATTEMPTS = "rapid_failed_attempts"
    SUSPICIOUS_USER_AGENT = "suspicious_user_agent"
    GEOGRAPHIC_ANOMALY = "geographic_anomaly"
    TIME_BASED_ANOMALY = "time_based_anomaly"
    VOLUME_ANOMALY = "volume_anomaly"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"


class AlertSeverity(Enum):
    """Security alert severity levels."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    """Available notification channels for security alerts."""

    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    SMS = "sms"


@dataclass
class SecurityAlert:
    """Represents a security alert."""

    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    alert_type: str
    title: str
    description: str
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    threat_indicators: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None


@dataclass
class AuditLogEntry:
    """Represents an audit log entry for security-sensitive operations."""

    entry_id: str
    timestamp: datetime
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source_ip: Optional[str] = None
    resource: Optional[str] = None
    action: str = ""
    result: str = "success"  # success, failure, error
    details: Dict[str, Any] = field(default_factory=dict)
    risk_level: SecurityLevel = SecurityLevel.LOW
    correlation_id: Optional[str] = None


@dataclass
class ThreatIndicator:
    """Represents a threat indicator for pattern matching."""

    indicator_id: str
    indicator_type: str  # ip, user_agent, pattern, etc.
    value: str
    threat_level: SecurityLevel
    description: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    source: str = "internal"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessPattern:
    """Tracks access patterns for anomaly detection."""

    user_id: str
    ip_addresses: Set[str] = field(default_factory=set)
    user_agents: Set[str] = field(default_factory=set)
    operations: List[str] = field(default_factory=list)
    access_times: List[datetime] = field(default_factory=list)
    failure_count: int = 0
    success_count: int = 0
    last_access: Optional[datetime] = None
    first_access: Optional[datetime] = None


class SecurityEventLogger:
    """Handles security event logging with enhanced threat detection."""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize security event logger."""
        self.storage_path = storage_path or Path.home() / ".openmanus" / "security"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.events_file = self.storage_path / "security_events.jsonl"
        self.audit_file = self.storage_path / "audit_log.jsonl"

        # In-memory storage for recent events (for performance)
        self.recent_events: deque = deque(maxlen=10000)
        self.recent_audit_entries: deque = deque(maxlen=10000)

        # Threat indicators
        self.threat_indicators: Dict[str, ThreatIndicator] = {}
        self.load_threat_indicators()

        # Event statistics for monitoring
        self.event_stats = defaultdict(int)
        self.hourly_stats = defaultdict(lambda: defaultdict(int))

    async def log_security_event(
        self,
        event_type: str,
        severity: SecurityLevel,
        context: SecurityContext,
        threat_type: Optional[ThreatType] = None,
        details: Optional[Dict[str, Any]] = None,
        blocked: bool = False,
    ) -> SecurityEvent:
        """Log a security event with enhanced context."""

        event = SecurityEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now(timezone.utc),
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

        # Store event
        self.recent_events.append(event)
        await self._persist_event(event)

        # Update statistics
        self._update_event_stats(event)

        # Check for threat indicators
        await self._check_threat_indicators(event)

        # Log to application logger
        log_level = self._get_log_level(severity)
        getattr(logger, log_level)(
            f"Security Event: {event_type}",
            {
                "event_id": event.event_id,
                "severity": severity.value,
                "threat_type": threat_type.value if threat_type else None,
                "source_ip": context.ip_address,
                "user_id": context.user_id,
                "blocked": blocked,
                "details": details,
            },
        )

        return event

    async def log_audit_entry(
        self,
        operation: str,
        action: str,
        context: SecurityContext,
        resource: Optional[str] = None,
        result: str = "success",
        details: Optional[Dict[str, Any]] = None,
        risk_level: SecurityLevel = SecurityLevel.LOW,
    ) -> AuditLogEntry:
        """Log an audit entry for security-sensitive operations."""

        entry = AuditLogEntry(
            entry_id=self._generate_event_id(),
            timestamp=datetime.now(timezone.utc),
            operation=operation,
            user_id=context.user_id,
            session_id=context.session_id,
            source_ip=context.ip_address,
            resource=resource,
            action=action,
            result=result,
            details=details or {},
            risk_level=risk_level,
            correlation_id=getattr(context, "correlation_id", None),
        )

        # Store audit entry
        self.recent_audit_entries.append(entry)
        await self._persist_audit_entry(entry)

        # Log to application logger
        logger.info(
            f"Audit: {operation} - {action}",
            {
                "entry_id": entry.entry_id,
                "operation": operation,
                "action": action,
                "user_id": context.user_id,
                "source_ip": context.ip_address,
                "resource": resource,
                "result": result,
                "risk_level": risk_level.value,
                "details": details,
            },
        )

        return entry

    def add_threat_indicator(
        self,
        indicator_type: str,
        value: str,
        threat_level: SecurityLevel,
        description: str,
        expires_at: Optional[datetime] = None,
        source: str = "manual",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ThreatIndicator:
        """Add a threat indicator for monitoring."""

        indicator = ThreatIndicator(
            indicator_id=self._generate_event_id(),
            indicator_type=indicator_type,
            value=value,
            threat_level=threat_level,
            description=description,
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
            source=source,
            metadata=metadata or {},
        )

        self.threat_indicators[indicator.indicator_id] = indicator
        self._save_threat_indicators()

        logger.info(
            f"Added threat indicator: {indicator_type} - {value}",
            {
                "indicator_id": indicator.indicator_id,
                "threat_level": threat_level.value,
                "description": description,
                "source": source,
            },
        )

        return indicator

    def remove_threat_indicator(self, indicator_id: str) -> bool:
        """Remove a threat indicator."""

        if indicator_id in self.threat_indicators:
            indicator = self.threat_indicators.pop(indicator_id)
            self._save_threat_indicators()

            logger.info(
                f"Removed threat indicator: {indicator.indicator_type} - {indicator.value}",
                {"indicator_id": indicator_id},
            )
            return True

        return False

    async def get_events(
        self,
        limit: int = 100,
        severity: Optional[SecurityLevel] = None,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None,
    ) -> List[SecurityEvent]:
        """Get security events with filtering."""

        events = list(self.recent_events)

        # Apply filters
        if severity:
            events = [e for e in events if e.severity == severity]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if since:
            events = [e for e in events if e.timestamp >= since]

        if user_id:
            events = [e for e in events if e.user_id == user_id]

        if source_ip:
            events = [e for e in events if e.source_ip == source_ip]

        # Sort by timestamp (newest first) and limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    async def get_audit_entries(
        self,
        limit: int = 100,
        operation: Optional[str] = None,
        user_id: Optional[str] = None,
        since: Optional[datetime] = None,
        result: Optional[str] = None,
    ) -> List[AuditLogEntry]:
        """Get audit entries with filtering."""

        entries = list(self.recent_audit_entries)

        # Apply filters
        if operation:
            entries = [e for e in entries if e.operation == operation]

        if user_id:
            entries = [e for e in entries if e.user_id == user_id]

        if since:
            entries = [e for e in entries if e.timestamp >= since]

        if result:
            entries = [e for e in entries if e.result == result]

        # Sort by timestamp (newest first) and limit
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries[:limit]

    def get_event_statistics(self) -> Dict[str, Any]:
        """Get security event statistics."""

        now = datetime.now(timezone.utc)
        hour_key = now.strftime("%Y-%m-%d-%H")

        return {
            "total_events": len(self.recent_events),
            "total_audit_entries": len(self.recent_audit_entries),
            "event_types": dict(self.event_stats),
            "hourly_stats": dict(self.hourly_stats),
            "current_hour_events": dict(self.hourly_stats.get(hour_key, {})),
            "threat_indicators_count": len(self.threat_indicators),
            "active_threat_indicators": len(
                [
                    t
                    for t in self.threat_indicators.values()
                    if not t.expires_at or t.expires_at > now
                ]
            ),
        }

    async def _persist_event(self, event: SecurityEvent):
        """Persist security event to storage."""

        try:
            with open(self.events_file, "a") as f:
                event_data = {
                    "event_id": event.event_id,
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type,
                    "severity": (
                        event.severity.value
                        if hasattr(event.severity, "value")
                        else event.severity
                    ),
                    "source_ip": event.source_ip,
                    "user_id": event.user_id,
                    "session_id": event.session_id,
                    "operation": event.operation,
                    "threat_type": (
                        event.threat_type.value
                        if event.threat_type and hasattr(event.threat_type, "value")
                        else str(event.threat_type) if event.threat_type else None
                    ),
                    "details": event.details,
                    "blocked": event.blocked,
                }
                f.write(json.dumps(event_data) + "\n")
        except Exception as e:
            logger.error(f"Failed to persist security event: {e}")

    async def _persist_audit_entry(self, entry: AuditLogEntry):
        """Persist audit entry to storage."""

        try:
            with open(self.audit_file, "a") as f:
                entry_data = {
                    "entry_id": entry.entry_id,
                    "timestamp": entry.timestamp.isoformat(),
                    "operation": entry.operation,
                    "user_id": entry.user_id,
                    "session_id": entry.session_id,
                    "source_ip": entry.source_ip,
                    "resource": entry.resource,
                    "action": entry.action,
                    "result": entry.result,
                    "details": entry.details,
                    "risk_level": (
                        entry.risk_level.value
                        if hasattr(entry.risk_level, "value")
                        else entry.risk_level
                    ),
                    "correlation_id": entry.correlation_id,
                }
                f.write(json.dumps(entry_data) + "\n")
        except Exception as e:
            logger.error(f"Failed to persist audit entry: {e}")

    def _update_event_stats(self, event: SecurityEvent):
        """Update event statistics."""

        self.event_stats[event.event_type] += 1

        # Update hourly statistics
        hour_key = event.timestamp.strftime("%Y-%m-%d-%H")
        self.hourly_stats[hour_key][event.event_type] += 1
        self.hourly_stats[hour_key]["total"] += 1

        # Clean old hourly stats (keep last 24 hours)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        cutoff_key = cutoff_time.strftime("%Y-%m-%d-%H")

        keys_to_remove = [key for key in self.hourly_stats.keys() if key < cutoff_key]
        for key in keys_to_remove:
            del self.hourly_stats[key]

    async def _check_threat_indicators(self, event: SecurityEvent):
        """Check event against threat indicators."""

        now = datetime.now(timezone.utc)

        for indicator in self.threat_indicators.values():
            # Skip expired indicators
            if indicator.expires_at and indicator.expires_at <= now:
                continue

            # Check different indicator types
            match_found = False

            if indicator.indicator_type == "ip" and event.source_ip == indicator.value:
                match_found = True
            elif (
                indicator.indicator_type == "user_id"
                and event.user_id == indicator.value
            ):
                match_found = True
            elif indicator.indicator_type == "pattern" and indicator.value in str(
                event.details
            ):
                match_found = True

            if match_found:
                logger.warning(
                    f"Threat indicator match: {indicator.description}",
                    {
                        "event_id": event.event_id,
                        "indicator_id": indicator.indicator_id,
                        "indicator_type": indicator.indicator_type,
                        "threat_level": indicator.threat_level.value,
                    },
                )

    def load_threat_indicators(self):
        """Load threat indicators from storage."""

        indicators_file = self.storage_path / "threat_indicators.json"

        if indicators_file.exists():
            try:
                with open(indicators_file, "r") as f:
                    data = json.load(f)

                for indicator_data in data.get("indicators", []):
                    indicator = ThreatIndicator(
                        indicator_id=indicator_data["indicator_id"],
                        indicator_type=indicator_data["indicator_type"],
                        value=indicator_data["value"],
                        threat_level=SecurityLevel(indicator_data["threat_level"]),
                        description=indicator_data["description"],
                        created_at=datetime.fromisoformat(indicator_data["created_at"]),
                        expires_at=(
                            datetime.fromisoformat(indicator_data["expires_at"])
                            if indicator_data.get("expires_at")
                            else None
                        ),
                        source=indicator_data.get("source", "unknown"),
                        metadata=indicator_data.get("metadata", {}),
                    )
                    self.threat_indicators[indicator.indicator_id] = indicator

                logger.info(f"Loaded {len(self.threat_indicators)} threat indicators")

            except Exception as e:
                logger.error(f"Failed to load threat indicators: {e}")

    def _save_threat_indicators(self):
        """Save threat indicators to storage."""

        indicators_file = self.storage_path / "threat_indicators.json"

        try:
            data = {
                "indicators": [
                    {
                        "indicator_id": indicator.indicator_id,
                        "indicator_type": indicator.indicator_type,
                        "value": indicator.value,
                        "threat_level": indicator.threat_level.value,
                        "description": indicator.description,
                        "created_at": indicator.created_at.isoformat(),
                        "expires_at": (
                            indicator.expires_at.isoformat()
                            if indicator.expires_at
                            else None
                        ),
                        "source": indicator.source,
                        "metadata": indicator.metadata,
                    }
                    for indicator in self.threat_indicators.values()
                ]
            }

            with open(indicators_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save threat indicators: {e}")

    def _generate_event_id(self) -> str:
        """Generate a unique event ID."""
        import uuid

        return str(uuid.uuid4())

    def _get_log_level(self, severity: SecurityLevel) -> str:
        """Get appropriate log level for security severity."""

        if severity == SecurityLevel.CRITICAL:
            return "critical"
        elif severity == SecurityLevel.HIGH:
            return "error"
        elif severity == SecurityLevel.MEDIUM:
            return "warning"
        else:
            return "info"


class IntrusionDetectionSystem:
    """Detects intrusion attempts and security anomalies."""

    def __init__(self, event_logger: SecurityEventLogger):
        """Initialize intrusion detection system."""
        self.event_logger = event_logger

        # Track access patterns for anomaly detection
        self.access_patterns: Dict[str, AccessPattern] = {}

        # Detection thresholds
        self.failed_login_threshold = 5
        self.rapid_request_threshold = 100  # requests per minute
        self.unusual_hour_threshold = 2  # outside normal hours
        self.geographic_distance_threshold = 1000  # km

        # Time windows for analysis
        self.analysis_window = timedelta(minutes=15)
        self.pattern_retention = timedelta(days=7)

        # Cleanup task
        self._cleanup_task = None
        # Don't start cleanup task during initialization - will be started when needed

    def ensure_cleanup_task_started(self):
        """Ensure cleanup task is started if event loop is available."""
        if self._cleanup_task is None:
            try:
                self._start_cleanup_task()
            except RuntimeError:
                pass  # No event loop available

    async def analyze_access_attempt(
        self,
        context: SecurityContext,
        success: bool,
        operation: str,
    ) -> List[AnomalyType]:
        """Analyze an access attempt for anomalies."""

        anomalies = []

        if not context.user_id:
            return anomalies

        # Get or create access pattern
        pattern = self.access_patterns.get(context.user_id)
        if not pattern:
            pattern = AccessPattern(user_id=context.user_id)
            self.access_patterns[context.user_id] = pattern

        # Update pattern
        now = datetime.now(timezone.utc)
        pattern.last_access = now
        if not pattern.first_access:
            pattern.first_access = now

        if context.ip_address:
            pattern.ip_addresses.add(context.ip_address)

        if context.user_agent:
            pattern.user_agents.add(context.user_agent)

        pattern.operations.append(operation)
        pattern.access_times.append(now)

        if success:
            pattern.success_count += 1
        else:
            pattern.failure_count += 1

        # Keep only recent data
        cutoff_time = now - self.pattern_retention
        pattern.access_times = [t for t in pattern.access_times if t > cutoff_time]
        pattern.operations = pattern.operations[-1000:]  # Keep last 1000 operations

        # Detect anomalies
        anomalies.extend(await self._detect_failed_login_anomaly(pattern, context))
        anomalies.extend(await self._detect_rapid_requests_anomaly(pattern, context))
        anomalies.extend(await self._detect_time_based_anomaly(pattern, context))
        anomalies.extend(await self._detect_behavioral_anomaly(pattern, context))
        anomalies.extend(await self._detect_geographic_anomaly(pattern, context))

        return anomalies

    async def _detect_failed_login_anomaly(
        self,
        pattern: AccessPattern,
        context: SecurityContext,
    ) -> List[AnomalyType]:
        """Detect rapid failed login attempts."""

        anomalies = []

        # Check recent failed attempts
        recent_time = datetime.now(timezone.utc) - self.analysis_window
        recent_failures = sum(
            1 for t in pattern.access_times[-pattern.failure_count :] if t > recent_time
        )

        if recent_failures >= self.failed_login_threshold:
            anomalies.append(AnomalyType.RAPID_FAILED_ATTEMPTS)

            await self.event_logger.log_security_event(
                "rapid_failed_attempts_detected",
                SecurityLevel.HIGH,
                context,
                ThreatType.SUSPICIOUS_PATTERN,
                {
                    "failed_attempts": recent_failures,
                    "threshold": self.failed_login_threshold,
                    "time_window_minutes": self.analysis_window.total_seconds() / 60,
                },
            )

        return anomalies

    async def _detect_rapid_requests_anomaly(
        self,
        pattern: AccessPattern,
        context: SecurityContext,
    ) -> List[AnomalyType]:
        """Detect unusually high request volume."""

        anomalies = []

        # Check requests in last minute
        one_minute_ago = datetime.now(timezone.utc) - timedelta(minutes=1)
        recent_requests = sum(1 for t in pattern.access_times if t > one_minute_ago)

        if recent_requests >= self.rapid_request_threshold:
            anomalies.append(AnomalyType.VOLUME_ANOMALY)

            await self.event_logger.log_security_event(
                "high_request_volume_detected",
                SecurityLevel.MEDIUM,
                context,
                ThreatType.RATE_LIMIT_EXCEEDED,
                {
                    "requests_per_minute": recent_requests,
                    "threshold": self.rapid_request_threshold,
                },
            )

        return anomalies

    async def _detect_time_based_anomaly(
        self,
        pattern: AccessPattern,
        context: SecurityContext,
    ) -> List[AnomalyType]:
        """Detect access outside normal hours."""

        anomalies = []

        now = datetime.now(timezone.utc)
        current_hour = now.hour

        # Define normal business hours (9 AM to 6 PM UTC)
        if current_hour < 9 or current_hour > 18:
            # Check if user normally accesses during these hours
            normal_hours_access = sum(
                1 for t in pattern.access_times if 9 <= t.hour <= 18
            )

            off_hours_access = sum(
                1 for t in pattern.access_times if t.hour < 9 or t.hour > 18
            )

            # If user rarely accesses during off hours
            if normal_hours_access > 10 and off_hours_access < 3:
                anomalies.append(AnomalyType.TIME_BASED_ANOMALY)

                await self.event_logger.log_security_event(
                    "unusual_access_time_detected",
                    SecurityLevel.LOW,
                    context,
                    ThreatType.SUSPICIOUS_PATTERN,
                    {
                        "access_hour": current_hour,
                        "normal_hours_access": normal_hours_access,
                        "off_hours_access": off_hours_access,
                    },
                )

        return anomalies

    async def _detect_behavioral_anomaly(
        self,
        pattern: AccessPattern,
        context: SecurityContext,
    ) -> List[AnomalyType]:
        """Detect unusual behavioral patterns."""

        anomalies = []

        # Check for unusual operations
        recent_operations = pattern.operations[-50:]  # Last 50 operations
        operation_counts = defaultdict(int)

        for op in recent_operations:
            operation_counts[op] += 1

        # Check for operations that are unusual for this user
        total_operations = len(pattern.operations)
        if total_operations > 100:  # Only analyze if we have enough data
            for operation, count in operation_counts.items():
                historical_frequency = (
                    pattern.operations.count(operation) / total_operations
                )
                recent_frequency = count / len(recent_operations)

                # If recent frequency is significantly higher than historical
                if (
                    recent_frequency > historical_frequency * 3
                    and recent_frequency > 0.3
                ):
                    anomalies.append(AnomalyType.BEHAVIORAL_ANOMALY)

                    await self.event_logger.log_security_event(
                        "unusual_operation_pattern_detected",
                        SecurityLevel.LOW,
                        context,
                        ThreatType.SUSPICIOUS_PATTERN,
                        {
                            "operation": operation,
                            "recent_frequency": recent_frequency,
                            "historical_frequency": historical_frequency,
                            "recent_count": count,
                        },
                    )
                    break  # Only report one behavioral anomaly per analysis

        return anomalies

    async def _detect_geographic_anomaly(
        self,
        pattern: AccessPattern,
        context: SecurityContext,
    ) -> List[AnomalyType]:
        """Detect access from unusual geographic locations."""

        anomalies = []

        # This is a simplified implementation
        # In a real system, you would use IP geolocation services

        if context.ip_address and len(pattern.ip_addresses) > 1:
            # Check if this is a new IP address
            if context.ip_address not in pattern.ip_addresses:
                # Simple check for private vs public IP ranges
                ip_parts = context.ip_address.split(".")
                if len(ip_parts) == 4:
                    try:
                        first_octet = int(ip_parts[0])
                        # Check if switching between private and public IPs
                        has_private = any(
                            self._is_private_ip(ip) for ip in pattern.ip_addresses
                        )
                        current_is_private = self._is_private_ip(context.ip_address)

                        if has_private != current_is_private:
                            anomalies.append(AnomalyType.GEOGRAPHIC_ANOMALY)

                            await self.event_logger.log_security_event(
                                "geographic_anomaly_detected",
                                SecurityLevel.MEDIUM,
                                context,
                                ThreatType.SUSPICIOUS_PATTERN,
                                {
                                    "new_ip": context.ip_address,
                                    "previous_ips": list(pattern.ip_addresses),
                                    "network_change": (
                                        "private_to_public"
                                        if current_is_private
                                        else "public_to_private"
                                    ),
                                },
                            )
                    except ValueError:
                        pass  # Invalid IP format

        return anomalies

    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP address is in private range."""
        try:
            parts = [int(x) for x in ip.split(".")]
            if len(parts) != 4:
                return False

            # Private IP ranges
            if parts[0] == 10:
                return True
            elif parts[0] == 172 and 16 <= parts[1] <= 31:
                return True
            elif parts[0] == 192 and parts[1] == 168:
                return True

            return False
        except (ValueError, IndexError):
            return False

    def get_access_pattern_summary(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get access pattern summary for a user."""

        pattern = self.access_patterns.get(user_id)
        if not pattern:
            return None

        now = datetime.now(timezone.utc)
        recent_time = now - timedelta(hours=24)

        recent_accesses = [t for t in pattern.access_times if t > recent_time]

        return {
            "user_id": user_id,
            "first_access": (
                pattern.first_access.isoformat() if pattern.first_access else None
            ),
            "last_access": (
                pattern.last_access.isoformat() if pattern.last_access else None
            ),
            "total_accesses": len(pattern.access_times),
            "recent_accesses_24h": len(recent_accesses),
            "success_count": pattern.success_count,
            "failure_count": pattern.failure_count,
            "unique_ips": len(pattern.ip_addresses),
            "unique_user_agents": len(pattern.user_agents),
            "common_operations": self._get_common_operations(pattern.operations),
        }

    def _get_common_operations(self, operations: List[str]) -> List[Tuple[str, int]]:
        """Get most common operations."""

        operation_counts = defaultdict(int)
        for op in operations:
            operation_counts[op] += 1

        return sorted(operation_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    def _start_cleanup_task(self):
        """Start background cleanup task."""

        async def cleanup_task():
            while True:
                try:
                    await asyncio.sleep(3600)  # Run every hour
                    await self._cleanup_old_patterns()
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")

        try:
            self._cleanup_task = asyncio.create_task(cleanup_task())
        except RuntimeError:
            # No event loop running, cleanup task will be started later
            pass

    async def _cleanup_old_patterns(self):
        """Clean up old access patterns."""

        cutoff_time = datetime.now(timezone.utc) - self.pattern_retention

        users_to_remove = []
        for user_id, pattern in self.access_patterns.items():
            if pattern.last_access and pattern.last_access < cutoff_time:
                users_to_remove.append(user_id)

        for user_id in users_to_remove:
            del self.access_patterns[user_id]

        if users_to_remove:
            logger.info(f"Cleaned up {len(users_to_remove)} old access patterns")


class SecurityAlertSystem:
    """Manages security alerts and notifications."""

    def __init__(self, event_logger: SecurityEventLogger):
        """Initialize security alert system."""
        self.event_logger = event_logger

        # Alert storage
        self.active_alerts: Dict[str, SecurityAlert] = {}
        self.alert_history: deque = deque(maxlen=10000)

        # Notification handlers
        self.notification_handlers: Dict[NotificationChannel, Callable] = {
            NotificationChannel.LOG: self._log_notification,
        }

        # Alert rules and thresholds
        self.alert_rules = self._setup_default_alert_rules()

        # Rate limiting for alerts to prevent spam
        self.alert_rate_limits: Dict[str, datetime] = {}
        self.alert_cooldown = timedelta(minutes=5)

    def _setup_default_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """Set up default alert rules."""

        return {
            "multiple_failed_logins": {
                "threshold": 5,
                "time_window": timedelta(minutes=15),
                "severity": AlertSeverity.HIGH,
                "description": "Multiple failed login attempts detected",
            },
            "high_risk_security_event": {
                "threshold": 1,
                "time_window": timedelta(minutes=1),
                "severity": AlertSeverity.CRITICAL,
                "description": "High-risk security event detected",
            },
            "unusual_access_pattern": {
                "threshold": 3,
                "time_window": timedelta(hours=1),
                "severity": AlertSeverity.MEDIUM,
                "description": "Unusual access pattern detected",
            },
            "threat_indicator_match": {
                "threshold": 1,
                "time_window": timedelta(minutes=1),
                "severity": AlertSeverity.HIGH,
                "description": "Known threat indicator detected",
            },
        }

    async def process_security_event(self, event: SecurityEvent):
        """Process a security event and generate alerts if needed."""

        # Check alert rules
        for rule_name, rule_config in self.alert_rules.items():
            if await self._should_trigger_alert(event, rule_name, rule_config):
                await self._create_alert(event, rule_name, rule_config)

    async def _should_trigger_alert(
        self,
        event: SecurityEvent,
        rule_name: str,
        rule_config: Dict[str, Any],
    ) -> bool:
        """Check if an alert should be triggered for this event."""

        # Check rate limiting
        last_alert_time = self.alert_rate_limits.get(rule_name)
        if (
            last_alert_time
            and datetime.now(timezone.utc) - last_alert_time < self.alert_cooldown
        ):
            return False

        # Rule-specific logic
        if rule_name == "multiple_failed_logins":
            return await self._check_failed_login_rule(event, rule_config)
        elif rule_name == "high_risk_security_event":
            return event.severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]
        elif rule_name == "unusual_access_pattern":
            return "anomaly" in event.event_type.lower()
        elif rule_name == "threat_indicator_match":
            return "threat_indicator" in event.event_type.lower()

        return False

    async def _check_failed_login_rule(
        self,
        event: SecurityEvent,
        rule_config: Dict[str, Any],
    ) -> bool:
        """Check failed login rule."""

        if (
            "failed" not in event.event_type.lower()
            and "login" not in event.event_type.lower()
        ):
            return False

        # Count recent failed login events for this user/IP
        threshold = rule_config["threshold"]
        time_window = rule_config["time_window"]
        cutoff_time = datetime.now(timezone.utc) - time_window

        recent_events = await self.event_logger.get_events(
            limit=1000,
            since=cutoff_time,
            user_id=event.user_id,
            source_ip=event.source_ip,
        )

        failed_login_count = sum(
            1
            for e in recent_events
            if "failed" in e.event_type.lower() and "login" in e.event_type.lower()
        )

        return failed_login_count >= threshold

    async def _create_alert(
        self,
        event: SecurityEvent,
        rule_name: str,
        rule_config: Dict[str, Any],
    ):
        """Create a security alert."""

        alert_id = f"{rule_name}_{event.event_id}"

        # Generate recommended actions based on alert type
        recommended_actions = self._get_recommended_actions(rule_name, event)

        alert = SecurityAlert(
            alert_id=alert_id,
            timestamp=datetime.now(timezone.utc),
            severity=rule_config["severity"],
            alert_type=rule_name,
            title=rule_config["description"],
            description=f"{rule_config['description']}: {event.event_type}",
            source_ip=event.source_ip,
            user_id=event.user_id,
            session_id=event.session_id,
            threat_indicators=[event.threat_type.value] if event.threat_type else [],
            recommended_actions=recommended_actions,
            metadata={
                "triggering_event": event.event_id,
                "rule_name": rule_name,
                "event_details": event.details,
            },
        )

        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Update rate limiting
        self.alert_rate_limits[rule_name] = datetime.now(timezone.utc)

        # Send notifications
        await self._send_notifications(alert)

        logger.warning(
            f"Security Alert Created: {alert.title}",
            {
                "alert_id": alert_id,
                "severity": alert.severity.value,
                "alert_type": rule_name,
                "source_ip": event.source_ip,
                "user_id": event.user_id,
            },
        )

    def _get_recommended_actions(
        self, rule_name: str, event: SecurityEvent
    ) -> List[str]:
        """Get recommended actions for an alert type."""

        actions = {
            "multiple_failed_logins": [
                "Review user account for compromise",
                "Consider temporary account lockout",
                "Check for credential stuffing attacks",
                "Verify IP address legitimacy",
            ],
            "high_risk_security_event": [
                "Immediate investigation required",
                "Review system logs for related events",
                "Consider isolating affected systems",
                "Notify security team",
            ],
            "unusual_access_pattern": [
                "Verify user identity through secondary channel",
                "Review recent account activity",
                "Check for account takeover indicators",
                "Monitor for additional anomalies",
            ],
            "threat_indicator_match": [
                "Block associated IP/user immediately",
                "Review all recent activity from source",
                "Check for lateral movement",
                "Update threat intelligence feeds",
            ],
        }

        return actions.get(
            rule_name, ["Review event details and take appropriate action"]
        )

    async def _send_notifications(self, alert: SecurityAlert):
        """Send alert notifications through configured channels."""

        # Determine notification channels based on severity
        channels = [NotificationChannel.LOG]  # Always log

        if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            # Add additional channels for high-severity alerts
            # This would be configured based on available integrations
            pass

        # Send notifications
        for channel in channels:
            handler = self.notification_handlers.get(channel)
            if handler:
                try:
                    await handler(alert)
                except Exception as e:
                    logger.error(
                        f"Failed to send notification via {channel.value}: {e}"
                    )

    async def _log_notification(self, alert: SecurityAlert):
        """Send alert notification via logging."""

        logger.warning(
            f"SECURITY ALERT: {alert.title}",
            {
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "description": alert.description,
                "source_ip": alert.source_ip,
                "user_id": alert.user_id,
                "recommended_actions": alert.recommended_actions,
                "metadata": alert.metadata,
            },
        )

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an active alert."""

        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now(timezone.utc)

            logger.info(
                f"Alert acknowledged: {alert.title}",
                {
                    "alert_id": alert_id,
                    "acknowledged_by": acknowledged_by,
                },
            )
            return True

        return False

    async def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve an active alert."""

        if alert_id in self.active_alerts:
            alert = self.active_alerts.pop(alert_id)
            alert.resolved = True
            alert.resolved_by = resolved_by
            alert.resolved_at = datetime.now(timezone.utc)

            logger.info(
                f"Alert resolved: {alert.title}",
                {
                    "alert_id": alert_id,
                    "resolved_by": resolved_by,
                },
            )
            return True

        return False

    def get_active_alerts(
        self, severity: Optional[AlertSeverity] = None
    ) -> List[SecurityAlert]:
        """Get active alerts, optionally filtered by severity."""

        alerts = list(self.active_alerts.values())

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        # Sort by timestamp (newest first)
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        return alerts

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""

        now = datetime.now(timezone.utc)
        last_24h = now - timedelta(hours=24)

        recent_alerts = [a for a in self.alert_history if a.timestamp > last_24h]

        severity_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            severity_counts[alert.severity.value] += 1

        recent_severity_counts = defaultdict(int)
        for alert in recent_alerts:
            recent_severity_counts[alert.severity.value] += 1

        return {
            "active_alerts_count": len(self.active_alerts),
            "total_alerts_24h": len(recent_alerts),
            "active_by_severity": dict(severity_counts),
            "recent_by_severity": dict(recent_severity_counts),
            "alert_rules_count": len(self.alert_rules),
        }


# Global instances for easy access
security_event_logger = SecurityEventLogger()
intrusion_detection_system = IntrusionDetectionSystem(security_event_logger)
security_alert_system = SecurityAlertSystem(security_event_logger)


# Convenience functions for easy integration
async def log_security_event(
    event_type: str,
    severity: SecurityLevel,
    context: SecurityContext,
    threat_type: Optional[ThreatType] = None,
    details: Optional[Dict[str, Any]] = None,
    blocked: bool = False,
) -> SecurityEvent:
    """Convenience function to log a security event."""

    event = await security_event_logger.log_security_event(
        event_type, severity, context, threat_type, details, blocked
    )

    # Process event for alerts
    await security_alert_system.process_security_event(event)

    return event


async def log_audit_entry(
    operation: str,
    action: str,
    context: SecurityContext,
    resource: Optional[str] = None,
    result: str = "success",
    details: Optional[Dict[str, Any]] = None,
    risk_level: SecurityLevel = SecurityLevel.LOW,
) -> AuditLogEntry:
    """Convenience function to log an audit entry."""

    return await security_event_logger.log_audit_entry(
        operation, action, context, resource, result, details, risk_level
    )


async def analyze_access_attempt(
    context: SecurityContext,
    success: bool,
    operation: str,
) -> List[AnomalyType]:
    """Convenience function to analyze an access attempt."""

    return await intrusion_detection_system.analyze_access_attempt(
        context, success, operation
    )


def add_threat_indicator(
    indicator_type: str,
    value: str,
    threat_level: SecurityLevel,
    description: str,
    expires_at: Optional[datetime] = None,
    source: str = "manual",
    metadata: Optional[Dict[str, Any]] = None,
) -> ThreatIndicator:
    """Convenience function to add a threat indicator."""

    return security_event_logger.add_threat_indicator(
        indicator_type, value, threat_level, description, expires_at, source, metadata
    )


async def get_security_dashboard() -> Dict[str, Any]:
    """Get comprehensive security monitoring dashboard data."""

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_statistics": security_event_logger.get_event_statistics(),
        "alert_statistics": security_alert_system.get_alert_statistics(),
        "active_alerts": [
            asdict(alert) for alert in security_alert_system.get_active_alerts()
        ],
        "recent_events": [
            {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "severity": (
                    event.severity.value
                    if hasattr(event.severity, "value")
                    else event.severity
                ),
                "source_ip": event.source_ip,
                "user_id": event.user_id,
                "blocked": event.blocked,
            }
            for event in await security_event_logger.get_events(limit=50)
        ],
        "threat_indicators": len(security_event_logger.threat_indicators),
    }
