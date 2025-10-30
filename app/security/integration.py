"""
Security monitoring integration module.

This module provides integration points for the security monitoring system
with the rest of the OpenManus application.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.logger import logger
from app.security.models import SecurityContext, SecurityLevel, ThreatType
from app.security.monitoring import (
    analyze_access_attempt,
    log_audit_entry,
    log_security_event,
    security_alert_system,
    security_event_logger,
)


class SecurityMonitoringMiddleware:
    """Middleware for integrating security monitoring with application operations."""

    def __init__(self):
        """Initialize security monitoring middleware."""
        self.enabled = True
        self.audit_sensitive_operations = {
            "credential_access",
            "file_write",
            "file_delete",
            "command_execution",
            "network_request",
            "configuration_change",
            "user_authentication",
            "permission_change",
        }

    async def monitor_operation(
        self,
        operation: str,
        context: SecurityContext,
        operation_func: callable,
        *args,
        **kwargs,
    ) -> Any:
        """Monitor an operation and log security events."""

        if not self.enabled:
            return await operation_func(*args, **kwargs)

        start_time = datetime.now(timezone.utc)
        result = None
        success = False
        error = None

        try:
            # Log operation start for sensitive operations
            if operation in self.audit_sensitive_operations:
                await log_audit_entry(
                    operation=operation,
                    action="start",
                    context=context,
                    details={
                        "start_time": start_time.isoformat(),
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                    },
                )

            # Execute the operation
            result = await operation_func(*args, **kwargs)
            success = True

            # Analyze access attempt for anomalies
            anomalies = await analyze_access_attempt(context, success, operation)

            if anomalies:
                await log_security_event(
                    "access_anomalies_detected",
                    SecurityLevel.MEDIUM,
                    context,
                    ThreatType.SUSPICIOUS_PATTERN,
                    {
                        "operation": operation,
                        "anomalies": [a.value for a in anomalies],
                        "anomaly_count": len(anomalies),
                    },
                )

        except Exception as e:
            error = str(e)
            success = False

            # Log security event for failed operations
            await log_security_event(
                "operation_failed",
                SecurityLevel.MEDIUM,
                context,
                ThreatType.SUSPICIOUS_PATTERN,
                {
                    "operation": operation,
                    "error": error,
                    "duration_ms": (
                        datetime.now(timezone.utc) - start_time
                    ).total_seconds()
                    * 1000,
                },
            )

            raise

        finally:
            # Log operation completion for sensitive operations
            if operation in self.audit_sensitive_operations:
                end_time = datetime.now(timezone.utc)
                duration = (end_time - start_time).total_seconds()

                await log_audit_entry(
                    operation=operation,
                    action="complete",
                    context=context,
                    result="success" if success else "failure",
                    details={
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "duration_seconds": duration,
                        "success": success,
                        "error": error,
                    },
                    risk_level=SecurityLevel.HIGH if not success else SecurityLevel.LOW,
                )

        return result

    async def monitor_authentication(
        self,
        context: SecurityContext,
        auth_method: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Monitor authentication attempts."""

        if not self.enabled:
            return

        event_type = "authentication_success" if success else "authentication_failure"
        severity = SecurityLevel.LOW if success else SecurityLevel.MEDIUM

        await log_security_event(
            event_type,
            severity,
            context,
            None if success else ThreatType.SUSPICIOUS_PATTERN,
            {
                "auth_method": auth_method,
                "success": success,
                **(details or {}),
            },
        )

        # Log audit entry
        await log_audit_entry(
            operation="user_authentication",
            action=auth_method,
            context=context,
            result="success" if success else "failure",
            details={
                "auth_method": auth_method,
                **(details or {}),
            },
            risk_level=SecurityLevel.MEDIUM,
        )

        # Analyze for anomalies
        anomalies = await analyze_access_attempt(context, success, "authentication")

        if anomalies and not success:
            await log_security_event(
                "authentication_anomalies_detected",
                SecurityLevel.HIGH,
                context,
                ThreatType.SUSPICIOUS_PATTERN,
                {
                    "auth_method": auth_method,
                    "anomalies": [a.value for a in anomalies],
                    "failed_attempt": True,
                },
            )

    async def monitor_file_operation(
        self,
        context: SecurityContext,
        operation: str,
        file_path: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Monitor file operations."""

        if not self.enabled:
            return

        # Determine risk level based on file path and operation
        risk_level = SecurityLevel.LOW

        # Higher risk for system files or sensitive operations
        sensitive_paths = [
            "/etc/",
            "/bin/",
            "/usr/",
            "C:\\Windows\\",
            "C:\\Program Files\\",
        ]
        sensitive_operations = ["delete", "execute", "modify_permissions"]

        if any(path in file_path for path in sensitive_paths):
            risk_level = SecurityLevel.HIGH
        elif operation in sensitive_operations:
            risk_level = SecurityLevel.MEDIUM

        # Log audit entry
        await log_audit_entry(
            operation="file_operation",
            action=operation,
            context=context,
            resource=file_path,
            result="success" if success else "failure",
            details={
                "file_path": file_path,
                "operation": operation,
                **(details or {}),
            },
            risk_level=risk_level,
        )

        # Log security event for high-risk operations
        if risk_level >= SecurityLevel.MEDIUM or not success:
            severity = (
                SecurityLevel.HIGH
                if risk_level == SecurityLevel.HIGH
                else SecurityLevel.MEDIUM
            )

            await log_security_event(
                f"file_{operation}_{'success' if success else 'failure'}",
                severity,
                context,
                ThreatType.SUSPICIOUS_PATTERN if not success else None,
                {
                    "file_path": file_path,
                    "operation": operation,
                    "risk_level": risk_level.value,
                    **(details or {}),
                },
            )

    async def monitor_network_request(
        self,
        context: SecurityContext,
        url: str,
        method: str,
        success: bool,
        response_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Monitor network requests."""

        if not self.enabled:
            return

        # Determine risk level based on URL and response
        risk_level = SecurityLevel.LOW

        # Higher risk for external URLs or error responses
        if not self._is_internal_url(url):
            risk_level = SecurityLevel.MEDIUM

        if response_code and response_code >= 400:
            risk_level = SecurityLevel.MEDIUM

        # Log audit entry for external requests
        if risk_level >= SecurityLevel.MEDIUM:
            await log_audit_entry(
                operation="network_request",
                action=method,
                context=context,
                resource=url,
                result="success" if success else "failure",
                details={
                    "url": url,
                    "method": method,
                    "response_code": response_code,
                    **(details or {}),
                },
                risk_level=risk_level,
            )

        # Log security event for suspicious requests
        if not success or (response_code and response_code >= 400):
            await log_security_event(
                "suspicious_network_request",
                SecurityLevel.MEDIUM,
                context,
                ThreatType.SUSPICIOUS_PATTERN,
                {
                    "url": url,
                    "method": method,
                    "response_code": response_code,
                    "success": success,
                    **(details or {}),
                },
            )

    def _is_internal_url(self, url: str) -> bool:
        """Check if URL is internal/trusted."""

        internal_domains = [
            "localhost",
            "127.0.0.1",
            "::1",
            "openmanus.local",
        ]

        return any(domain in url.lower() for domain in internal_domains)

    async def monitor_command_execution(
        self,
        context: SecurityContext,
        command: str,
        success: bool,
        exit_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Monitor command execution."""

        if not self.enabled:
            return

        # Determine risk level based on command
        risk_level = self._assess_command_risk(command)

        # Always log command execution
        await log_audit_entry(
            operation="command_execution",
            action="execute",
            context=context,
            resource=command,
            result="success" if success else "failure",
            details={
                "command": command,
                "exit_code": exit_code,
                "success": success,
                **(details or {}),
            },
            risk_level=risk_level,
        )

        # Log security event for high-risk commands or failures
        if risk_level >= SecurityLevel.MEDIUM or not success:
            severity = (
                SecurityLevel.HIGH
                if risk_level == SecurityLevel.HIGH
                else SecurityLevel.MEDIUM
            )

            await log_security_event(
                "command_execution_monitored",
                severity,
                context,
                (
                    ThreatType.COMMAND_INJECTION
                    if risk_level == SecurityLevel.HIGH
                    else None
                ),
                {
                    "command": command,
                    "exit_code": exit_code,
                    "success": success,
                    "risk_level": risk_level.value,
                    **(details or {}),
                },
            )

    def _assess_command_risk(self, command: str) -> SecurityLevel:
        """Assess the risk level of a command."""

        high_risk_commands = [
            "rm",
            "del",
            "format",
            "shutdown",
            "reboot",
            "kill",
            "chmod",
            "chown",
            "sudo",
            "su",
            "passwd",
            "useradd",
            "userdel",
            "groupadd",
            "groupdel",
            "crontab",
            "at",
        ]

        medium_risk_commands = [
            "wget",
            "curl",
            "nc",
            "netcat",
            "telnet",
            "ssh",
            "scp",
            "rsync",
            "git",
            "pip",
            "npm",
            "docker",
        ]

        command_lower = command.lower()

        # Check for high-risk commands
        if any(cmd in command_lower for cmd in high_risk_commands):
            return SecurityLevel.HIGH

        # Check for medium-risk commands
        if any(cmd in command_lower for cmd in medium_risk_commands):
            return SecurityLevel.MEDIUM

        return SecurityLevel.LOW


class SecurityEventProcessor:
    """Processes security events for real-time analysis and response."""

    def __init__(self):
        """Initialize security event processor."""
        self.processing_enabled = True
        self.auto_response_enabled = False  # Disabled by default for safety

        # Event processing rules
        self.processing_rules = {
            "rapid_failed_attempts": self._handle_rapid_failed_attempts,
            "high_risk_command": self._handle_high_risk_command,
            "suspicious_file_access": self._handle_suspicious_file_access,
            "threat_indicator_match": self._handle_threat_indicator_match,
        }

    async def process_event(self, event_data: Dict[str, Any]):
        """Process a security event for automated response."""

        if not self.processing_enabled:
            return

        event_type = event_data.get("event_type", "")

        # Apply processing rules
        for rule_name, handler in self.processing_rules.items():
            if await self._matches_rule(event_data, rule_name):
                try:
                    await handler(event_data)
                except Exception as e:
                    logger.error(
                        f"Error processing security event with rule {rule_name}: {e}"
                    )

    async def _matches_rule(self, event_data: Dict[str, Any], rule_name: str) -> bool:
        """Check if event matches a processing rule."""

        event_type = event_data.get("event_type", "").lower()
        severity = event_data.get("severity", "").lower()

        if rule_name == "rapid_failed_attempts":
            return "rapid_failed_attempts" in event_type
        elif rule_name == "high_risk_command":
            return "command_execution" in event_type and severity in [
                "high",
                "critical",
            ]
        elif rule_name == "suspicious_file_access":
            return "file_" in event_type and severity in ["medium", "high", "critical"]
        elif rule_name == "threat_indicator_match":
            return "threat_indicator" in event_type

        return False

    async def _handle_rapid_failed_attempts(self, event_data: Dict[str, Any]):
        """Handle rapid failed login attempts."""

        source_ip = event_data.get("source_ip")
        user_id = event_data.get("user_id")

        logger.warning(
            "Rapid failed attempts detected - consider blocking",
            {
                "source_ip": source_ip,
                "user_id": user_id,
                "auto_response_enabled": self.auto_response_enabled,
            },
        )

        if self.auto_response_enabled and source_ip:
            # Auto-block IP (would need integration with firewall/security system)
            logger.warning(f"Auto-blocking IP {source_ip} due to rapid failed attempts")

    async def _handle_high_risk_command(self, event_data: Dict[str, Any]):
        """Handle high-risk command execution."""

        command = event_data.get("details", {}).get("command", "")
        user_id = event_data.get("user_id")

        logger.error(
            "High-risk command executed",
            {
                "command": command,
                "user_id": user_id,
                "requires_investigation": True,
            },
        )

    async def _handle_suspicious_file_access(self, event_data: Dict[str, Any]):
        """Handle suspicious file access."""

        file_path = event_data.get("details", {}).get("file_path", "")
        operation = event_data.get("details", {}).get("operation", "")
        user_id = event_data.get("user_id")

        logger.warning(
            "Suspicious file access detected",
            {
                "file_path": file_path,
                "operation": operation,
                "user_id": user_id,
                "requires_review": True,
            },
        )

    async def _handle_threat_indicator_match(self, event_data: Dict[str, Any]):
        """Handle threat indicator match."""

        source_ip = event_data.get("source_ip")
        user_id = event_data.get("user_id")

        logger.critical(
            "Known threat indicator detected",
            {
                "source_ip": source_ip,
                "user_id": user_id,
                "immediate_action_required": True,
            },
        )

        if self.auto_response_enabled:
            # Immediate blocking for known threats
            logger.critical(
                f"Auto-blocking threat source: IP={source_ip}, User={user_id}"
            )


# Global instances
security_middleware = SecurityMonitoringMiddleware()
security_event_processor = SecurityEventProcessor()


# Decorator for monitoring operations
def monitor_security(operation_name: str):
    """Decorator to monitor operations for security events."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract or create security context
            context = kwargs.get("security_context")
            if not context:
                context = SecurityContext(
                    operation=operation_name,
                    timestamp=datetime.now(timezone.utc),
                )

            return await security_middleware.monitor_operation(
                operation_name, context, func, *args, **kwargs
            )

        return wrapper

    return decorator


# Context manager for security monitoring
class SecurityMonitoringContext:
    """Context manager for security monitoring operations."""

    def __init__(
        self,
        operation: str,
        context: SecurityContext,
        auto_log: bool = True,
    ):
        """Initialize security monitoring context."""
        self.operation = operation
        self.context = context
        self.auto_log = auto_log
        self.start_time = None
        self.success = False
        self.error = None

    async def __aenter__(self):
        """Enter security monitoring context."""
        self.start_time = datetime.now(timezone.utc)

        if self.auto_log:
            await log_audit_entry(
                operation=self.operation,
                action="start",
                context=self.context,
                details={"start_time": self.start_time.isoformat()},
            )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit security monitoring context."""
        end_time = datetime.now(timezone.utc)
        self.success = exc_type is None

        if exc_type:
            self.error = str(exc_val)

        if self.auto_log:
            duration = (end_time - self.start_time).total_seconds()

            await log_audit_entry(
                operation=self.operation,
                action="complete",
                context=self.context,
                result="success" if self.success else "failure",
                details={
                    "start_time": self.start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_seconds": duration,
                    "success": self.success,
                    "error": self.error,
                },
                risk_level=(
                    SecurityLevel.HIGH if not self.success else SecurityLevel.LOW
                ),
            )

        # Analyze for anomalies
        anomalies = await analyze_access_attempt(
            self.context, self.success, self.operation
        )

        if anomalies:
            await log_security_event(
                "operation_anomalies_detected",
                SecurityLevel.MEDIUM,
                self.context,
                ThreatType.SUSPICIOUS_PATTERN,
                {
                    "operation": self.operation,
                    "anomalies": [a.value for a in anomalies],
                    "success": self.success,
                    "error": self.error,
                },
            )
