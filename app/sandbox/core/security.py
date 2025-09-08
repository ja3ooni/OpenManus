"""Enhanced security components for sandbox management."""

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from app.logger import define_log_level
from app.security.models import SecurityLevel

logger = define_log_level()


class SecurityViolationType(Enum):
    """Types of security violations."""

    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"
    NETWORK_VIOLATION = "network_violation"
    FILE_ACCESS_VIOLATION = "file_access_violation"
    PROCESS_VIOLATION = "process_violation"
    PERMISSION_VIOLATION = "permission_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


class NetworkPolicy(Enum):
    """Network access policies."""

    NONE = "none"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    FULL = "full"


@dataclass
class ResourceLimits:
    """Resource limits for sandbox containers."""

    memory_mb: int = 512
    cpu_percent: float = 50.0
    disk_mb: int = 1024
    max_processes: int = 50
    max_open_files: int = 1024
    max_network_connections: int = 10
    execution_timeout: int = 300


@dataclass
class SecurityPolicy:
    """Security policy configuration for sandboxes."""

    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    network_policy: NetworkPolicy = NetworkPolicy.NONE
    allowed_commands: Set[str] = field(default_factory=set)
    blocked_commands: Set[str] = field(
        default_factory=lambda: {
            "rm",
            "rmdir",
            "del",
            "format",
            "fdisk",
            "mkfs",
            "shutdown",
            "reboot",
            "halt",
            "poweroff",
            "kill",
            "killall",
            "pkill",
            "chmod",
            "chown",
            "chgrp",
            "mount",
            "umount",
            "sudo",
            "su",
            "passwd",
            "useradd",
            "userdel",
            "usermod",
            "iptables",
            "netstat",
            "ss",
            "lsof",
        }
    )
    allowed_file_extensions: Set[str] = field(
        default_factory=lambda: {
            ".txt",
            ".py",
            ".js",
            ".json",
            ".yaml",
            ".yml",
            ".md",
            ".csv",
        }
    )
    blocked_file_patterns: Set[str] = field(
        default_factory=lambda: {
            "/etc/",
            "/bin/",
            "/sbin/",
            "/usr/bin/",
            "/usr/sbin/",
            "/proc/",
            "/sys/",
            "/dev/",
            "/root/",
            "/home/",
        }
    )
    max_file_size_mb: int = 10
    enable_monitoring: bool = True
    violation_threshold: int = 5
    auto_terminate_on_violation: bool = True


@dataclass
class SecurityViolation:
    """Security violation record."""

    violation_id: str
    timestamp: datetime
    violation_type: SecurityViolationType
    severity: SecurityLevel
    container_id: str
    details: Dict[str, Any]
    action_taken: str


@dataclass
class SecurityMetrics:
    """Security monitoring metrics."""

    container_id: str
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_mb: float
    disk_usage_mb: float
    network_connections: int
    process_count: int
    open_files: int
    violations_count: int


class SandboxSecurityManager:
    """Enhanced security manager for Docker sandboxes."""

    def __init__(self, policy: Optional[SecurityPolicy] = None):
        """Initialize security manager with policy."""
        self.policy = policy or SecurityPolicy()
        self.violations: List[SecurityViolation] = []
        self.metrics_history: List[SecurityMetrics] = []
        self.monitored_containers: Dict[str, Any] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def apply_security_policy(self, container) -> None:
        """Apply security policy to a container."""
        try:
            await self._apply_resource_limits(container)
            await self._apply_network_policy(container)
            await self._apply_filesystem_restrictions(container)

            if self.policy.enable_monitoring:
                await self.start_monitoring(container)

            logger.info(f"Security policy applied to container {container.id[:12]}")

        except Exception as e:
            logger.error(
                f"Failed to apply security policy to container {container.id[:12]}: {e}"
            )
            raise

    async def _apply_resource_limits(self, container) -> None:
        """Apply resource limits to container."""
        try:
            limits = self.policy.resource_limits

            container.update(
                mem_limit=f"{limits.memory_mb}m",
                cpu_period=100000,
                cpu_quota=int(100000 * (limits.cpu_percent / 100)),
                pids_limit=limits.max_processes,
            )

            container.exec_run(
                f"sh -c 'ulimit -n {limits.max_open_files}'", privileged=False
            )

            logger.debug(
                f"Resource limits applied: Memory={limits.memory_mb}MB, "
                f"CPU={limits.cpu_percent}%, Processes={limits.max_processes}"
            )

        except Exception as e:
            logger.error(f"Failed to apply resource limits: {e}")
            raise

    async def _apply_network_policy(self, container) -> None:
        """Apply network policy to container."""
        try:
            policy = self.policy.network_policy

            if policy == NetworkPolicy.NONE:
                for network in container.attrs.get("NetworkSettings", {}).get(
                    "Networks", {}
                ):
                    try:
                        network_obj = container.client.networks.get(network)
                        network_obj.disconnect(container)
                    except Exception as e:
                        logger.warning(
                            f"Failed to disconnect from network {network}: {e}"
                        )

            elif policy == NetworkPolicy.RESTRICTED:
                await self._apply_network_restrictions(container)

            logger.debug(f"Network policy applied: {policy.value}")

        except Exception as e:
            logger.error(f"Failed to apply network policy: {e}")
            raise

    async def _apply_network_restrictions(self, container) -> None:
        """Apply network restrictions using iptables."""
        try:
            restricted_rules = [
                "iptables -A OUTPUT -d 127.0.0.0/8 -j DROP",
                "iptables -A OUTPUT -d 10.0.0.0/8 -j DROP",
                "iptables -A OUTPUT -d 172.16.0.0/12 -j DROP",
                "iptables -A OUTPUT -d 192.168.0.0/16 -j DROP",
                "iptables -A OUTPUT -d 169.254.0.0/16 -j DROP",
                "iptables -A OUTPUT -p udp --dport 53 -j ACCEPT",
                "iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT",
                "iptables -A OUTPUT -p tcp --dport 80 -j ACCEPT",
                "iptables -A OUTPUT -p tcp --dport 443 -j ACCEPT",
            ]

            for rule in restricted_rules:
                try:
                    container.exec_run(rule, privileged=True)
                except Exception as e:
                    logger.warning(f"Failed to apply iptables rule '{rule}': {e}")

        except Exception as e:
            logger.error(f"Failed to apply network restrictions: {e}")

    async def _apply_filesystem_restrictions(self, container) -> None:
        """Apply filesystem access restrictions."""
        try:
            sensitive_dirs = ["/etc", "/bin", "/sbin", "/usr/bin", "/usr/sbin"]

            for dir_path in sensitive_dirs:
                try:
                    container.exec_run(f"chmod -R a-w {dir_path}", privileged=True)
                except Exception as e:
                    logger.warning(f"Failed to restrict directory {dir_path}: {e}")

            container.exec_run("mkdir -p /sandbox/work", privileged=True)
            container.exec_run("chown -R 1000:1000 /sandbox", privileged=True)

            logger.debug("Filesystem restrictions applied")

        except Exception as e:
            logger.error(f"Failed to apply filesystem restrictions: {e}")

    async def start_monitoring(self, container) -> None:
        """Start security monitoring for a container."""
        async with self._lock:
            if container.id in self.monitoring_tasks:
                return

            self.monitored_containers[container.id] = container
            task = asyncio.create_task(self._monitor_container(container))
            self.monitoring_tasks[container.id] = task

            logger.info(
                f"Started security monitoring for container {container.id[:12]}"
            )

    async def stop_monitoring(self, container_id: str) -> None:
        """Stop security monitoring for a container."""
        async with self._lock:
            if container_id in self.monitoring_tasks:
                self.monitoring_tasks[container_id].cancel()
                del self.monitoring_tasks[container_id]

            if container_id in self.monitored_containers:
                del self.monitored_containers[container_id]

            logger.info(
                f"Stopped security monitoring for container {container_id[:12]}"
            )

    async def _monitor_container(self, container) -> None:
        """Monitor container for security violations."""
        violation_count = 0

        try:
            while True:
                try:
                    metrics = await self._collect_metrics(container)
                    self.metrics_history.append(metrics)

                    violations = await self._check_violations(container, metrics)

                    if violations:
                        violation_count += len(violations)
                        self.violations.extend(violations)

                        for violation in violations:
                            logger.warning(
                                f"Security violation detected: {violation.violation_type.value} "
                                f"in container {container.id[:12]}"
                            )

                        if (
                            violation_count >= self.policy.violation_threshold
                            and self.policy.auto_terminate_on_violation
                        ):
                            await self._terminate_container(
                                container, "Security violation threshold exceeded"
                            )
                            break

                    if len(self.metrics_history) > 1000:
                        self.metrics_history = self.metrics_history[-500:]

                    await asyncio.sleep(5)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error monitoring container {container.id[:12]}: {e}")
                    await asyncio.sleep(10)

        except Exception as e:
            logger.error(
                f"Monitoring task failed for container {container.id[:12]}: {e}"
            )

    async def _collect_metrics(self, container) -> SecurityMetrics:
        """Collect security metrics from container."""
        try:
            stats = container.stats(stream=False)

            cpu_usage = 0.0
            if "cpu_stats" in stats and "precpu_stats" in stats:
                cpu_delta = (
                    stats["cpu_stats"]["cpu_usage"]["total_usage"]
                    - stats["precpu_stats"]["cpu_usage"]["total_usage"]
                )
                system_delta = (
                    stats["cpu_stats"]["system_cpu_usage"]
                    - stats["precpu_stats"]["system_cpu_usage"]
                )
                if system_delta > 0:
                    cpu_usage = (cpu_delta / system_delta) * 100.0

            memory_usage = 0.0
            if "memory_stats" in stats:
                memory_usage = stats["memory_stats"].get("usage", 0) / (1024 * 1024)

            process_count = 0
            network_connections = 0
            open_files = 0

            try:
                result = container.exec_run("ps aux | wc -l")
                if result.exit_code == 0:
                    process_count = int(result.output.decode().strip()) - 1

                result = container.exec_run("netstat -an | wc -l")
                if result.exit_code == 0:
                    network_connections = int(result.output.decode().strip())

                result = container.exec_run("lsof | wc -l")
                if result.exit_code == 0:
                    open_files = int(result.output.decode().strip())

            except Exception as e:
                logger.debug(f"Failed to collect detailed metrics: {e}")

            return SecurityMetrics(
                container_id=container.id,
                timestamp=datetime.utcnow(),
                cpu_usage_percent=cpu_usage,
                memory_usage_mb=memory_usage,
                disk_usage_mb=0.0,
                network_connections=network_connections,
                process_count=process_count,
                open_files=open_files,
                violations_count=len(
                    [v for v in self.violations if v.container_id == container.id]
                ),
            )

        except Exception as e:
            logger.error(
                f"Failed to collect metrics for container {container.id[:12]}: {e}"
            )
            return SecurityMetrics(
                container_id=container.id,
                timestamp=datetime.utcnow(),
                cpu_usage_percent=0.0,
                memory_usage_mb=0.0,
                disk_usage_mb=0.0,
                network_connections=0,
                process_count=0,
                open_files=0,
                violations_count=0,
            )

    async def _check_violations(
        self, container, metrics: SecurityMetrics
    ) -> List[SecurityViolation]:
        """Check for security violations based on metrics."""
        violations = []
        limits = self.policy.resource_limits

        try:
            if metrics.cpu_usage_percent > limits.cpu_percent * 1.2:
                violations.append(
                    SecurityViolation(
                        violation_id=f"cpu_{container.id}_{int(time.time())}",
                        timestamp=datetime.utcnow(),
                        violation_type=SecurityViolationType.RESOURCE_LIMIT_EXCEEDED,
                        severity=SecurityLevel.MEDIUM,
                        container_id=container.id,
                        details={
                            "resource": "cpu",
                            "usage": metrics.cpu_usage_percent,
                            "limit": limits.cpu_percent,
                        },
                        action_taken="logged",
                    )
                )

            if metrics.memory_usage_mb > limits.memory_mb * 1.1:
                violations.append(
                    SecurityViolation(
                        violation_id=f"memory_{container.id}_{int(time.time())}",
                        timestamp=datetime.utcnow(),
                        violation_type=SecurityViolationType.RESOURCE_LIMIT_EXCEEDED,
                        severity=SecurityLevel.HIGH,
                        container_id=container.id,
                        details={
                            "resource": "memory",
                            "usage": metrics.memory_usage_mb,
                            "limit": limits.memory_mb,
                        },
                        action_taken="logged",
                    )
                )

            if metrics.process_count > limits.max_processes:
                violations.append(
                    SecurityViolation(
                        violation_id=f"processes_{container.id}_{int(time.time())}",
                        timestamp=datetime.utcnow(),
                        violation_type=SecurityViolationType.RESOURCE_LIMIT_EXCEEDED,
                        severity=SecurityLevel.MEDIUM,
                        container_id=container.id,
                        details={
                            "resource": "processes",
                            "count": metrics.process_count,
                            "limit": limits.max_processes,
                        },
                        action_taken="logged",
                    )
                )

            if metrics.network_connections > limits.max_network_connections:
                violations.append(
                    SecurityViolation(
                        violation_id=f"network_{container.id}_{int(time.time())}",
                        timestamp=datetime.utcnow(),
                        violation_type=SecurityViolationType.NETWORK_VIOLATION,
                        severity=SecurityLevel.HIGH,
                        container_id=container.id,
                        details={
                            "connections": metrics.network_connections,
                            "limit": limits.max_network_connections,
                        },
                        action_taken="logged",
                    )
                )

            suspicious_violations = await self._check_suspicious_processes(container)
            violations.extend(suspicious_violations)

        except Exception as e:
            logger.error(
                f"Error checking violations for container {container.id[:12]}: {e}"
            )

        return violations

    async def _check_suspicious_processes(self, container) -> List[SecurityViolation]:
        """Check for suspicious processes running in container."""
        violations = []

        try:
            result = container.exec_run("ps aux")
            if result.exit_code != 0:
                return violations

            processes = result.output.decode().strip().split("\n")[1:]

            for process_line in processes:
                if not process_line.strip():
                    continue

                parts = process_line.split()
                if len(parts) < 11:
                    continue

                command = " ".join(parts[10:])

                for blocked_cmd in self.policy.blocked_commands:
                    if blocked_cmd in command.lower():
                        violations.append(
                            SecurityViolation(
                                violation_id=f"suspicious_process_{container.id}_{int(time.time())}",
                                timestamp=datetime.utcnow(),
                                violation_type=SecurityViolationType.PROCESS_VIOLATION,
                                severity=SecurityLevel.HIGH,
                                container_id=container.id,
                                details={
                                    "blocked_command": blocked_cmd,
                                    "full_command": command,
                                    "process_line": process_line,
                                },
                                action_taken="logged",
                            )
                        )
                        break

        except Exception as e:
            logger.error(f"Error checking suspicious processes: {e}")

        return violations

    async def _terminate_container(self, container, reason: str) -> None:
        """Terminate container due to security violation."""
        try:
            logger.warning(
                f"Terminating container {container.id[:12]} due to: {reason}"
            )

            await self.stop_monitoring(container.id)

            container.stop(timeout=5)
            container.remove(force=True)

            violation = SecurityViolation(
                violation_id=f"termination_{container.id}_{int(time.time())}",
                timestamp=datetime.utcnow(),
                violation_type=SecurityViolationType.SUSPICIOUS_ACTIVITY,
                severity=SecurityLevel.CRITICAL,
                container_id=container.id,
                details={"reason": reason},
                action_taken="container_terminated",
            )
            self.violations.append(violation)

        except Exception as e:
            logger.error(f"Failed to terminate container {container.id[:12]}: {e}")

    async def validate_file_access(self, container_id: str, file_path: str) -> bool:
        """Validate if file access is allowed by security policy."""
        try:
            if self.policy.allowed_file_extensions:
                _, ext = os.path.splitext(file_path.lower())
                if ext and ext not in self.policy.allowed_file_extensions:
                    logger.warning(
                        f"File access denied - invalid extension: {file_path}"
                    )
                    return False

            for pattern in self.policy.blocked_file_patterns:
                if pattern in file_path:
                    logger.warning(
                        f"File access denied - blocked pattern '{pattern}': {file_path}"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating file access: {e}")
            return False

    async def validate_command_execution(self, container_id: str, command: str) -> bool:
        """Validate if command execution is allowed by security policy."""
        try:
            command_lower = command.lower().strip()

            if self.policy.allowed_commands:
                cmd_parts = command_lower.split()
                if cmd_parts and cmd_parts[0] not in self.policy.allowed_commands:
                    logger.warning(
                        f"Command execution denied - not in allowed list: {command}"
                    )
                    return False

            for blocked_cmd in self.policy.blocked_commands:
                if blocked_cmd in command_lower:
                    logger.warning(
                        f"Command execution denied - blocked command '{blocked_cmd}': {command}"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating command execution: {e}")
            return False

    async def get_security_report(
        self, container_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate security report for container(s)."""
        try:
            violations = self.violations
            metrics = self.metrics_history

            if container_id:
                violations = [v for v in violations if v.container_id == container_id]
                metrics = [m for m in metrics if m.container_id == container_id]

            violation_counts = {}
            for violation in violations:
                vtype = violation.violation_type.value
                violation_counts[vtype] = violation_counts.get(vtype, 0) + 1

            recent_violations = [
                v
                for v in violations
                if v.timestamp > datetime.utcnow() - timedelta(hours=24)
            ]

            return {
                "report_timestamp": datetime.utcnow().isoformat(),
                "container_id": container_id,
                "total_violations": len(violations),
                "recent_violations_24h": len(recent_violations),
                "violation_types": violation_counts,
                "monitored_containers": len(self.monitored_containers),
                "policy_summary": {
                    "network_policy": self.policy.network_policy.value,
                    "resource_limits": {
                        "memory_mb": self.policy.resource_limits.memory_mb,
                        "cpu_percent": self.policy.resource_limits.cpu_percent,
                        "max_processes": self.policy.resource_limits.max_processes,
                    },
                    "monitoring_enabled": self.policy.enable_monitoring,
                },
                "recent_metrics": metrics[-10:] if metrics else [],
            }

        except Exception as e:
            logger.error(f"Error generating security report: {e}")
            return {"error": str(e)}

    async def cleanup(self) -> None:
        """Clean up security manager resources."""
        try:
            for container_id in list(self.monitoring_tasks.keys()):
                await self.stop_monitoring(container_id)

            self.violations.clear()
            self.metrics_history.clear()
            self.monitored_containers.clear()

            logger.info("Security manager cleanup completed")

        except Exception as e:
            logger.error(f"Error during security manager cleanup: {e}")
