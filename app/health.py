"""
Health check system for OpenManus.

This module provides comprehensive health checks for monitoring
and load balancer integration in containerized environments.
"""

import asyncio
import os
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import psutil
from pydantic import BaseModel

from app.config import config
from app.logger import logger
from app.monitoring import metrics_collector, system_monitor


class HealthStatus(Enum):
    """Health check status levels"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class HealthCheckResult(BaseModel):
    """Result of a health check"""

    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    duration_ms: float
    details: Optional[Dict[str, Any]] = None


class HealthChecker:
    """Comprehensive health check system"""

    def __init__(self):
        self.start_time = time.time()
        self.health_checks: Dict[str, callable] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}

        # Register default health checks
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default health checks"""
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("configuration", self._check_configuration)
        self.register_check("dependencies", self._check_dependencies)
        self.register_check("storage", self._check_storage)
        self.register_check("network", self._check_network)

    def register_check(self, name: str, check_function: callable):
        """Register a health check function"""
        self.health_checks[name] = check_function
        logger.debug(f"Registered health check: {name}")

    async def run_health_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check"""
        if name not in self.health_checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Health check '{name}' not found",
                timestamp=datetime.now(timezone.utc),
                duration_ms=0.0,
            )

        start_time = time.time()

        try:
            check_function = self.health_checks[name]

            if asyncio.iscoroutinefunction(check_function):
                result = await check_function()
            else:
                result = check_function()

            duration_ms = (time.time() - start_time) * 1000

            if isinstance(result, HealthCheckResult):
                result.duration_ms = duration_ms
                self.last_results[name] = result
                return result
            else:
                # Convert simple result to HealthCheckResult
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = f"Check {name} {'passed' if result else 'failed'}"

                result = HealthCheckResult(
                    name=name,
                    status=status,
                    message=message,
                    timestamp=datetime.now(timezone.utc),
                    duration_ms=duration_ms,
                )

                self.last_results[name] = result
                return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            result = HealthCheckResult(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.now(timezone.utc),
                duration_ms=duration_ms,
                details={"error": str(e), "error_type": type(e).__name__},
            )

            self.last_results[name] = result
            logger.error(f"Health check {name} failed: {e}")
            return result

    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}

        # Run checks concurrently
        tasks = [self.run_health_check(name) for name in self.health_checks.keys()]

        check_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, (name, result) in enumerate(
            zip(self.health_checks.keys(), check_results)
        ):
            if isinstance(result, Exception):
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check exception: {str(result)}",
                    timestamp=datetime.now(timezone.utc),
                    duration_ms=0.0,
                )
            else:
                results[name] = result

        return results

    def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource usage"""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Determine status based on resource usage
            status = HealthStatus.HEALTHY
            issues = []

            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
                issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif cpu_percent > 80:
                status = HealthStatus.DEGRADED
                issues.append(f"CPU usage high: {cpu_percent:.1f}%")

            if memory.percent > 95:
                status = HealthStatus.CRITICAL
                issues.append(f"Memory usage critical: {memory.percent:.1f}%")
            elif memory.percent > 85:
                status = HealthStatus.DEGRADED
                issues.append(f"Memory usage high: {memory.percent:.1f}%")

            if disk.percent > 95:
                status = HealthStatus.CRITICAL
                issues.append(f"Disk usage critical: {disk.percent:.1f}%")
            elif disk.percent > 85:
                status = HealthStatus.DEGRADED
                issues.append(f"Disk usage high: {disk.percent:.1f}%")

            message = "System resources healthy" if not issues else "; ".join(issues)

            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=message,
                timestamp=datetime.now(timezone.utc),
                duration_ms=0.0,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3),
                },
            )

        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {str(e)}",
                timestamp=datetime.now(timezone.utc),
                duration_ms=0.0,
            )

    def _check_configuration(self) -> HealthCheckResult:
        """Check configuration validity"""
        try:
            # Basic configuration checks
            issues = []

            # Check LLM configuration
            if not config.llm or not config.llm.get("default"):
                issues.append("No default LLM configuration")
            else:
                default_llm = config.llm["default"]
                if not default_llm.api_key or default_llm.api_key == "YOUR_API_KEY":
                    issues.append("LLM API key not configured")
                if not default_llm.model:
                    issues.append("LLM model not specified")

            # Check workspace directory
            if not config.workspace_root.exists():
                issues.append("Workspace directory does not exist")
            elif not os.access(config.workspace_root, os.W_OK):
                issues.append("Workspace directory not writable")

            status = HealthStatus.HEALTHY if not issues else HealthStatus.DEGRADED
            message = "Configuration valid" if not issues else "; ".join(issues)

            return HealthCheckResult(
                name="configuration",
                status=status,
                message=message,
                timestamp=datetime.now(timezone.utc),
                duration_ms=0.0,
                details={
                    "config_file_exists": (
                        config.root_path / "config" / "config.toml"
                    ).exists(),
                    "workspace_writable": os.access(config.workspace_root, os.W_OK),
                    "llm_configured": bool(config.llm and config.llm.get("default")),
                },
            )

        except Exception as e:
            return HealthCheckResult(
                name="configuration",
                status=HealthStatus.CRITICAL,
                message=f"Configuration check failed: {str(e)}",
                timestamp=datetime.now(timezone.utc),
                duration_ms=0.0,
            )

    async def _check_dependencies(self) -> HealthCheckResult:
        """Check external dependencies"""
        try:
            issues = []
            details = {}

            # Check LLM API connectivity
            try:
                # This is a basic connectivity check - in production you might want
                # to make an actual API call with a minimal request
                import aiohttp

                default_llm = config.llm.get("default", {})
                base_url = default_llm.get("base_url")

                if base_url:
                    async with aiohttp.ClientSession() as session:
                        try:
                            async with session.get(
                                base_url, timeout=aiohttp.ClientTimeout(total=5)
                            ) as response:
                                details["llm_api_reachable"] = response.status < 500
                        except Exception:
                            details["llm_api_reachable"] = False
                            issues.append("LLM API not reachable")
                else:
                    details["llm_api_reachable"] = False
                    issues.append("LLM API URL not configured")

            except ImportError:
                issues.append("aiohttp not available for dependency checks")

            # Check MCP servers if configured
            if config.mcp_config and config.mcp_config.servers:
                details["mcp_servers_configured"] = len(config.mcp_config.servers)
            else:
                details["mcp_servers_configured"] = 0

            status = HealthStatus.HEALTHY if not issues else HealthStatus.DEGRADED
            message = "Dependencies healthy" if not issues else "; ".join(issues)

            return HealthCheckResult(
                name="dependencies",
                status=status,
                message=message,
                timestamp=datetime.now(timezone.utc),
                duration_ms=0.0,
                details=details,
            )

        except Exception as e:
            return HealthCheckResult(
                name="dependencies",
                status=HealthStatus.CRITICAL,
                message=f"Dependency check failed: {str(e)}",
                timestamp=datetime.now(timezone.utc),
                duration_ms=0.0,
            )

    def _check_storage(self) -> HealthCheckResult:
        """Check storage and file system health"""
        try:
            issues = []
            details = {}

            # Check workspace directory
            workspace_path = config.workspace_root
            if workspace_path.exists():
                details["workspace_exists"] = True
                details["workspace_writable"] = os.access(workspace_path, os.W_OK)

                if not details["workspace_writable"]:
                    issues.append("Workspace directory not writable")
            else:
                details["workspace_exists"] = False
                issues.append("Workspace directory does not exist")

            # Check log directory
            log_dir = config.root_path / "logs"
            if log_dir.exists():
                details["logs_writable"] = os.access(log_dir, os.W_OK)
                if not details["logs_writable"]:
                    issues.append("Log directory not writable")
            else:
                try:
                    log_dir.mkdir(exist_ok=True)
                    details["logs_writable"] = True
                except Exception:
                    details["logs_writable"] = False
                    issues.append("Cannot create log directory")

            # Check available disk space
            disk_usage = psutil.disk_usage(str(config.root_path))
            free_gb = disk_usage.free / (1024**3)
            details["free_space_gb"] = free_gb

            if free_gb < 0.1:  # Less than 100MB
                issues.append(f"Very low disk space: {free_gb:.1f}GB")
            elif free_gb < 1.0:  # Less than 1GB
                issues.append(f"Low disk space: {free_gb:.1f}GB")

            status = HealthStatus.HEALTHY if not issues else HealthStatus.DEGRADED
            message = "Storage healthy" if not issues else "; ".join(issues)

            return HealthCheckResult(
                name="storage",
                status=status,
                message=message,
                timestamp=datetime.now(timezone.utc),
                duration_ms=0.0,
                details=details,
            )

        except Exception as e:
            return HealthCheckResult(
                name="storage",
                status=HealthStatus.CRITICAL,
                message=f"Storage check failed: {str(e)}",
                timestamp=datetime.now(timezone.utc),
                duration_ms=0.0,
            )

    async def _check_network(self) -> HealthCheckResult:
        """Check network connectivity"""
        try:
            import aiohttp

            issues = []
            details = {}

            # Test basic internet connectivity
            test_urls = [
                "https://httpbin.org/status/200",
                "https://www.google.com",
            ]

            reachable_count = 0

            async with aiohttp.ClientSession() as session:
                for url in test_urls:
                    try:
                        async with session.get(
                            url, timeout=aiohttp.ClientTimeout(total=5)
                        ) as response:
                            if response.status == 200:
                                reachable_count += 1
                    except Exception:
                        pass

            details["internet_connectivity"] = reachable_count > 0
            details["reachable_urls"] = reachable_count
            details["total_test_urls"] = len(test_urls)

            if reachable_count == 0:
                issues.append("No internet connectivity")
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            message = "Network healthy" if not issues else "; ".join(issues)

            return HealthCheckResult(
                name="network",
                status=status,
                message=message,
                timestamp=datetime.now(timezone.utc),
                duration_ms=0.0,
                details=details,
            )

        except ImportError:
            return HealthCheckResult(
                name="network",
                status=HealthStatus.DEGRADED,
                message="Network check unavailable (aiohttp not installed)",
                timestamp=datetime.now(timezone.utc),
                duration_ms=0.0,
            )
        except Exception as e:
            return HealthCheckResult(
                name="network",
                status=HealthStatus.CRITICAL,
                message=f"Network check failed: {str(e)}",
                timestamp=datetime.now(timezone.utc),
                duration_ms=0.0,
            )

    def get_overall_health(self) -> HealthCheckResult:
        """Get overall system health based on all checks"""
        if not self.last_results:
            return HealthCheckResult(
                name="overall",
                status=HealthStatus.UNHEALTHY,
                message="No health checks have been run",
                timestamp=datetime.now(timezone.utc),
                duration_ms=0.0,
            )

        # Determine overall status
        statuses = [result.status for result in self.last_results.values()]

        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        # Count issues
        critical_count = statuses.count(HealthStatus.CRITICAL)
        unhealthy_count = statuses.count(HealthStatus.UNHEALTHY)
        degraded_count = statuses.count(HealthStatus.DEGRADED)
        healthy_count = statuses.count(HealthStatus.HEALTHY)

        # Create message
        if overall_status == HealthStatus.HEALTHY:
            message = f"All {len(statuses)} health checks passed"
        else:
            issues = []
            if critical_count > 0:
                issues.append(f"{critical_count} critical")
            if unhealthy_count > 0:
                issues.append(f"{unhealthy_count} unhealthy")
            if degraded_count > 0:
                issues.append(f"{degraded_count} degraded")

            message = f"Health issues: {', '.join(issues)} ({healthy_count} healthy)"

        return HealthCheckResult(
            name="overall",
            status=overall_status,
            message=message,
            timestamp=datetime.now(timezone.utc),
            duration_ms=0.0,
            details={
                "total_checks": len(statuses),
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "critical": critical_count,
                "uptime_seconds": time.time() - self.start_time,
            },
        )

    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary"""
        overall_health = self.get_overall_health()

        return {
            "status": overall_health.status.value,
            "message": overall_health.message,
            "timestamp": overall_health.timestamp.isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "duration_ms": result.duration_ms,
                    "timestamp": result.timestamp.isoformat(),
                }
                for name, result in self.last_results.items()
            },
            "overall": {
                "status": overall_health.status.value,
                "details": overall_health.details,
            },
        }


# Global health checker instance
health_checker = HealthChecker()
