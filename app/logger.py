import asyncio
import json
import os
import sys
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
from uuid import uuid4

from loguru import logger as _logger

from app.config import PROJECT_ROOT

# Context variables for correlation tracking
correlation_id_var: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
agent_id_var: ContextVar[Optional[str]] = ContextVar("agent_id", default=None)
operation_var: ContextVar[Optional[str]] = ContextVar("operation", default=None)

_print_level = "INFO"
_sensitive_fields = {
    "password",
    "token",
    "key",
    "secret",
    "credential",
    "auth",
    "api_key",
    "access_token",
    "refresh_token",
    "private_key",
}


class StructuredLogger:
    """Enhanced logger with structured logging, correlation IDs, and monitoring integration"""

    def __init__(self, logger_instance):
        self._logger = logger_instance
        self.metrics = PerformanceMetrics()
        self.log_counts = {
            "debug": 0,
            "info": 0,
            "warning": 0,
            "error": 0,
            "critical": 0,
        }
        self.error_patterns = {}  # Track error patterns for alerting
        self.alert_thresholds = {
            "error_rate_per_minute": 10,
            "critical_errors_per_hour": 5,
            "slow_operations_per_minute": 20,
        }
        self.alert_history = []  # Track recent alerts to prevent spam

    def _get_context(self) -> Dict[str, Any]:
        """Get current logging context with enhanced information"""
        return {
            "correlation_id": correlation_id_var.get(),
            "request_id": request_id_var.get(),
            "agent_id": agent_id_var.get(),
            "operation": operation_var.get(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "process_id": os.getpid(),
            "thread_id": os.getpid(),  # In Python, thread ID is more complex, using PID for now
            "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
        }

    def _sanitize_data(self, data: Any) -> Any:
        """Remove sensitive information from log data with enhanced patterns"""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                key_lower = key.lower()
                if any(sensitive in key_lower for sensitive in _sensitive_fields):
                    sanitized[key] = "[REDACTED]"
                elif key_lower in ["url", "uri"] and isinstance(value, str):
                    # Sanitize URLs that might contain sensitive info
                    sanitized[key] = self._sanitize_url(value)
                else:
                    sanitized[key] = self._sanitize_data(value)
            return sanitized
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        elif isinstance(data, str):
            # Check for potential sensitive patterns in strings
            if self._contains_sensitive_pattern(data):
                return "[REDACTED]"
            elif len(data) > 1000:  # Increased limit for better debugging
                return data[:997] + "..."
        return data

    def _sanitize_url(self, url: str) -> str:
        """Sanitize URLs to remove sensitive query parameters"""
        try:
            from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

            parsed = urlparse(url)
            if parsed.query:
                query_params = parse_qs(parsed.query)
                sanitized_params = {}
                for key, values in query_params.items():
                    if any(sensitive in key.lower() for sensitive in _sensitive_fields):
                        sanitized_params[key] = ["[REDACTED]"]
                    else:
                        sanitized_params[key] = values
                sanitized_query = urlencode(sanitized_params, doseq=True)
                return urlunparse(parsed._replace(query=sanitized_query))
        except Exception:
            pass
        return url

    def _contains_sensitive_pattern(self, text: str) -> bool:
        """Check if text contains sensitive patterns like API keys, tokens, etc."""
        import re

        # Common patterns for sensitive data
        patterns = [
            r"[A-Za-z0-9]{32,}",  # Long alphanumeric strings (potential tokens)
            r"sk-[A-Za-z0-9]{32,}",  # OpenAI API keys
            r"Bearer [A-Za-z0-9\-._~+/]+=*",  # Bearer tokens
            r"Basic [A-Za-z0-9+/]+=*",  # Basic auth
        ]

        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False

    def _format_message(
        self, message: str, extra: Optional[Dict[str, Any]] = None, level: str = "info"
    ) -> Dict[str, Any]:
        """Format log message with context and extra data"""
        log_data = {"message": message, "level": level, "context": self._get_context()}

        if extra:
            log_data["extra"] = self._sanitize_data(extra)

        return log_data

    def _increment_log_count(self, level: str):
        """Increment log count for metrics"""
        self.log_counts[level] = self.log_counts.get(level, 0) + 1

    def _track_error_pattern(self, message: str):
        """Track error patterns for alerting"""
        # Simple error pattern tracking
        error_key = message[:100]  # Use first 100 chars as pattern key
        self.error_patterns[error_key] = self.error_patterns.get(error_key, 0) + 1

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message with structured data"""
        self._increment_log_count("debug")
        log_data = self._format_message(message, extra, "debug")
        self._logger.debug(json.dumps(log_data, default=str))

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message with structured data"""
        self._increment_log_count("info")
        log_data = self._format_message(message, extra, "info")
        self._logger.info(json.dumps(log_data, default=str))

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message with structured data"""
        self._increment_log_count("warning")
        log_data = self._format_message(message, extra, "warning")
        self._logger.warning(json.dumps(log_data, default=str))

    def error(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
    ):
        """Log error message with structured data"""
        self._increment_log_count("error")
        self._track_error_pattern(message)
        log_data = self._format_message(message, extra, "error")

        if exc_info:
            import traceback

            log_data["traceback"] = traceback.format_exc()
            self._logger.exception(json.dumps(log_data, default=str))
        else:
            self._logger.error(json.dumps(log_data, default=str))

        # Check alert conditions after error logging
        self.check_alert_conditions()

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log critical message with structured data"""
        self._increment_log_count("critical")
        self._track_error_pattern(message)
        log_data = self._format_message(message, extra, "critical")
        self._logger.critical(json.dumps(log_data, default=str))

        # Check alert conditions after critical logging
        self.check_alert_conditions()

    def exception(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log exception with structured data"""
        self.error(message, extra, exc_info=True)

    def log_performance(
        self, operation: str, duration: float, extra: Optional[Dict[str, Any]] = None
    ):
        """Log performance metrics with enhanced data"""
        perf_data = {
            "operation": operation,
            "duration_ms": round(duration * 1000, 2),
            "duration_seconds": duration,
            "performance_metric": True,
            "slow_operation": duration > 5.0,  # Flag slow operations
        }
        if extra:
            perf_data.update(extra)

        # Log as warning if operation is slow
        if duration > 5.0:
            self.warning(f"Slow operation: {operation} took {duration:.3f}s", perf_data)
        else:
            self.info(
                f"Performance: {operation} completed in {duration:.3f}s", perf_data
            )

        self.metrics.record_operation(operation, duration)

    def log_security_event(
        self, event_type: str, details: Dict[str, Any], severity: str = "warning"
    ):
        """Log security-related events with severity levels"""
        security_data = {
            "security_event": True,
            "event_type": event_type,
            "severity": severity,
            "details": self._sanitize_data(details),
            "requires_attention": severity in ["high", "critical"],
        }

        if severity == "critical":
            self.critical(f"Critical security event: {event_type}", security_data)
        elif severity == "high":
            self.error(f"High severity security event: {event_type}", security_data)
        else:
            self.warning(f"Security event: {event_type}", security_data)

    def log_business_event(self, event_type: str, details: Dict[str, Any]):
        """Log business/application events for analytics"""
        business_data = {
            "business_event": True,
            "event_type": event_type,
            "details": self._sanitize_data(details),
        }
        self.info(f"Business event: {event_type}", business_data)

    def check_alert_conditions(self):
        """Check if any alert conditions are met and trigger alerts if necessary"""
        current_time = time.time()

        # Check error rate (errors per minute)
        recent_errors = sum(
            1 for pattern, count in self.error_patterns.items() if count > 0
        )  # Simplified check

        # Check for critical error threshold
        critical_count = self.log_counts.get("critical", 0)

        # Check for slow operations
        slow_ops = sum(
            1
            for op_times in self.metrics.operation_times.values()
            for duration in op_times[-10:]
            if duration > 5.0
        )

        alerts = []

        if recent_errors > self.alert_thresholds["error_rate_per_minute"]:
            alerts.append(
                {
                    "type": "high_error_rate",
                    "severity": "warning",
                    "message": f"High error rate detected: {recent_errors} error patterns",
                    "timestamp": current_time,
                }
            )

        if critical_count > self.alert_thresholds["critical_errors_per_hour"]:
            alerts.append(
                {
                    "type": "critical_errors",
                    "severity": "critical",
                    "message": f"Critical error threshold exceeded: {critical_count} critical errors",
                    "timestamp": current_time,
                }
            )

        if slow_ops > self.alert_thresholds["slow_operations_per_minute"]:
            alerts.append(
                {
                    "type": "slow_operations",
                    "severity": "warning",
                    "message": f"High number of slow operations: {slow_ops} operations > 5s",
                    "timestamp": current_time,
                }
            )

        # Store alerts and log them
        for alert in alerts:
            # Prevent alert spam - only alert if not alerted in last 5 minutes
            recent_alerts = [
                a
                for a in self.alert_history
                if current_time - a.get("timestamp", 0) < 300
                and a.get("type") == alert["type"]
            ]

            if not recent_alerts:
                self.alert_history.append(alert)
                self.warning(
                    f"ALERT: {alert['message']}",
                    {
                        "alert_type": alert["type"],
                        "severity": alert["severity"],
                        "alert_data": alert,
                    },
                )

        # Clean old alerts (keep last 100)
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]

    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics with enhanced metrics"""
        total_logs = sum(self.log_counts.values())
        error_rate = (
            (self.log_counts.get("error", 0) + self.log_counts.get("critical", 0))
            / max(total_logs, 1)
        ) * 100

        return {
            "log_counts": self.log_counts.copy(),
            "total_logs": total_logs,
            "error_rate_percentage": round(error_rate, 2),
            "error_patterns": dict(
                sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[
                    :10
                ]
            ),
            "performance_stats": self.metrics.get_all_stats(),
            "recent_alerts": self.alert_history[-10:],  # Last 10 alerts
            "alert_thresholds": self.alert_thresholds.copy(),
        }


class PerformanceMetrics:
    """Collect and manage performance metrics"""

    def __init__(self):
        self.operation_times: Dict[str, list] = {}
        self.operation_counts: Dict[str, int] = {}
        self.start_time = time.time()

    def record_operation(self, operation: str, duration: float):
        """Record operation performance"""
        if operation not in self.operation_times:
            self.operation_times[operation] = []
            self.operation_counts[operation] = 0

        self.operation_times[operation].append(duration)
        self.operation_counts[operation] += 1

        # Keep only last 1000 measurements per operation
        if len(self.operation_times[operation]) > 1000:
            self.operation_times[operation] = self.operation_times[operation][-1000:]

    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for a specific operation"""
        if operation not in self.operation_times:
            return {}

        times = self.operation_times[operation]
        return {
            "count": self.operation_counts[operation],
            "avg_duration": sum(times) / len(times),
            "min_duration": min(times),
            "max_duration": max(times),
            "recent_avg": sum(times[-10:]) / min(len(times), 10),
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """Get all performance statistics"""
        stats = {"uptime_seconds": time.time() - self.start_time, "operations": {}}

        for operation in self.operation_times:
            stats["operations"][operation] = self.get_operation_stats(operation)

        return stats


class HealthChecker:
    """System health monitoring with enhanced capabilities"""

    def __init__(self):
        self.checks = {}
        self.last_check_time = None
        self.check_interval = 30  # seconds
        self.check_history = {}  # Store recent check results
        self.max_history = 100

    def register_check(self, name: str, check_func: callable, critical: bool = False):
        """Register a health check function"""
        self.checks[name] = {
            "func": check_func,
            "critical": critical,
            "last_result": None,
            "failure_count": 0,
        }

    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks with enhanced error handling"""
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "healthy",
            "checks": {},
            "summary": {
                "total_checks": len(self.checks),
                "healthy": 0,
                "unhealthy": 0,
                "warnings": 0,
            },
        }

        for name, check_info in self.checks.items():
            check_func = check_info["func"]
            start_time = time.time()

            try:
                if hasattr(check_func, "__call__"):
                    if hasattr(check_func, "__await__"):
                        result = await check_func()
                    else:
                        result = check_func()

                    response_time = (time.time() - start_time) * 1000  # ms

                    # Determine status based on result
                    if isinstance(result, dict) and "status" in result:
                        status = result["status"]
                    else:
                        status = "healthy"

                    check_result = {
                        "status": status,
                        "details": result,
                        "response_time_ms": round(response_time, 2),
                        "critical": check_info["critical"],
                    }

                    results["checks"][name] = check_result
                    check_info["last_result"] = check_result
                    check_info["failure_count"] = 0

                    # Update summary
                    if status == "healthy":
                        results["summary"]["healthy"] += 1
                    elif status == "warning":
                        results["summary"]["warnings"] += 1
                    else:
                        results["summary"]["unhealthy"] += 1
                        if check_info["critical"]:
                            results["overall_status"] = "critical"
                        elif results["overall_status"] == "healthy":
                            results["overall_status"] = "degraded"

                else:
                    check_result = {
                        "status": "error",
                        "error": "Invalid check function",
                        "critical": check_info["critical"],
                    }
                    results["checks"][name] = check_result
                    check_info["failure_count"] += 1
                    results["summary"]["unhealthy"] += 1

            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                check_result = {
                    "status": "unhealthy",
                    "error": str(e),
                    "response_time_ms": round(response_time, 2),
                    "critical": check_info["critical"],
                }
                results["checks"][name] = check_result
                check_info["last_result"] = check_result
                check_info["failure_count"] += 1
                results["summary"]["unhealthy"] += 1

                if check_info["critical"]:
                    results["overall_status"] = "critical"
                elif results["overall_status"] == "healthy":
                    results["overall_status"] = "degraded"

        # Store in history
        self._store_check_history(results)
        self.last_check_time = time.time()
        return results

    def _store_check_history(self, results: Dict[str, Any]):
        """Store check results in history"""
        timestamp = results["timestamp"]
        if len(self.check_history) >= self.max_history:
            # Remove oldest entry
            oldest_key = min(self.check_history.keys())
            del self.check_history[oldest_key]

        self.check_history[timestamp] = {
            "overall_status": results["overall_status"],
            "summary": results["summary"],
        }

    def get_check_history(self, limit: int = 50) -> Dict[str, Any]:
        """Get recent check history"""
        sorted_history = dict(sorted(self.check_history.items(), reverse=True)[:limit])
        return sorted_history

    def get_basic_health(self) -> Dict[str, Any]:
        """Get basic system health information with enhanced metrics"""
        try:
            import psutil

            # CPU information
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Memory information
            memory = psutil.virtual_memory()

            # Disk information
            disk_path = "/" if os.name != "nt" else "C:"
            disk = psutil.disk_usage(disk_path)

            # Network information
            network = psutil.net_io_counters()

            # Process information
            process_count = len(psutil.pids())

            # Load average (Unix only)
            load_avg = None
            if hasattr(os, "getloadavg"):
                try:
                    load_avg = os.getloadavg()
                except OSError:
                    pass

            health_data = {
                "status": "healthy",
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "load_avg": load_avg,
                },
                "memory": {
                    "percent": memory.percent,
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                },
                "disk": {
                    "percent": disk.percent,
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv,
                },
                "processes": {"count": process_count},
            }

            # Determine health status based on thresholds
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
                health_data["status"] = "warning"
            if cpu_percent > 95 or memory.percent > 95 or disk.percent > 95:
                health_data["status"] = "unhealthy"

            return health_data

        except ImportError:
            return {
                "status": "warning",
                "message": "psutil not installed - limited health monitoring available",
                "basic_info": {"platform": os.name, "pid": os.getpid()},
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def check_dependencies(self) -> Dict[str, Any]:
        """Check if critical dependencies are available"""
        dependencies = {
            "psutil": False,
            "loguru": False,
            "asyncio": False,
            "json": False,
            "datetime": False,
            "pathlib": False,
        }

        # Check each dependency
        for dep_name in dependencies.keys():
            try:
                __import__(dep_name)
                dependencies[dep_name] = True
            except ImportError:
                pass

        missing = [dep for dep, available in dependencies.items() if not available]
        critical_missing = [
            dep for dep in missing if dep in ["psutil", "loguru", "asyncio"]
        ]

        status = "healthy"
        if critical_missing:
            status = "unhealthy"
        elif missing:
            status = "warning"

        return {
            "status": status,
            "dependencies": dependencies,
            "missing": missing,
            "critical_missing": critical_missing,
            "message": (
                f"Critical dependencies missing: {', '.join(critical_missing)}"
                if critical_missing
                else (
                    f"Missing dependencies: {', '.join(missing)}"
                    if missing
                    else "All dependencies available"
                )
            ),
        }

    def check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        try:
            import psutil

            # Check workspace directory disk space
            workspace_path = PROJECT_ROOT
            disk_usage = psutil.disk_usage(str(workspace_path))

            free_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            used_percent = (disk_usage.used / disk_usage.total) * 100

            status = "healthy"
            if used_percent > 95:
                status = "critical"
            elif used_percent > 90:
                status = "warning"

            return {
                "status": status,
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "used_percent": round(used_percent, 2),
                "message": f"Disk usage: {used_percent:.1f}% ({free_gb:.1f}GB free)",
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to check disk space",
            }

    def check_log_directory(self) -> Dict[str, Any]:
        """Check if log directory is writable and has space"""
        try:
            logs_dir = PROJECT_ROOT / "logs"

            # Check if directory exists and is writable
            if not logs_dir.exists():
                logs_dir.mkdir(exist_ok=True)

            # Test write access
            test_file = logs_dir / "health_check_test.tmp"
            try:
                test_file.write_text("test")
                test_file.unlink()
                writable = True
            except Exception:
                writable = False

            # Count log files
            log_files = list(logs_dir.glob("*.log"))
            log_count = len(log_files)

            # Calculate total log size
            total_size = sum(f.stat().st_size for f in log_files if f.exists())
            total_size_mb = total_size / (1024 * 1024)

            status = "healthy"
            if not writable:
                status = "critical"
            elif total_size_mb > 1000:  # More than 1GB of logs
                status = "warning"

            return {
                "status": status,
                "writable": writable,
                "log_files_count": log_count,
                "total_size_mb": round(total_size_mb, 2),
                "message": f"Log directory: {log_count} files, {total_size_mb:.1f}MB",
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to check log directory",
            }


def set_correlation_id(correlation_id: str):
    """Set correlation ID for current context"""
    correlation_id_var.set(correlation_id)


def set_request_id(request_id: str):
    """Set request ID for current context"""
    request_id_var.set(request_id)


def set_agent_id(agent_id: str):
    """Set agent ID for current context"""
    agent_id_var.set(agent_id)


def set_operation(operation: str):
    """Set current operation for context"""
    operation_var.set(operation)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID"""
    return correlation_id_var.get()


def generate_correlation_id() -> str:
    """Generate a new correlation ID"""
    return str(uuid4())


class LoggingContext:
    """Context manager for logging with correlation IDs"""

    def __init__(
        self,
        correlation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        self.correlation_id = correlation_id or generate_correlation_id()
        self.request_id = request_id
        self.agent_id = agent_id
        self.operation = operation

        # Store previous values for restoration
        self.prev_correlation_id = None
        self.prev_request_id = None
        self.prev_agent_id = None
        self.prev_operation = None

    def __enter__(self):
        # Store previous values
        self.prev_correlation_id = correlation_id_var.get()
        self.prev_request_id = request_id_var.get()
        self.prev_agent_id = agent_id_var.get()
        self.prev_operation = operation_var.get()

        # Set new values
        correlation_id_var.set(self.correlation_id)
        if self.request_id:
            request_id_var.set(self.request_id)
        if self.agent_id:
            agent_id_var.set(self.agent_id)
        if self.operation:
            operation_var.set(self.operation)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous values
        correlation_id_var.set(self.prev_correlation_id)
        request_id_var.set(self.prev_request_id)
        agent_id_var.set(self.prev_agent_id)
        operation_var.set(self.prev_operation)


def define_log_level(print_level="INFO", logfile_level="DEBUG", name: str = None):
    """Adjust the log level to above level"""
    global _print_level
    _print_level = print_level

    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d%H%M%S")
    log_name = (
        f"{name}_{formatted_date}" if name else formatted_date
    )  # name a log with prefix name

    # Ensure logs directory exists
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    _logger.add(logs_dir / f"{log_name}.log", level=logfile_level)

    return StructuredLogger(_logger)


# Global instances
logger = define_log_level()
health_checker = HealthChecker()

# Register enhanced health checks
health_checker.register_check(
    "basic_system", health_checker.get_basic_health, critical=True
)
health_checker.register_check(
    "dependencies", health_checker.check_dependencies, critical=False
)
health_checker.register_check(
    "disk_space", health_checker.check_disk_space, critical=True
)
health_checker.register_check(
    "log_directory", health_checker.check_log_directory, critical=True
)


# Register logging health check
def check_logging_health():
    """Check logging system health"""
    try:
        stats = logger.get_log_statistics()
        error_rate = stats["log_counts"].get("error", 0) + stats["log_counts"].get(
            "critical", 0
        )
        total_logs = sum(stats["log_counts"].values())

        if total_logs == 0:
            return {"status": "warning", "message": "No logs recorded yet"}

        error_percentage = (error_rate / total_logs) * 100

        if error_percentage > 50:
            return {
                "status": "unhealthy",
                "message": f"High error rate: {error_percentage:.1f}%",
                "stats": stats,
            }
        elif error_percentage > 20:
            return {
                "status": "warning",
                "message": f"Elevated error rate: {error_percentage:.1f}%",
                "stats": stats,
            }
        else:
            return {
                "status": "healthy",
                "message": f"Error rate: {error_percentage:.1f}%",
                "stats": stats,
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


health_checker.register_check("logging_system", check_logging_health, critical=False)


class PerformanceMonitor:
    """Context manager and decorator for performance monitoring"""

    def __init__(self, operation_name: str, logger_instance: StructuredLogger = None):
        self.operation_name = operation_name
        self.logger = logger_instance or logger
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting operation: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        success = exc_type is None
        extra_data = {
            "operation": self.operation_name,
            "duration_seconds": duration,
            "success": success,
        }

        if not success:
            extra_data["error_type"] = exc_type.__name__ if exc_type else None
            extra_data["error_message"] = str(exc_val) if exc_val else None

        self.logger.log_performance(self.operation_name, duration, extra_data)

    def get_duration(self) -> Optional[float]:
        """Get operation duration if completed"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


def monitor_performance(operation_name: str):
    """Decorator for monitoring function performance"""

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                with PerformanceMonitor(operation_name):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                with PerformanceMonitor(operation_name):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


# Export commonly used functions
__all__ = [
    "logger",
    "health_checker",
    "StructuredLogger",
    "HealthChecker",
    "PerformanceMonitor",
    "monitor_performance",
    "LoggingContext",
    "set_correlation_id",
    "set_request_id",
    "set_agent_id",
    "set_operation",
    "get_correlation_id",
    "generate_correlation_id",
]


if __name__ == "__main__":
    # Test the enhanced logging system
    with LoggingContext(operation="test_logging"):
        logger.info("Starting application with enhanced logging")
        logger.debug("Debug message with correlation ID")
        logger.warning("Warning message")

        # Test performance monitoring
        with PerformanceMonitor("test_operation"):
            time.sleep(0.1)  # Simulate work

        # Test error logging with alerts
        logger.error("Test error message")
        logger.critical("Test critical message")

        # Test security event logging
        logger.log_security_event(
            "test_security_event",
            {
                "user_id": "test_user",
                "action": "login_attempt",
                "ip_address": "192.168.1.1",
            },
            severity="high",
        )

        # Test business event logging
        logger.log_business_event(
            "user_action", {"action_type": "document_created", "user_id": "test_user"}
        )

        try:
            raise ValueError("Test error for exception logging")
        except Exception as e:
            logger.exception(f"An error occurred: {e}")

    # Print logging statistics
    stats = logger.get_log_statistics()
    print(f"Logging statistics: {stats}")
