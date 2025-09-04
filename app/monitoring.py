"""
Monitoring and observability module for OpenManus.

This module provides comprehensive monitoring capabilities including:
- Performance metrics collection
- Health check endpoints
- System resource monitoring
- Dashboard configuration
- Alert management
"""

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil

from app.config import config
from app.logger import HealthChecker, logger


class MetricType(Enum):
    """Types of metrics that can be collected"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Represents a monitoring alert"""

    alert_id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metric_name: str
    current_value: float
    threshold_value: float
    labels: Dict[str, str]
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None


class AlertManager:
    """Manages monitoring alerts and notifications"""

    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.max_history = 1000

    def add_alert_rule(
        self,
        name: str,
        metric_name: str,
        threshold: float,
        comparison: str = "greater_than",
        severity: AlertSeverity = AlertSeverity.MEDIUM,
        labels: Dict[str, str] = None,
    ):
        """Add an alert rule"""
        self.alert_rules[name] = {
            "metric_name": metric_name,
            "threshold": threshold,
            "comparison": comparison,
            "severity": severity,
            "labels": labels or {},
            "enabled": True,
        }

    def check_alert_rules(self, metrics: Dict[str, Any]):
        """Check all alert rules against current metrics"""
        for rule_name, rule in self.alert_rules.items():
            if not rule.get("enabled", True):
                continue

            metric_value = self._get_metric_value(metrics, rule["metric_name"])
            if metric_value is None:
                continue

            should_alert = self._evaluate_condition(
                metric_value, rule["threshold"], rule["comparison"]
            )

            alert_id = f"{rule_name}_{rule['metric_name']}"

            if should_alert and alert_id not in self.active_alerts:
                # Create new alert
                alert = Alert(
                    alert_id=alert_id,
                    name=rule_name,
                    severity=rule["severity"],
                    message=f"{rule_name}: {rule['metric_name']} is {metric_value} (threshold: {rule['threshold']})",
                    timestamp=datetime.now(timezone.utc),
                    metric_name=rule["metric_name"],
                    current_value=metric_value,
                    threshold_value=rule["threshold"],
                    labels=rule["labels"],
                )

                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)

                # Log the alert
                logger.warning(
                    f"ALERT TRIGGERED: {alert.message}",
                    {
                        "alert_id": alert_id,
                        "severity": alert.severity.value,
                        "metric_name": alert.metric_name,
                        "current_value": alert.current_value,
                        "threshold": alert.threshold_value,
                    },
                )

            elif not should_alert and alert_id in self.active_alerts:
                # Resolve existing alert
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_timestamp = datetime.now(timezone.utc)

                del self.active_alerts[alert_id]

                logger.info(
                    f"ALERT RESOLVED: {alert.message}",
                    {
                        "alert_id": alert_id,
                        "resolution_time": alert.resolved_timestamp.isoformat(),
                    },
                )

        # Clean up old history
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history :]

    def _get_metric_value(
        self, metrics: Dict[str, Any], metric_path: str
    ) -> Optional[float]:
        """Extract metric value from metrics dict using dot notation"""
        try:
            value = metrics
            for key in metric_path.split("."):
                value = value[key]
            return float(value) if value is not None else None
        except (KeyError, TypeError, ValueError):
            return None

    def _evaluate_condition(
        self, value: float, threshold: float, comparison: str
    ) -> bool:
        """Evaluate alert condition"""
        if comparison == "greater_than":
            return value > threshold
        elif comparison == "less_than":
            return value < threshold
        elif comparison == "equals":
            return abs(value - threshold) < 0.001  # Float comparison
        elif comparison == "greater_equal":
            return value >= threshold
        elif comparison == "less_equal":
            return value <= threshold
        else:
            return False

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 50) -> List[Alert]:
        """Get recent alert history"""
        return self.alert_history[-limit:]

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        active_by_severity = {}
        for alert in self.active_alerts.values():
            severity = alert.severity.value
            active_by_severity[severity] = active_by_severity.get(severity, 0) + 1

        return {
            "active_alerts_count": len(self.active_alerts),
            "active_by_severity": active_by_severity,
            "total_rules": len(self.alert_rules),
            "enabled_rules": sum(
                1 for rule in self.alert_rules.values() if rule.get("enabled", True)
            ),
            "history_count": len(self.alert_history),
        }


@dataclass
class Metric:
    """Represents a single metric measurement"""

    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str]
    unit: Optional[str] = None


@dataclass
class HealthCheckResult:
    """Result of a health check"""

    name: str
    status: str  # "healthy", "unhealthy", "warning"
    message: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None
    response_time_ms: Optional[float] = None


@dataclass
class SystemMetrics:
    """System-level metrics"""

    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    open_files: int
    timestamp: datetime


class MetricsCollector:
    """Collects and manages application metrics"""

    def __init__(self):
        self.metrics: List[Metric] = []
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}
        self.timers: Dict[str, List[float]] = {}
        self.max_metrics_history = 10000
        self.start_time = time.time()
        self.custom_metrics = {}  # For application-specific metrics

    def increment_counter(
        self, name: str, value: float = 1.0, labels: Dict[str, str] = None
    ):
        """Increment a counter metric"""
        labels = labels or {}
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        self.counters[key] = self.counters.get(key, 0) + value

        metric = Metric(
            name=name,
            value=self.counters[key],
            metric_type=MetricType.COUNTER,
            timestamp=datetime.now(timezone.utc),
            labels=labels,
        )
        self._add_metric(metric)

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric value"""
        labels = labels or {}
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        self.gauges[key] = value

        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            timestamp=datetime.now(timezone.utc),
            labels=labels,
        )
        self._add_metric(metric)

    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram value"""
        labels = labels or {}
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"

        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)

        # Keep only last 1000 values
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]

        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            timestamp=datetime.now(timezone.utc),
            labels=labels,
        )
        self._add_metric(metric)

    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None):
        """Record a timer measurement"""
        labels = labels or {}
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"

        if key not in self.timers:
            self.timers[key] = []
        self.timers[key].append(duration)

        # Keep only last 1000 values
        if len(self.timers[key]) > 1000:
            self.timers[key] = self.timers[key][-1000:]

        metric = Metric(
            name=name,
            value=duration,
            metric_type=MetricType.TIMER,
            timestamp=datetime.now(timezone.utc),
            labels=labels,
            unit="seconds",
        )
        self._add_metric(metric)

    def _add_metric(self, metric: Metric):
        """Add metric to history"""
        self.metrics.append(metric)

        # Keep metrics history under limit
        if len(self.metrics) > self.max_metrics_history:
            self.metrics = self.metrics[-self.max_metrics_history :]

    def get_counter_value(self, name: str, labels: Dict[str, str] = None) -> float:
        """Get current counter value"""
        labels = labels or {}
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        return self.counters.get(key, 0)

    def get_gauge_value(
        self, name: str, labels: Dict[str, str] = None
    ) -> Optional[float]:
        """Get current gauge value"""
        labels = labels or {}
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        return self.gauges.get(key)

    def get_histogram_stats(
        self, name: str, labels: Dict[str, str] = None
    ) -> Dict[str, float]:
        """Get histogram statistics"""
        labels = labels or {}
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        values = self.histograms.get(key, [])

        if not values:
            return {}

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "p50": self._percentile(values, 50),
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99),
        }

    def get_timer_stats(
        self, name: str, labels: Dict[str, str] = None
    ) -> Dict[str, float]:
        """Get timer statistics"""
        labels = labels or {}
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        values = self.timers.get(key, [])

        if not values:
            return {}

        return {
            "count": len(values),
            "min_ms": min(values) * 1000,
            "max_ms": max(values) * 1000,
            "avg_ms": (sum(values) / len(values)) * 1000,
            "p50_ms": self._percentile(values, 50) * 1000,
            "p95_ms": self._percentile(values, 95) * 1000,
            "p99_ms": self._percentile(values, 99) * 1000,
        }

    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]

    def record_custom_metric(
        self, name: str, value: Any, labels: Dict[str, str] = None
    ):
        """Record a custom application metric"""
        labels = labels or {}
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"

        self.custom_metrics[key] = {
            "value": value,
            "timestamp": datetime.now(timezone.utc),
            "labels": labels,
        }

    def get_uptime_seconds(self) -> float:
        """Get collector uptime in seconds"""
        return time.time() - self.start_time

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics"""
        return {
            "total_metrics": len(self.metrics),
            "counters_count": len(self.counters),
            "gauges_count": len(self.gauges),
            "histograms_count": len(self.histograms),
            "timers_count": len(self.timers),
            "custom_metrics_count": len(self.custom_metrics),
            "uptime_seconds": self.get_uptime_seconds(),
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics with enhanced data"""
        return {
            "counters": {k: v for k, v in self.counters.items()},
            "gauges": {k: v for k, v in self.gauges.items()},
            "histograms": {
                k: self.get_histogram_stats(
                    k.split(":")[0], json.loads(k.split(":", 1)[1]) if ":" in k else {}
                )
                for k in self.histograms.keys()
            },
            "timers": {
                k: self.get_timer_stats(
                    k.split(":")[0], json.loads(k.split(":", 1)[1]) if ":" in k else {}
                )
                for k in self.timers.keys()
            },
            "custom_metrics": {k: v for k, v in self.custom_metrics.items()},
            "summary": self.get_metrics_summary(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class SystemMonitor:
    """Monitors system resources and health"""

    def __init__(self):
        self.start_time = time.time()

    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)

            # Disk metrics
            disk = psutil.disk_usage("/" if os.name != "nt" else "C:")
            disk_used_gb = disk.used / (1024 * 1024 * 1024)
            disk_free_gb = disk.free / (1024 * 1024 * 1024)

            # Network metrics
            network = psutil.net_io_counters()

            # Process metrics
            process_count = len(psutil.pids())

            # Open files (current process)
            try:
                current_process = psutil.Process()
                open_files = len(current_process.open_files())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                open_files = 0

            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_percent=disk.percent,
                disk_used_gb=disk_used_gb,
                disk_free_gb=disk_free_gb,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                process_count=process_count,
                open_files=open_files,
                timestamp=datetime.now(timezone.utc),
            )
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            # Return minimal metrics on error
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_percent=0.0,
                disk_used_gb=0.0,
                disk_free_gb=0.0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                process_count=0,
                open_files=0,
                timestamp=datetime.now(timezone.utc),
            )

    def get_uptime_seconds(self) -> float:
        """Get application uptime in seconds"""
        return time.time() - self.start_time

    def get_process_info(self) -> Dict[str, Any]:
        """Get current process information"""
        try:
            process = psutil.Process()
            return {
                "pid": process.pid,
                "name": process.name(),
                "status": process.status(),
                "create_time": process.create_time(),
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "memory_info": process.memory_info()._asdict(),
                "num_threads": process.num_threads(),
                "num_fds": process.num_fds() if hasattr(process, "num_fds") else 0,
            }
        except Exception as e:
            logger.error(f"Failed to get process info: {e}")
            return {"error": str(e)}


class PerformanceTimer:
    """Context manager for timing operations"""

    def __init__(
        self,
        operation_name: str,
        metrics_collector: MetricsCollector,
        labels: Dict[str, str] = None,
    ):
        self.operation_name = operation_name
        self.metrics_collector = metrics_collector
        self.labels = labels or {}
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics_collector.record_timer(
                self.operation_name, duration, self.labels
            )

            # Also log performance
            logger.log_performance(
                self.operation_name,
                duration,
                {"labels": self.labels, "success": exc_type is None},
            )


class MonitoringDashboard:
    """Provides monitoring dashboard data and configuration"""

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        system_monitor: SystemMonitor,
        health_checker: HealthChecker,
        alert_manager: AlertManager = None,
    ):
        self.metrics_collector = metrics_collector
        self.system_monitor = system_monitor
        self.health_checker = health_checker
        self.alert_manager = alert_manager or AlertManager()

        # Set up default alert rules
        self._setup_default_alerts()

    def _setup_default_alerts(self):
        """Set up default monitoring alert rules"""
        # System resource alerts
        self.alert_manager.add_alert_rule(
            "high_cpu_usage",
            "system.cpu_percent",
            80.0,
            "greater_than",
            AlertSeverity.MEDIUM,
        )

        self.alert_manager.add_alert_rule(
            "critical_cpu_usage",
            "system.cpu_percent",
            95.0,
            "greater_than",
            AlertSeverity.CRITICAL,
        )

        self.alert_manager.add_alert_rule(
            "high_memory_usage",
            "system.memory_percent",
            85.0,
            "greater_than",
            AlertSeverity.MEDIUM,
        )

        self.alert_manager.add_alert_rule(
            "critical_memory_usage",
            "system.memory_percent",
            95.0,
            "greater_than",
            AlertSeverity.CRITICAL,
        )

        self.alert_manager.add_alert_rule(
            "high_disk_usage",
            "system.disk_percent",
            90.0,
            "greater_than",
            AlertSeverity.HIGH,
        )

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data with alerts"""
        try:
            # Get health checks
            health_results = await self.health_checker.run_health_checks()

            # Get system metrics
            system_metrics = self.system_monitor.get_system_metrics()

            # Get application metrics
            app_metrics = self.metrics_collector.get_all_metrics()

            # Get process info
            process_info = self.system_monitor.get_process_info()

            # Prepare metrics for alert checking
            dashboard_metrics = {
                "system": asdict(system_metrics),
                "process": process_info,
                "metrics": app_metrics,
            }

            # Check alert rules
            self.alert_manager.check_alert_rules(dashboard_metrics)

            # Get alert information
            active_alerts = self.alert_manager.get_active_alerts()
            alert_summary = self.alert_manager.get_alert_summary()

            # Determine overall status considering alerts
            overall_status = "operational"
            if health_results["overall_status"] != "healthy":
                overall_status = "degraded"
            elif any(
                alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]
                for alert in active_alerts
            ):
                overall_status = "degraded"

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": self.system_monitor.get_uptime_seconds(),
                "health": health_results,
                "system": asdict(system_metrics),
                "process": process_info,
                "metrics": app_metrics,
                "alerts": {
                    "active": [asdict(alert) for alert in active_alerts],
                    "summary": alert_summary,
                    "recent_history": [
                        asdict(alert)
                        for alert in self.alert_manager.get_alert_history(10)
                    ],
                },
                "status": overall_status,
            }
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "error": str(e),
            }

    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        try:
            lines = []

            # Add system metrics
            system_metrics = self.system_monitor.get_system_metrics()
            lines.append(f"# HELP system_cpu_percent CPU usage percentage")
            lines.append(f"# TYPE system_cpu_percent gauge")
            lines.append(f"system_cpu_percent {system_metrics.cpu_percent}")

            lines.append(f"# HELP system_memory_percent Memory usage percentage")
            lines.append(f"# TYPE system_memory_percent gauge")
            lines.append(f"system_memory_percent {system_metrics.memory_percent}")

            lines.append(f"# HELP system_disk_percent Disk usage percentage")
            lines.append(f"# TYPE system_disk_percent gauge")
            lines.append(f"system_disk_percent {system_metrics.disk_percent}")

            # Add application counters
            for key, value in self.metrics_collector.counters.items():
                metric_name = key.split(":")[0]
                lines.append(f"# HELP {metric_name} Application counter")
                lines.append(f"# TYPE {metric_name} counter")
                lines.append(f"{metric_name} {value}")

            # Add application gauges
            for key, value in self.metrics_collector.gauges.items():
                metric_name = key.split(":")[0]
                lines.append(f"# HELP {metric_name} Application gauge")
                lines.append(f"# TYPE {metric_name} gauge")
                lines.append(f"{metric_name} {value}")

            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Failed to generate Prometheus metrics: {e}")
            return f"# Error generating metrics: {e}"


# Global instances
metrics_collector = MetricsCollector()
system_monitor = SystemMonitor()
alert_manager = AlertManager()
# Create health checker instance for monitoring
monitoring_health_checker = HealthChecker()

monitoring_dashboard = MonitoringDashboard(
    metrics_collector, system_monitor, monitoring_health_checker, alert_manager
)


def timer(operation_name: str, labels: Dict[str, str] = None):
    """Decorator for timing function execution"""

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                with PerformanceTimer(operation_name, metrics_collector, labels):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                with PerformanceTimer(operation_name, metrics_collector, labels):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


def increment_counter(name: str, value: float = 1.0, labels: Dict[str, str] = None):
    """Convenience function to increment a counter"""
    metrics_collector.increment_counter(name, value, labels)


def set_gauge(name: str, value: float, labels: Dict[str, str] = None):
    """Convenience function to set a gauge"""
    metrics_collector.set_gauge(name, value, labels)


def record_histogram(name: str, value: float, labels: Dict[str, str] = None):
    """Convenience function to record a histogram value"""
    metrics_collector.record_histogram(name, value, labels)
