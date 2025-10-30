"""
Performance metrics collection and reporting system for OpenManus.

This module provides comprehensive performance monitoring including:
- Response time tracking and SLA monitoring
- Performance dashboards and alerting
- Automated performance testing and benchmarking
- Integration with existing monitoring infrastructure
"""

import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.config import config
from app.logger import logger
from app.monitoring import (
    AlertSeverity,
    MetricsCollector,
    SystemMonitor,
    alert_manager,
    metrics_collector,
    system_monitor,
)


class PerformanceLevel(Enum):
    """Performance level classifications"""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


class SLAStatus(Enum):
    """SLA compliance status"""

    MEETING = "meeting"
    AT_RISK = "at_risk"
    BREACHED = "breached"


@dataclass
class SLATarget:
    """Service Level Agreement target definition"""

    name: str
    metric_name: str
    target_value: float
    comparison: str  # "less_than", "greater_than", "equals"
    measurement_window_minutes: int = 60
    breach_threshold_percent: float = 95.0  # % of measurements that must meet target
    warning_threshold_percent: float = 90.0  # % threshold for warnings
    enabled: bool = True


@dataclass
class SLAMeasurement:
    """Individual SLA measurement"""

    timestamp: datetime
    value: float
    meets_target: bool
    target_value: float


@dataclass
class SLAReport:
    """SLA compliance report"""

    sla_name: str
    status: SLAStatus
    compliance_percent: float
    target_value: float
    current_value: float
    measurements_count: int
    breach_count: int
    window_start: datetime
    window_end: datetime
    trend: str  # "improving", "stable", "degrading"


@dataclass
class PerformanceBenchmark:
    """Performance benchmark definition"""

    name: str
    operation: str
    target_response_time_ms: float
    target_throughput_rps: float
    max_error_rate_percent: float = 5.0
    concurrent_users: int = 1
    duration_seconds: int = 60
    warmup_seconds: int = 10


@dataclass
class BenchmarkResult:
    """Performance benchmark test result"""

    benchmark_name: str
    timestamp: datetime
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    throughput_rps: float
    error_rate_percent: float
    performance_level: PerformanceLevel
    meets_targets: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance-specific alert"""

    alert_id: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: AlertSeverity
    message: str
    timestamp: datetime
    performance_level: PerformanceLevel
    suggested_actions: List[str] = field(default_factory=list)


class PerformanceMetricsCollector:
    """Enhanced metrics collector focused on performance monitoring"""

    def __init__(self, base_collector: MetricsCollector = None):
        self.base_collector = base_collector or metrics_collector
        self.response_times: Dict[str, List[Tuple[datetime, float]]] = {}
        self.throughput_data: Dict[str, List[Tuple[datetime, int]]] = {}
        self.error_rates: Dict[str, List[Tuple[datetime, float]]] = {}
        self.sla_targets: Dict[str, SLATarget] = {}
        self.sla_measurements: Dict[str, List[SLAMeasurement]] = {}
        self.performance_history: List[Dict[str, Any]] = []
        self.max_history_size = 10000
        self.cleanup_interval_hours = 24

        # Set up default SLA targets
        self._setup_default_sla_targets()

        # Set up performance alert rules
        self._setup_performance_alerts()

    def _setup_default_sla_targets(self):
        """Set up default SLA targets for common operations"""
        default_targets = [
            SLATarget(
                name="agent_response_time",
                metric_name="agent.response_time_ms",
                target_value=30000,  # 30 seconds
                comparison="less_than",
                measurement_window_minutes=60,
            ),
            SLATarget(
                name="tool_execution_time",
                metric_name="tool.execution_time_ms",
                target_value=10000,  # 10 seconds
                comparison="less_than",
                measurement_window_minutes=30,
            ),
            SLATarget(
                name="llm_api_response_time",
                metric_name="llm.api_response_time_ms",
                target_value=15000,  # 15 seconds
                comparison="less_than",
                measurement_window_minutes=15,
            ),
            SLATarget(
                name="system_availability",
                metric_name="system.availability_percent",
                target_value=99.5,
                comparison="greater_than",
                measurement_window_minutes=60,
            ),
            SLATarget(
                name="error_rate",
                metric_name="system.error_rate_percent",
                target_value=5.0,
                comparison="less_than",
                measurement_window_minutes=30,
            ),
        ]

        for target in default_targets:
            self.sla_targets[target.name] = target

    def _setup_performance_alerts(self):
        """Set up performance-specific alert rules"""
        # Response time alerts
        alert_manager.add_alert_rule(
            "slow_agent_response",
            "performance.agent_avg_response_time_ms",
            25000,  # 25 seconds
            "greater_than",
            AlertSeverity.MEDIUM,
            {"category": "performance"},
        )

        alert_manager.add_alert_rule(
            "critical_agent_response",
            "performance.agent_avg_response_time_ms",
            45000,  # 45 seconds
            "greater_than",
            AlertSeverity.CRITICAL,
            {"category": "performance"},
        )

        # Throughput alerts
        alert_manager.add_alert_rule(
            "low_throughput",
            "performance.requests_per_second",
            0.1,  # Less than 0.1 RPS
            "less_than",
            AlertSeverity.MEDIUM,
            {"category": "performance"},
        )

        # Error rate alerts
        alert_manager.add_alert_rule(
            "high_error_rate",
            "performance.error_rate_percent",
            10.0,
            "greater_than",
            AlertSeverity.HIGH,
            {"category": "performance"},
        )

    def record_response_time(
        self, operation: str, response_time_ms: float, labels: Dict[str, str] = None
    ):
        """Record response time for an operation"""
        timestamp = datetime.now(timezone.utc)
        labels = labels or {}

        # Store in internal tracking
        if operation not in self.response_times:
            self.response_times[operation] = []

        self.response_times[operation].append((timestamp, response_time_ms))

        # Clean up old data (keep last 1000 measurements per operation)
        if len(self.response_times[operation]) > 1000:
            self.response_times[operation] = self.response_times[operation][-1000:]

        # Record in base metrics collector
        self.base_collector.record_timer(
            f"{operation}.response_time", response_time_ms / 1000, labels
        )
        self.base_collector.record_histogram(
            f"{operation}.response_time_ms", response_time_ms, labels
        )

        # Update SLA measurements
        self._update_sla_measurements(f"{operation}.response_time_ms", response_time_ms)

        # Log performance data
        logger.log_performance(
            operation,
            response_time_ms / 1000,
            {"response_time_ms": response_time_ms, "labels": labels},
        )

    def record_throughput(
        self, operation: str, request_count: int, labels: Dict[str, str] = None
    ):
        """Record throughput for an operation"""
        timestamp = datetime.now(timezone.utc)
        labels = labels or {}

        # Store in internal tracking
        if operation not in self.throughput_data:
            self.throughput_data[operation] = []

        self.throughput_data[operation].append((timestamp, request_count))

        # Clean up old data
        if len(self.throughput_data[operation]) > 1000:
            self.throughput_data[operation] = self.throughput_data[operation][-1000:]

        # Record in base metrics collector
        self.base_collector.increment_counter(
            f"{operation}.requests_total", request_count, labels
        )

    def record_error_rate(
        self, operation: str, error_rate_percent: float, labels: Dict[str, str] = None
    ):
        """Record error rate for an operation"""
        timestamp = datetime.now(timezone.utc)
        labels = labels or {}

        # Store in internal tracking
        if operation not in self.error_rates:
            self.error_rates[operation] = []

        self.error_rates[operation].append((timestamp, error_rate_percent))

        # Clean up old data
        if len(self.error_rates[operation]) > 1000:
            self.error_rates[operation] = self.error_rates[operation][-1000:]

        # Record in base metrics collector
        self.base_collector.set_gauge(
            f"{operation}.error_rate_percent", error_rate_percent, labels
        )

        # Update SLA measurements
        self._update_sla_measurements(
            f"{operation}.error_rate_percent", error_rate_percent
        )

    def _update_sla_measurements(self, metric_name: str, value: float):
        """Update SLA measurements for a metric"""
        for sla_name, sla_target in self.sla_targets.items():
            if sla_target.metric_name == metric_name and sla_target.enabled:
                meets_target = self._evaluate_sla_condition(
                    value, sla_target.target_value, sla_target.comparison
                )

                measurement = SLAMeasurement(
                    timestamp=datetime.now(timezone.utc),
                    value=value,
                    meets_target=meets_target,
                    target_value=sla_target.target_value,
                )

                if sla_name not in self.sla_measurements:
                    self.sla_measurements[sla_name] = []

                self.sla_measurements[sla_name].append(measurement)

                # Keep measurements within window
                window_start = datetime.now(timezone.utc) - timedelta(
                    minutes=sla_target.measurement_window_minutes
                )
                self.sla_measurements[sla_name] = [
                    m
                    for m in self.sla_measurements[sla_name]
                    if m.timestamp >= window_start
                ]

    def _evaluate_sla_condition(
        self, value: float, target: float, comparison: str
    ) -> bool:
        """Evaluate if a value meets SLA target"""
        if comparison == "less_than":
            return value < target
        elif comparison == "greater_than":
            return value > target
        elif comparison == "equals":
            return abs(value - target) < 0.001
        elif comparison == "less_equal":
            return value <= target
        elif comparison == "greater_equal":
            return value >= target
        else:
            return False

    def get_response_time_stats(
        self, operation: str, window_minutes: int = 60
    ) -> Dict[str, float]:
        """Get response time statistics for an operation"""
        if operation not in self.response_times:
            return {}

        window_start = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        recent_times = [
            rt for ts, rt in self.response_times[operation] if ts >= window_start
        ]

        if not recent_times:
            return {}

        return {
            "count": len(recent_times),
            "avg_ms": statistics.mean(recent_times),
            "median_ms": statistics.median(recent_times),
            "min_ms": min(recent_times),
            "max_ms": max(recent_times),
            "p95_ms": self._percentile(recent_times, 95),
            "p99_ms": self._percentile(recent_times, 99),
            "std_dev_ms": (
                statistics.stdev(recent_times) if len(recent_times) > 1 else 0
            ),
        }

    def get_throughput_stats(
        self, operation: str, window_minutes: int = 60
    ) -> Dict[str, float]:
        """Get throughput statistics for an operation"""
        if operation not in self.throughput_data:
            return {}

        window_start = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        recent_data = [
            (ts, count)
            for ts, count in self.throughput_data[operation]
            if ts >= window_start
        ]

        if not recent_data:
            return {}

        total_requests = sum(count for _, count in recent_data)
        time_span_seconds = (
            (recent_data[-1][0] - recent_data[0][0]).total_seconds()
            if len(recent_data) > 1
            else window_minutes * 60
        )

        rps = total_requests / max(time_span_seconds, 1)

        return {
            "total_requests": total_requests,
            "requests_per_second": rps,
            "requests_per_minute": rps * 60,
            "time_span_seconds": time_span_seconds,
        }

    def get_error_rate_stats(
        self, operation: str, window_minutes: int = 60
    ) -> Dict[str, float]:
        """Get error rate statistics for an operation"""
        if operation not in self.error_rates:
            return {}

        window_start = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        recent_rates = [
            rate for ts, rate in self.error_rates[operation] if ts >= window_start
        ]

        if not recent_rates:
            return {}

        return {
            "count": len(recent_rates),
            "avg_error_rate": statistics.mean(recent_rates),
            "max_error_rate": max(recent_rates),
            "min_error_rate": min(recent_rates),
            "current_error_rate": recent_rates[-1] if recent_rates else 0,
        }

    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]

    def get_sla_report(self, sla_name: str) -> Optional[SLAReport]:
        """Generate SLA compliance report"""
        if sla_name not in self.sla_targets:
            return None

        sla_target = self.sla_targets[sla_name]
        measurements = self.sla_measurements.get(sla_name, [])

        if not measurements:
            return SLAReport(
                sla_name=sla_name,
                status=SLAStatus.MEETING,
                compliance_percent=100.0,
                target_value=sla_target.target_value,
                current_value=0.0,
                measurements_count=0,
                breach_count=0,
                window_start=datetime.now(timezone.utc),
                window_end=datetime.now(timezone.utc),
                trend="stable",
            )

        # Calculate compliance
        meeting_target = sum(1 for m in measurements if m.meets_target)
        compliance_percent = (meeting_target / len(measurements)) * 100

        # Determine status
        if compliance_percent >= sla_target.breach_threshold_percent:
            status = SLAStatus.MEETING
        elif compliance_percent >= sla_target.warning_threshold_percent:
            status = SLAStatus.AT_RISK
        else:
            status = SLAStatus.BREACHED

        # Calculate trend
        if len(measurements) >= 10:
            recent_half = measurements[len(measurements) // 2 :]
            older_half = measurements[: len(measurements) // 2]

            recent_compliance = (
                sum(1 for m in recent_half if m.meets_target) / len(recent_half)
            ) * 100
            older_compliance = (
                sum(1 for m in older_half if m.meets_target) / len(older_half)
            ) * 100

            if recent_compliance > older_compliance + 5:
                trend = "improving"
            elif recent_compliance < older_compliance - 5:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return SLAReport(
            sla_name=sla_name,
            status=status,
            compliance_percent=compliance_percent,
            target_value=sla_target.target_value,
            current_value=measurements[-1].value if measurements else 0.0,
            measurements_count=len(measurements),
            breach_count=len(measurements) - meeting_target,
            window_start=(
                measurements[0].timestamp
                if measurements
                else datetime.now(timezone.utc)
            ),
            window_end=(
                measurements[-1].timestamp
                if measurements
                else datetime.now(timezone.utc)
            ),
            trend=trend,
        )

    def get_all_sla_reports(self) -> Dict[str, SLAReport]:
        """Get SLA reports for all configured targets"""
        reports = {}
        for sla_name in self.sla_targets.keys():
            report = self.get_sla_report(sla_name)
            if report:
                reports[sla_name] = report
        return reports

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "response_times": {},
            "throughput": {},
            "error_rates": {},
            "sla_compliance": {},
            "overall_performance": "good",
        }

        # Collect response time stats
        for operation in self.response_times.keys():
            summary["response_times"][operation] = self.get_response_time_stats(
                operation
            )

        # Collect throughput stats
        for operation in self.throughput_data.keys():
            summary["throughput"][operation] = self.get_throughput_stats(operation)

        # Collect error rate stats
        for operation in self.error_rates.keys():
            summary["error_rates"][operation] = self.get_error_rate_stats(operation)

        # Collect SLA reports
        sla_reports = self.get_all_sla_reports()
        summary["sla_compliance"] = {
            name: asdict(report) for name, report in sla_reports.items()
        }

        # Determine overall performance level
        breached_slas = [
            report
            for report in sla_reports.values()
            if report.status == SLAStatus.BREACHED
        ]
        at_risk_slas = [
            report
            for report in sla_reports.values()
            if report.status == SLAStatus.AT_RISK
        ]

        if breached_slas:
            summary["overall_performance"] = "critical"
        elif at_risk_slas:
            summary["overall_performance"] = "degraded"
        elif any(report.compliance_percent < 95 for report in sla_reports.values()):
            summary["overall_performance"] = "acceptable"
        else:
            summary["overall_performance"] = "excellent"

        return summary

    def add_sla_target(self, sla_target: SLATarget):
        """Add a new SLA target"""
        self.sla_targets[sla_target.name] = sla_target

    def remove_sla_target(self, sla_name: str):
        """Remove an SLA target"""
        if sla_name in self.sla_targets:
            del self.sla_targets[sla_name]
        if sla_name in self.sla_measurements:
            del self.sla_measurements[sla_name]

    def cleanup_old_data(self):
        """Clean up old performance data"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(
            hours=self.cleanup_interval_hours
        )

        # Clean response times
        for operation in list(self.response_times.keys()):
            self.response_times[operation] = [
                (ts, rt)
                for ts, rt in self.response_times[operation]
                if ts >= cutoff_time
            ]
            if not self.response_times[operation]:
                del self.response_times[operation]

        # Clean throughput data
        for operation in list(self.throughput_data.keys()):
            self.throughput_data[operation] = [
                (ts, count)
                for ts, count in self.throughput_data[operation]
                if ts >= cutoff_time
            ]
            if not self.throughput_data[operation]:
                del self.throughput_data[operation]

        # Clean error rates
        for operation in list(self.error_rates.keys()):
            self.error_rates[operation] = [
                (ts, rate)
                for ts, rate in self.error_rates[operation]
                if ts >= cutoff_time
            ]
            if not self.error_rates[operation]:
                del self.error_rates[operation]


class PerformanceBenchmarkRunner:
    """Runs automated performance benchmarks and tests"""

    def __init__(self, metrics_collector: PerformanceMetricsCollector):
        self.metrics_collector = metrics_collector
        self.benchmarks: Dict[str, PerformanceBenchmark] = {}
        self.benchmark_results: List[BenchmarkResult] = []
        self.max_results_history = 100

        # Set up default benchmarks
        self._setup_default_benchmarks()

    def _setup_default_benchmarks(self):
        """Set up default performance benchmarks"""
        default_benchmarks = [
            PerformanceBenchmark(
                name="agent_basic_response",
                operation="agent.process_request",
                target_response_time_ms=5000,
                target_throughput_rps=0.5,
                max_error_rate_percent=2.0,
                concurrent_users=1,
                duration_seconds=30,
            ),
            PerformanceBenchmark(
                name="tool_execution_performance",
                operation="tool.execute",
                target_response_time_ms=3000,
                target_throughput_rps=1.0,
                max_error_rate_percent=1.0,
                concurrent_users=2,
                duration_seconds=60,
            ),
            PerformanceBenchmark(
                name="concurrent_requests",
                operation="agent.concurrent_processing",
                target_response_time_ms=10000,
                target_throughput_rps=2.0,
                max_error_rate_percent=5.0,
                concurrent_users=5,
                duration_seconds=120,
            ),
        ]

        for benchmark in default_benchmarks:
            self.benchmarks[benchmark.name] = benchmark

    async def run_benchmark(
        self, benchmark_name: str, test_function: Callable = None
    ) -> BenchmarkResult:
        """Run a specific performance benchmark"""
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Benchmark '{benchmark_name}' not found")

        benchmark = self.benchmarks[benchmark_name]
        logger.info(f"Starting performance benchmark: {benchmark_name}")

        start_time = time.time()
        response_times = []
        error_count = 0
        total_requests = 0

        try:
            # Warmup period
            if benchmark.warmup_seconds > 0:
                logger.info(f"Warming up for {benchmark.warmup_seconds} seconds...")
                await asyncio.sleep(benchmark.warmup_seconds)

            # Run benchmark
            end_time = start_time + benchmark.duration_seconds
            tasks = []

            # Create concurrent tasks
            for _ in range(benchmark.concurrent_users):
                task = asyncio.create_task(
                    self._run_benchmark_worker(
                        benchmark, test_function, end_time, response_times
                    )
                )
                tasks.append(task)

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count errors and requests
            for result in results:
                if isinstance(result, Exception):
                    error_count += 1
                elif isinstance(result, dict):
                    total_requests += result.get("requests", 0)
                    error_count += result.get("errors", 0)

        except Exception as e:
            logger.error(f"Benchmark {benchmark_name} failed: {e}")
            error_count += 1

        # Calculate results
        duration = time.time() - start_time
        successful_requests = total_requests - error_count

        if response_times:
            avg_response_time = statistics.mean(response_times)
            p50_response_time = self._percentile(response_times, 50)
            p95_response_time = self._percentile(response_times, 95)
            p99_response_time = self._percentile(response_times, 99)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = p50_response_time = p95_response_time = 0
            p99_response_time = min_response_time = max_response_time = 0

        throughput = successful_requests / duration if duration > 0 else 0
        error_rate = (error_count / max(total_requests, 1)) * 100

        # Determine performance level and target compliance
        performance_level = self._determine_performance_level(
            avg_response_time, throughput, error_rate, benchmark
        )

        meets_targets = (
            avg_response_time <= benchmark.target_response_time_ms
            and throughput >= benchmark.target_throughput_rps
            and error_rate <= benchmark.max_error_rate_percent
        )

        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            timestamp=datetime.now(timezone.utc),
            duration_seconds=duration,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=error_count,
            avg_response_time_ms=avg_response_time,
            p50_response_time_ms=p50_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            min_response_time_ms=min_response_time,
            max_response_time_ms=max_response_time,
            throughput_rps=throughput,
            error_rate_percent=error_rate,
            performance_level=performance_level,
            meets_targets=meets_targets,
            details={
                "benchmark_config": asdict(benchmark),
                "response_times_sample": response_times[:100],  # First 100 samples
            },
        )

        # Store result
        self.benchmark_results.append(result)
        if len(self.benchmark_results) > self.max_results_history:
            self.benchmark_results = self.benchmark_results[-self.max_results_history :]

        # Record metrics
        self.metrics_collector.record_response_time(
            f"benchmark.{benchmark_name}", avg_response_time
        )
        self.metrics_collector.record_throughput(
            f"benchmark.{benchmark_name}", total_requests
        )
        self.metrics_collector.record_error_rate(
            f"benchmark.{benchmark_name}", error_rate
        )

        logger.info(
            f"Benchmark {benchmark_name} completed",
            {
                "duration_seconds": duration,
                "total_requests": total_requests,
                "avg_response_time_ms": avg_response_time,
                "throughput_rps": throughput,
                "error_rate_percent": error_rate,
                "performance_level": performance_level.value,
                "meets_targets": meets_targets,
            },
        )

        return result

    async def _run_benchmark_worker(
        self,
        benchmark: PerformanceBenchmark,
        test_function: Callable,
        end_time: float,
        response_times: List[float],
    ) -> Dict[str, int]:
        """Worker function for running benchmark requests"""
        requests = 0
        errors = 0

        while time.time() < end_time:
            request_start = time.time()
            try:
                if test_function:
                    await test_function()
                else:
                    # Default test - just sleep to simulate work
                    await asyncio.sleep(0.1)

                response_time_ms = (time.time() - request_start) * 1000
                response_times.append(response_time_ms)
                requests += 1

            except Exception as e:
                logger.debug(f"Benchmark request failed: {e}")
                errors += 1
                requests += 1

            # Small delay between requests
            await asyncio.sleep(0.01)

        return {"requests": requests, "errors": errors}

    def _determine_performance_level(
        self,
        avg_response_time: float,
        throughput: float,
        error_rate: float,
        benchmark: PerformanceBenchmark,
    ) -> PerformanceLevel:
        """Determine performance level based on metrics"""
        # Calculate performance scores (0-100)
        response_score = max(
            0, 100 - (avg_response_time / benchmark.target_response_time_ms) * 100
        )
        throughput_score = min(
            100, (throughput / benchmark.target_throughput_rps) * 100
        )
        error_score = max(
            0, 100 - (error_rate / benchmark.max_error_rate_percent) * 100
        )

        # Overall score
        overall_score = (response_score + throughput_score + error_score) / 3

        if overall_score >= 90:
            return PerformanceLevel.EXCELLENT
        elif overall_score >= 75:
            return PerformanceLevel.GOOD
        elif overall_score >= 60:
            return PerformanceLevel.ACCEPTABLE
        elif overall_score >= 40:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL

    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]

    def add_benchmark(self, benchmark: PerformanceBenchmark):
        """Add a new benchmark"""
        self.benchmarks[benchmark.name] = benchmark

    def get_benchmark_history(
        self, benchmark_name: str, limit: int = 10
    ) -> List[BenchmarkResult]:
        """Get recent benchmark results for a specific benchmark"""
        results = [
            r for r in self.benchmark_results if r.benchmark_name == benchmark_name
        ]
        return sorted(results, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends across benchmarks"""
        trends = {}

        for benchmark_name in self.benchmarks.keys():
            recent_results = self.get_benchmark_history(benchmark_name, 5)

            if len(recent_results) >= 2:
                # Calculate trend
                response_times = [r.avg_response_time_ms for r in recent_results]
                throughputs = [r.throughput_rps for r in recent_results]
                error_rates = [r.error_rate_percent for r in recent_results]

                trends[benchmark_name] = {
                    "response_time_trend": self._calculate_trend(response_times),
                    "throughput_trend": self._calculate_trend(
                        throughputs, reverse=True
                    ),
                    "error_rate_trend": self._calculate_trend(error_rates),
                    "recent_results_count": len(recent_results),
                    "latest_performance_level": recent_results[
                        0
                    ].performance_level.value,
                }

        return trends

    def _calculate_trend(self, values: List[float], reverse: bool = False) -> str:
        """Calculate trend direction from a list of values"""
        if len(values) < 2:
            return "stable"

        # Calculate linear regression slope
        n = len(values)
        x_values = list(range(n))
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n

        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # Determine trend (reverse for metrics where lower is better)
        threshold = 0.1
        if reverse:
            slope = -slope

        if slope > threshold:
            return "improving"
        elif slope < -threshold:
            return "degrading"
        else:
            return "stable"


# Global performance monitoring instances
performance_metrics_collector = PerformanceMetricsCollector()
performance_benchmark_runner = PerformanceBenchmarkRunner(performance_metrics_collector)


def performance_timer(operation_name: str, labels: Dict[str, str] = None):
    """Decorator for timing operations and recording performance metrics"""

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    response_time_ms = (time.time() - start_time) * 1000
                    performance_metrics_collector.record_response_time(
                        operation_name, response_time_ms, labels
                    )
                    return result
                except Exception as e:
                    response_time_ms = (time.time() - start_time) * 1000
                    performance_metrics_collector.record_response_time(
                        operation_name, response_time_ms, labels
                    )
                    # Record error
                    performance_metrics_collector.record_error_rate(
                        operation_name, 100.0, labels
                    )
                    raise

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    response_time_ms = (time.time() - start_time) * 1000
                    performance_metrics_collector.record_response_time(
                        operation_name, response_time_ms, labels
                    )
                    return result
                except Exception as e:
                    response_time_ms = (time.time() - start_time) * 1000
                    performance_metrics_collector.record_response_time(
                        operation_name, response_time_ms, labels
                    )
                    # Record error
                    performance_metrics_collector.record_error_rate(
                        operation_name, 100.0, labels
                    )
                    raise

            return sync_wrapper

    return decorator
