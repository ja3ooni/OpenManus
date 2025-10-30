"""
Automated performance testing and benchmarking for OpenManus.

This module provides comprehensive automated performance testing and benchmarking
capabilities, integrating with the metrics collection system to provide
continuous performance validation and regression detection.
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
from app.performance.metrics import (
    BenchmarkResult,
    PerformanceBenchmark,
    PerformanceLevel,
    PerformanceMetricsCollector,
    performance_metrics_collector,
)


class TestType(Enum):
    """Types of performance tests"""
    
    LOAD = "load"
    STRESS = "stress"
    SPIKE = "spike"
    ENDURANCE = "endurance"
    BASELINE = "baseline"
    REGRESSION = "regression"


class TestStatus(Enum):
    """Performance test execution status"""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PerformanceTestConfig:
    """Configuration for performance tests"""
    
    name: str
    test_type: TestType
    duration_seconds: int = 60
    concurrent_users: int = 1
    ramp_up_seconds: int = 10
    ramp_down_seconds: int = 10
    target_rps: float = 1.0
    max_response_time_ms: float = 30000
    max_error_rate_percent: float = 5.0
    warmup_requests: int = 10
    cooldown_seconds: int = 5
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceTestResult:
    """Results from a performance test execution"""
    
    test_name: str
    test_type: TestType
    status: TestStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    requests_per_second: float = 0.0
    avg_response_time_ms: float = 0.0
    p50_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    min_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    error_rate_percent: float = 0.0
    performance_level: PerformanceLevel = PerformanceLevel.GOOD
    meets_targets: bool = False
    errors: List[str] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionTestResult:
    """Results from performance regression testing"""
    
    test_name: str
    baseline_result: PerformanceTestResult
    current_result: PerformanceTestResult
    regression_detected: bool
    performance_change_percent: float
    response_time_change_percent: float
    throughput_change_percent: float
    error_rate_change_percent: float
    regression_threshold_percent: float = 10.0
    details: Dict[str, Any] = field(default_factory=dict)


class PerformanceTestRunner:
    """Automated performance test runner with benchmarking capabilities"""
    
    def __init__(self, metrics_collector: PerformanceMetricsCollector = None):
        self.metrics_collector = metrics_collector or performance_metrics_collector
        self.test_configs: Dict[str, PerformanceTestConfig] = {}
        self.test_results: List[PerformanceTestResult] = []
        self.baseline_results: Dict[str, PerformanceTestResult] = {}
        self.running_tests: Dict[str, asyncio.Task] = {}
        self.max_results_history = 1000
        
        # Set up default test configurations
        self._setup_default_tests()
    
    def _setup_default_tests(self):
        """Set up default performance test configurations"""
        default_tests = [
            PerformanceTestConfig(
                name="agent_baseline_performance",
                test_type=TestType.BASELINE,
                duration_seconds=30,
                concurrent_users=1,
                target_rps=0.5,
                max_response_time_ms=30000,
                tags=["agent", "baseline"],
            ),
            PerformanceTestConfig(
                name="agent_load_test",
                test_type=TestType.LOAD,
                duration_seconds=120,
                concurrent_users=5,
                ramp_up_seconds=30,
                target_rps=2.0,
                max_response_time_ms=45000,
                tags=["agent", "load"],
            ),
            PerformanceTestConfig(
                name="tool_execution_performance",
                test_type=TestType.BASELINE,
                duration_seconds=60,
                concurrent_users=3,
                target_rps=1.0,
                max_response_time_ms=15000,
                tags=["tools", "baseline"],
            ),
            PerformanceTestConfig(
                name="concurrent_request_stress",
                test_type=TestType.STRESS,
                duration_seconds=180,
                concurrent_users=10,
                ramp_up_seconds=60,
                target_rps=5.0,
                max_response_time_ms=60000,
                max_error_rate_percent=10.0,
                tags=["stress", "concurrent"],
            ),
            PerformanceTestConfig(
                name="memory_endurance_test",
                test_type=TestType.ENDURANCE,
                duration_seconds=600,  # 10 minutes
                concurrent_users=2,
                target_rps=0.5,
                max_response_time_ms=30000,
                tags=["endurance", "memory"],
            ),
        ]
        
        for test_config in default_tests:
            self.test_configs[test_config.name] = test_config
    
    async def run_test(
        self, 
        test_name: str, 
        test_function: Optional[Callable] = None
    ) -> PerformanceTestResult:
        """Run a specific performance test"""
        if test_name not in self.test_configs:
            raise ValueError(f"Test configuration '{test_name}' not found")
        
        config = self.test_configs[test_name]
        
        if not config.enabled:
            logger.info(f"Test '{test_name}' is disabled, skipping")
            return self._create_skipped_result(test_name, config)
        
        logger.info(f"Starting performance test: {test_name}")
        
        result = PerformanceTestResult(
            test_name=test_name,
            test_type=config.test_type,
            status=TestStatus.RUNNING,
            start_time=datetime.now(timezone.utc),
        )
        
        try:
            # Warmup phase
            if config.warmup_requests > 0:
                logger.info(f"Warming up with {config.warmup_requests} requests...")
                await self._run_warmup(config, test_function)
            
            # Main test execution
            await self._execute_test(config, result, test_function)
            
            # Cooldown phase
            if config.cooldown_seconds > 0:
                logger.info(f"Cooling down for {config.cooldown_seconds} seconds...")
                await asyncio.sleep(config.cooldown_seconds)
            
            # Calculate final metrics
            self._calculate_test_metrics(result, config)
            
            result.status = TestStatus.COMPLETED
            result.end_time = datetime.now(timezone.utc)
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            
            # Record metrics
            self._record_test_metrics(result)
            
            logger.info(
                f"Performance test '{test_name}' completed",
                {
                    "duration_seconds": result.duration_seconds,
                    "total_requests": result.total_requests,
                    "avg_response_time_ms": result.avg_response_time_ms,
                    "requests_per_second": result.requests_per_second,
                    "error_rate_percent": result.error_rate_percent,
                    "performance_level": result.performance_level.value,
                    "meets_targets": result.meets_targets,
                }
            )
            
        except Exception as e:
            logger.error(f"Performance test '{test_name}' failed: {e}")
            result.status = TestStatus.FAILED
            result.errors.append(str(e))
            result.end_time = datetime.now(timezone.utc)
        
        # Store result
        self.test_results.append(result)
        if len(self.test_results) > self.max_results_history:
            self.test_results = self.test_results[-self.max_results_history:]
        
        return result
    
    async def _run_warmup(
        self, 
        config: PerformanceTestConfig, 
        test_function: Optional[Callable]
    ):
        """Run warmup requests before the main test"""
        for i in range(config.warmup_requests):
            try:
                if test_function:
                    await test_function()
                else:
                    await self._default_test_function()
                
                # Small delay between warmup requests
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Warmup request {i+1} failed: {e}")
    
    async def _execute_test(
        self,
        config: PerformanceTestConfig,
        result: PerformanceTestResult,
        test_function: Optional[Callable]
    ):
        """Execute the main performance test"""
        start_time = time.time()
        end_time = start_time + config.duration_seconds
        
        # Create worker tasks
        tasks = []
        for worker_id in range(config.concurrent_users):
            task = asyncio.create_task(
                self._test_worker(
                    worker_id, config, result, test_function, start_time, end_time
                )
            )
            tasks.append(task)
        
        # Handle ramp-up
        if config.ramp_up_seconds > 0:
            ramp_interval = config.ramp_up_seconds / config.concurrent_users
            for i, task in enumerate(tasks):
                if i > 0:  # Don't delay the first worker
                    await asyncio.sleep(ramp_interval)
        
        # Wait for all workers to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _test_worker(
        self,
        worker_id: int,
        config: PerformanceTestConfig,
        result: PerformanceTestResult,
        test_function: Optional[Callable],
        start_time: float,
        end_time: float
    ):
        """Individual test worker that executes requests"""
        worker_requests = 0
        worker_errors = 0
        
        # Calculate request interval for target RPS
        request_interval = config.concurrent_users / config.target_rps if config.target_rps > 0 else 1.0
        
        while time.time() < end_time:
            request_start = time.time()
            
            try:
                if test_function:
                    await test_function()
                else:
                    await self._default_test_function()
                
                # Record successful request
                response_time_ms = (time.time() - request_start) * 1000
                result.response_times.append(response_time_ms)
                result.timestamps.append(datetime.now(timezone.utc))
                worker_requests += 1
                
            except Exception as e:
                worker_errors += 1
                result.errors.append(f"Worker {worker_id}: {str(e)}")
                logger.debug(f"Test worker {worker_id} request failed: {e}")
            
            # Rate limiting
            elapsed = time.time() - request_start
            if elapsed < request_interval:
                await asyncio.sleep(request_interval - elapsed)
        
        # Update result counters (thread-safe)
        result.total_requests += worker_requests
        result.successful_requests += worker_requests - worker_errors
        result.failed_requests += worker_errors
    
    async def _default_test_function(self):
        """Default test function that simulates basic agent operation"""
        # Simulate some processing time
        await asyncio.sleep(0.1 + (time.time() % 0.1))  # 0.1-0.2 seconds
        
        # Simulate occasional failures
        if time.time() % 100 < 2:  # 2% failure rate
            raise Exception("Simulated test failure")
    
    def _calculate_test_metrics(
        self, 
        result: PerformanceTestResult, 
        config: PerformanceTestConfig
    ):
        """Calculate final test metrics"""
        if not result.response_times:
            return
        
        # Response time statistics
        result.avg_response_time_ms = statistics.mean(result.response_times)
        result.min_response_time_ms = min(result.response_times)
        result.max_response_time_ms = max(result.response_times)
        
        # Percentiles
        sorted_times = sorted(result.response_times)
        result.p50_response_time_ms = self._percentile(sorted_times, 50)
        result.p95_response_time_ms = self._percentile(sorted_times, 95)
        result.p99_response_time_ms = self._percentile(sorted_times, 99)
        
        # Throughput
        if result.duration_seconds > 0:
            result.requests_per_second = result.total_requests / result.duration_seconds
        
        # Error rate
        if result.total_requests > 0:
            result.error_rate_percent = (result.failed_requests / result.total_requests) * 100
        
        # Performance level assessment
        result.performance_level = self._assess_performance_level(result, config)
        
        # Target compliance
        result.meets_targets = (
            result.avg_response_time_ms <= config.max_response_time_ms and
            result.requests_per_second >= config.target_rps * 0.8 and  # 80% of target
            result.error_rate_percent <= config.max_error_rate_percent
        )
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        index = int((percentile / 100) * len(values))
        return values[min(index, len(values) - 1)]
    
    def _assess_performance_level(
        self, 
        result: PerformanceTestResult, 
        config: PerformanceTestConfig
    ) -> PerformanceLevel:
        """Assess overall performance level based on metrics"""
        # Response time assessment
        response_time_ratio = result.avg_response_time_ms / config.max_response_time_ms
        
        # Throughput assessment
        throughput_ratio = result.requests_per_second / config.target_rps if config.target_rps > 0 else 1.0
        
        # Error rate assessment
        error_ratio = result.error_rate_percent / config.max_error_rate_percent if config.max_error_rate_percent > 0 else 0
        
        # Overall assessment
        if error_ratio > 1.0 or response_time_ratio > 1.5:
            return PerformanceLevel.CRITICAL
        elif error_ratio > 0.8 or response_time_ratio > 1.2 or throughput_ratio < 0.6:
            return PerformanceLevel.POOR
        elif error_ratio > 0.5 or response_time_ratio > 1.0 or throughput_ratio < 0.8:
            return PerformanceLevel.ACCEPTABLE
        elif throughput_ratio >= 1.0 and response_time_ratio <= 0.8 and error_ratio <= 0.2:
            return PerformanceLevel.EXCELLENT
        else:
            return PerformanceLevel.GOOD
    
    def _record_test_metrics(self, result: PerformanceTestResult):
        """Record test metrics in the metrics collector"""
        test_name = result.test_name
        
        # Record response time
        self.metrics_collector.record_response_time(
            f"test.{test_name}", 
            result.avg_response_time_ms,
            {"test_type": result.test_type.value}
        )
        
        # Record throughput
        self.metrics_collector.record_throughput(
            f"test.{test_name}", 
            result.total_requests,
            {"test_type": result.test_type.value}
        )
        
        # Record error rate
        self.metrics_collector.record_error_rate(
            f"test.{test_name}", 
            result.error_rate_percent,
            {"test_type": result.test_type.value}
        )
    
    def _create_skipped_result(
        self, 
        test_name: str, 
        config: PerformanceTestConfig
    ) -> PerformanceTestResult:
        """Create a result for a skipped test"""
        return PerformanceTestResult(
            test_name=test_name,
            test_type=config.test_type,
            status=TestStatus.CANCELLED,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            metadata={"reason": "Test disabled"}
        )
    
    async def run_all_tests(
        self, 
        test_function: Optional[Callable] = None,
        test_filter: Optional[List[str]] = None
    ) -> List[PerformanceTestResult]:
        """Run all configured performance tests"""
        results = []
        
        # Filter tests if specified
        tests_to_run = test_filter or list(self.test_configs.keys())
        
        for test_name in tests_to_run:
            if test_name in self.test_configs:
                try:
                    result = await self.run_test(test_name, test_function)
                    results.append(result)
                    
                    # Brief pause between tests
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    logger.error(f"Failed to run test '{test_name}': {e}")
        
        return results
    
    async def run_regression_test(
        self, 
        test_name: str, 
        test_function: Optional[Callable] = None,
        regression_threshold_percent: float = 10.0
    ) -> RegressionTestResult:
        """Run regression test against baseline"""
        if test_name not in self.baseline_results:
            raise ValueError(f"No baseline result found for test '{test_name}'")
        
        # Run current test
        current_result = await self.run_test(test_name, test_function)
        baseline_result = self.baseline_results[test_name]
        
        # Compare results
        response_time_change = self._calculate_percentage_change(
            baseline_result.avg_response_time_ms,
            current_result.avg_response_time_ms
        )
        
        throughput_change = self._calculate_percentage_change(
            baseline_result.requests_per_second,
            current_result.requests_per_second
        )
        
        error_rate_change = self._calculate_percentage_change(
            baseline_result.error_rate_percent,
            current_result.error_rate_percent
        )
        
        # Overall performance change (weighted)
        performance_change = (
            response_time_change * 0.4 +  # Response time weight
            -throughput_change * 0.4 +    # Throughput weight (negative because higher is better)
            error_rate_change * 0.2       # Error rate weight
        )
        
        # Detect regression
        regression_detected = (
            abs(performance_change) > regression_threshold_percent or
            response_time_change > regression_threshold_percent or
            throughput_change < -regression_threshold_percent or
            error_rate_change > regression_threshold_percent
        )
        
        regression_result = RegressionTestResult(
            test_name=test_name,
            baseline_result=baseline_result,
            current_result=current_result,
            regression_detected=regression_detected,
            performance_change_percent=performance_change,
            response_time_change_percent=response_time_change,
            throughput_change_percent=throughput_change,
            error_rate_change_percent=error_rate_change,
            regression_threshold_percent=regression_threshold_percent,
            details={
                "baseline_timestamp": baseline_result.start_time.isoformat(),
                "current_timestamp": current_result.start_time.isoformat(),
                "comparison_metrics": {
                    "response_time": {
                        "baseline": baseline_result.avg_response_time_ms,
                        "current": current_result.avg_response_time_ms,
                        "change_percent": response_time_change,
                    },
                    "throughput": {
                        "baseline": baseline_result.requests_per_second,
                        "current": current_result.requests_per_second,
                        "change_percent": throughput_change,
                    },
                    "error_rate": {
                        "baseline": baseline_result.error_rate_percent,
                        "current": current_result.error_rate_percent,
                        "change_percent": error_rate_change,
                    },
                }
            }
        )
        
        if regression_detected:
            logger.warning(
                f"Performance regression detected in test '{test_name}'",
                {
                    "performance_change_percent": performance_change,
                    "response_time_change_percent": response_time_change,
                    "throughput_change_percent": throughput_change,
                    "error_rate_change_percent": error_rate_change,
                }
            )
        
        return regression_result
    
    def _calculate_percentage_change(self, baseline: float, current: float) -> float:
        """Calculate percentage change from baseline to current"""
        if baseline == 0:
            return 0.0 if current == 0 else 100.0
        return ((current - baseline) / baseline) * 100
    
    def set_baseline(self, test_name: str, result: PerformanceTestResult):
        """Set baseline result for regression testing"""
        self.baseline_results[test_name] = result
        logger.info(f"Set baseline for test '{test_name}'")
    
    def get_test_history(
        self, 
        test_name: str, 
        limit: int = 10
    ) -> List[PerformanceTestResult]:
        """Get test execution history"""
        test_results = [
            result for result in self.test_results 
            if result.test_name == test_name
        ]
        return sorted(test_results, key=lambda x: x.start_time, reverse=True)[:limit]
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all test results"""
        if not self.test_results:
            return {"total_tests": 0, "summary": "No tests executed"}
        
        # Group by test name
        tests_by_name = {}
        for result in self.test_results:
            if result.test_name not in tests_by_name:
                tests_by_name[result.test_name] = []
            tests_by_name[result.test_name].append(result)
        
        # Calculate summary statistics
        summary = {
            "total_tests_executed": len(self.test_results),
            "unique_tests": len(tests_by_name),
            "test_summaries": {},
            "overall_performance": "good",
        }
        
        performance_levels = []
        
        for test_name, results in tests_by_name.items():
            latest_result = max(results, key=lambda x: x.start_time)
            
            summary["test_summaries"][test_name] = {
                "latest_result": asdict(latest_result),
                "execution_count": len(results),
                "success_rate": len([r for r in results if r.status == TestStatus.COMPLETED]) / len(results),
                "avg_response_time_ms": statistics.mean([r.avg_response_time_ms for r in results if r.avg_response_time_ms > 0]),
                "avg_throughput_rps": statistics.mean([r.requests_per_second for r in results if r.requests_per_second > 0]),
            }
            
            performance_levels.append(latest_result.performance_level)
        
        # Determine overall performance
        if any(level == PerformanceLevel.CRITICAL for level in performance_levels):
            summary["overall_performance"] = "critical"
        elif any(level == PerformanceLevel.POOR for level in performance_levels):
            summary["overall_performance"] = "poor"
        elif any(level == PerformanceLevel.ACCEPTABLE for level in performance_levels):
            summary["overall_performance"] = "acceptable"
        elif all(level == PerformanceLevel.EXCELLENT for level in performance_levels):
            summary["overall_performance"] = "excellent"
        
        return summary
    
    def add_test_config(self, config: PerformanceTestConfig):
        """Add a new test configuration"""
        self.test_configs[config.name] = config
        logger.info(f"Added performance test configuration: {config.name}")
    
    def remove_test_config(self, test_name: str):
        """Remove a test configuration"""
        if test_name in self.test_configs:
            del self.test_configs[test_name]
            logger.info(f"Removed performance test configuration: {test_name}")


# Global performance test runner instance
performance_test_runner = PerformanceTestRunner()


# Convenience functions
async def run_performance_test(
    test_name: str, 
    test_function: Optional[Callable] = None
) -> PerformanceTestResult:
    """Run a specific performance test"""
    return await performance_test_runner.run_test(test_name, test_function)


async def run_all_performance_tests(
    test_function: Optional[Callable] = None,
    test_filter: Optional[List[str]] = None
) -> List[PerformanceTestResult]:
    """Run all configured performance tests"""
    return await performance_test_runner.run_all_tests(test_function, test_filter)


async def run_regression_test(
    test_name: str,
    test_function: Optional[Callable] = None,
    regression_threshold_percent: float = 10.0
) -> RegressionTestResult:
    """Run regression test against baseline"""
    return await performance_test_runner.run_regression_test(
        test_name, test_function, regression_threshold_percent
    )


def set_performance_baseline(test_name: str, result: PerformanceTestResult):
    """Set baseline result for regression testing"""
    performance_test_runner.set_baseline(test_name, result)


def get_performance_test_summary() -> Dict[str, Any]:
    """Get summary of all performance test results"""
    return performance_test_runner.get_test_summary()