"""
Comprehensive load testing implementation for OpenManus performance validation.

This module implements load testing scenarios with varying request patterns,
stress testing for system breaking points, and performance benchmarking
with baseline metrics.
"""

import asyncio
import gc
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, patch

import psutil
import pytest

from app.agent.manus import Manus
from app.config import Config
from app.exceptions import OpenManusError
from app.schema import AgentResponse
from tests.base import PerformanceTestCase


@dataclass
class LoadTestMetrics:
    """Metrics collected during load testing."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    memory_peak_mb: float
    memory_delta_mb: float
    cpu_avg_percent: float
    error_rate: float


@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios."""

    concurrent_users: int
    requests_per_user: int
    ramp_up_time: float
    test_duration: float
    request_delay: float
    timeout: float


class LoadTestRunner:
    """Load test runner with comprehensive metrics collection."""

    def __init__(self, agent: Manus):
        self.agent = agent
        self.metrics = []
        self.response_times = []
        self.errors = []
        self.start_time = None
        self.end_time = None
        self.memory_samples = []
        self.cpu_samples = []

    async def run_load_test(
        self, config: LoadTestConfig, test_scenarios: List[str]
    ) -> LoadTestMetrics:
        """Run a comprehensive load test with the given configuration."""
        self.start_time = time.time()
        start_memory = psutil.virtual_memory().used

        # Start resource monitoring
        monitor_task = asyncio.create_task(self._monitor_resources())

        try:
            # Execute load test
            await self._execute_load_test(config, test_scenarios)
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        self.end_time = time.time()
        end_memory = psutil.virtual_memory().used

        # Calculate metrics
        return self._calculate_metrics(start_memory, end_memory)

    async def _execute_load_test(
        self, config: LoadTestConfig, test_scenarios: List[str]
    ):
        """Execute the actual load test."""
        # Create user simulation tasks
        user_tasks = []

        for user_id in range(config.concurrent_users):
            # Stagger user start times for ramp-up
            start_delay = (config.ramp_up_time / config.concurrent_users) * user_id

            task = asyncio.create_task(
                self._simulate_user(user_id, config, test_scenarios, start_delay)
            )
            user_tasks.append(task)

        # Wait for all users to complete
        await asyncio.gather(*user_tasks, return_exceptions=True)

    async def _simulate_user(
        self,
        user_id: int,
        config: LoadTestConfig,
        test_scenarios: List[str],
        start_delay: float,
    ):
        """Simulate a single user's behavior."""
        # Wait for ramp-up delay
        if start_delay > 0:
            await asyncio.sleep(start_delay)

        # Execute requests for this user
        for request_num in range(config.requests_per_user):
            scenario = test_scenarios[request_num % len(test_scenarios)]

            request_start = time.time()

            try:
                # Execute request with timeout
                response = await asyncio.wait_for(
                    self.agent.run(scenario), timeout=config.timeout
                )

                request_end = time.time()
                response_time = request_end - request_start

                self.response_times.append(response_time)

                # Validate response
                if isinstance(response, AgentResponse) and response.success:
                    self.metrics.append(
                        {
                            "user_id": user_id,
                            "request_num": request_num,
                            "response_time": response_time,
                            "success": True,
                            "timestamp": request_start,
                        }
                    )
                else:
                    self.errors.append(
                        {
                            "user_id": user_id,
                            "request_num": request_num,
                            "error": "Invalid response",
                            "timestamp": request_start,
                        }
                    )

            except asyncio.TimeoutError:
                self.errors.append(
                    {
                        "user_id": user_id,
                        "request_num": request_num,
                        "error": "Timeout",
                        "timestamp": request_start,
                    }
                )
            except Exception as e:
                self.errors.append(
                    {
                        "user_id": user_id,
                        "request_num": request_num,
                        "error": str(e),
                        "timestamp": request_start,
                    }
                )

            # Add delay between requests
            if config.request_delay > 0:
                await asyncio.sleep(config.request_delay)

    async def _monitor_resources(self):
        """Monitor system resources during the test."""
        while True:
            try:
                memory_mb = psutil.virtual_memory().used / (1024 * 1024)
                cpu_percent = psutil.cpu_percent()

                self.memory_samples.append(memory_mb)
                self.cpu_samples.append(cpu_percent)

                await asyncio.sleep(1.0)  # Sample every second
            except asyncio.CancelledError:
                break

    def _calculate_metrics(self, start_memory: int, end_memory: int) -> LoadTestMetrics:
        """Calculate comprehensive load test metrics."""
        total_requests = len(self.metrics) + len(self.errors)
        successful_requests = len(self.metrics)
        failed_requests = len(self.errors)

        total_time = self.end_time - self.start_time if self.end_time else 0

        if self.response_times:
            self.response_times.sort()
            avg_response_time = statistics.mean(self.response_times)
            min_response_time = min(self.response_times)
            max_response_time = max(self.response_times)

            # Calculate percentiles
            n = len(self.response_times)
            p50_idx = int(0.50 * n)
            p95_idx = int(0.95 * n)
            p99_idx = int(0.99 * n)

            p50_response_time = self.response_times[min(p50_idx, n - 1)]
            p95_response_time = self.response_times[min(p95_idx, n - 1)]
            p99_response_time = self.response_times[min(p99_idx, n - 1)]
        else:
            avg_response_time = 0
            min_response_time = 0
            max_response_time = 0
            p50_response_time = 0
            p95_response_time = 0
            p99_response_time = 0

        requests_per_second = total_requests / total_time if total_time > 0 else 0

        memory_peak_mb = max(self.memory_samples) if self.memory_samples else 0
        memory_delta_mb = (end_memory - start_memory) / (1024 * 1024)

        cpu_avg_percent = statistics.mean(self.cpu_samples) if self.cpu_samples else 0

        error_rate = failed_requests / total_requests if total_requests > 0 else 0

        return LoadTestMetrics(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            memory_peak_mb=memory_peak_mb,
            memory_delta_mb=memory_delta_mb,
            cpu_avg_percent=cpu_avg_percent,
            error_rate=error_rate,
        )


@pytest.mark.performance
@pytest.mark.slow
class TestLoadTesting(PerformanceTestCase):
    """Comprehensive load testing scenarios."""

    @pytest.fixture
    def load_test_config(self, test_config):
        """Configuration optimized for load testing."""
        config = test_config.copy()
        config.update(
            {
                "max_concurrent_requests": 50,
                "request_timeout": 60,
                "memory_limit": "2GB",
                "performance_monitoring": True,
                "enable_caching": True,
            }
        )
        return config

    @pytest.fixture
    async def load_test_agent(self, mock_llm_client, load_test_config):
        """Agent configured for load testing."""
        with patch("app.config.Config._instance", None):
            config = Config()
            config._config = load_test_config

            with patch("app.agent.manus.LLMClient") as mock_llm_class:
                mock_llm_class.return_value = mock_llm_client
                agent = Manus()
                return agent

    @pytest.fixture
    def basic_test_scenarios(self) -> List[str]:
        """Basic test scenarios for load testing."""
        return [
            "What is 2 + 2?",
            "List three colors",
            "Count from 1 to 5",
            "What is the capital of France?",
            "Say hello world",
            "Calculate 10 * 5",
            "Name a programming language",
            "What day comes after Monday?",
            "How many letters in 'hello'?",
            "What is 100 divided by 4?",
        ]

    @pytest.fixture
    def complex_test_scenarios(self) -> List[str]:
        """Complex test scenarios for load testing."""
        return [
            "Create a Python function to calculate fibonacci numbers",
            "Explain the concept of machine learning in simple terms",
            "Write a sorting algorithm and explain how it works",
            "Analyze the pros and cons of different database types",
            "Generate a report on renewable energy trends",
            "Create a web scraping script for news articles",
            "Explain blockchain technology and its applications",
            "Design a REST API for a todo application",
            "Write a comprehensive guide on Python testing",
            "Create a data visualization for sales metrics",
        ]

    async def test_light_load_baseline(
        self, load_test_agent, mock_llm_client, basic_test_scenarios
    ):
        """Test system performance under light load to establish baseline."""
        # Setup mock responses
        mock_llm_client.set_responses(
            [f"Response to: {scenario}" for scenario in basic_test_scenarios * 5]
        )
        mock_llm_client.set_delay(0.1)  # 100ms simulated processing time

        config = LoadTestConfig(
            concurrent_users=5,
            requests_per_user=10,
            ramp_up_time=2.0,
            test_duration=30.0,
            request_delay=0.5,
            timeout=10.0,
        )

        runner = LoadTestRunner(load_test_agent)
        metrics = await runner.run_load_test(config, basic_test_scenarios)

        # Baseline performance assertions
        assert (
            metrics.error_rate < 0.05
        ), f"Error rate {metrics.error_rate:.2%} should be under 5%"
        assert (
            metrics.avg_response_time < 2.0
        ), f"Average response time {metrics.avg_response_time:.2f}s should be under 2s"
        assert (
            metrics.p95_response_time < 5.0
        ), f"P95 response time {metrics.p95_response_time:.2f}s should be under 5s"
        assert (
            metrics.requests_per_second > 2.0
        ), f"RPS {metrics.requests_per_second:.2f} should be above 2.0"
        assert (
            metrics.memory_delta_mb < 100
        ), f"Memory delta {metrics.memory_delta_mb:.2f}MB should be under 100MB"

        # Log baseline metrics for comparison
        print(f"\nBaseline Load Test Metrics:")
        print(f"  Total Requests: {metrics.total_requests}")
        print(f"  Success Rate: {(1 - metrics.error_rate):.2%}")
        print(f"  Avg Response Time: {metrics.avg_response_time:.3f}s")
        print(f"  P95 Response Time: {metrics.p95_response_time:.3f}s")
        print(f"  Requests/Second: {metrics.requests_per_second:.2f}")
        print(f"  Memory Delta: {metrics.memory_delta_mb:.2f}MB")

    async def test_moderate_load_performance(
        self, load_test_agent, mock_llm_client, basic_test_scenarios
    ):
        """Test system performance under moderate load."""
        # Setup mock responses with slight delay
        mock_llm_client.set_responses(
            [f"Response to: {scenario}" for scenario in basic_test_scenarios * 10]
        )
        mock_llm_client.set_delay(0.2)  # 200ms simulated processing time

        config = LoadTestConfig(
            concurrent_users=15,
            requests_per_user=10,
            ramp_up_time=5.0,
            test_duration=60.0,
            request_delay=0.3,
            timeout=15.0,
        )

        runner = LoadTestRunner(load_test_agent)
        metrics = await runner.run_load_test(config, basic_test_scenarios)

        # Moderate load performance assertions
        assert (
            metrics.error_rate < 0.10
        ), f"Error rate {metrics.error_rate:.2%} should be under 10%"
        assert (
            metrics.avg_response_time < 5.0
        ), f"Average response time {metrics.avg_response_time:.2f}s should be under 5s"
        assert (
            metrics.p95_response_time < 10.0
        ), f"P95 response time {metrics.p95_response_time:.2f}s should be under 10s"
        assert (
            metrics.requests_per_second > 1.5
        ), f"RPS {metrics.requests_per_second:.2f} should be above 1.5"
        assert (
            metrics.memory_delta_mb < 200
        ), f"Memory delta {metrics.memory_delta_mb:.2f}MB should be under 200MB"
        assert (
            metrics.cpu_avg_percent < 80
        ), f"CPU usage {metrics.cpu_avg_percent:.1f}% should be under 80%"

        print(f"\nModerate Load Test Metrics:")
        print(f"  Total Requests: {metrics.total_requests}")
        print(f"  Success Rate: {(1 - metrics.error_rate):.2%}")
        print(f"  Avg Response Time: {metrics.avg_response_time:.3f}s")
        print(f"  P95 Response Time: {metrics.p95_response_time:.3f}s")
        print(f"  Requests/Second: {metrics.requests_per_second:.2f}")
        print(f"  Memory Delta: {metrics.memory_delta_mb:.2f}MB")
        print(f"  CPU Usage: {metrics.cpu_avg_percent:.1f}%")

    async def test_high_load_stress(
        self, load_test_agent, mock_llm_client, complex_test_scenarios
    ):
        """Test system performance under high load with complex scenarios."""
        # Setup mock responses for complex scenarios
        mock_llm_client.set_responses(
            [
                f"Complex response to: {scenario}"
                for scenario in complex_test_scenarios * 5
            ]
        )
        mock_llm_client.set_delay(0.5)  # 500ms simulated processing time

        config = LoadTestConfig(
            concurrent_users=25,
            requests_per_user=8,
            ramp_up_time=10.0,
            test_duration=120.0,
            request_delay=0.2,
            timeout=30.0,
        )

        runner = LoadTestRunner(load_test_agent)
        metrics = await runner.run_load_test(config, complex_test_scenarios)

        # High load stress assertions (more lenient)
        assert (
            metrics.error_rate < 0.20
        ), f"Error rate {metrics.error_rate:.2%} should be under 20%"
        assert (
            metrics.avg_response_time < 10.0
        ), f"Average response time {metrics.avg_response_time:.2f}s should be under 10s"
        assert (
            metrics.p95_response_time < 25.0
        ), f"P95 response time {metrics.p95_response_time:.2f}s should be under 25s"
        assert (
            metrics.requests_per_second > 0.8
        ), f"RPS {metrics.requests_per_second:.2f} should be above 0.8"
        assert (
            metrics.memory_delta_mb < 500
        ), f"Memory delta {metrics.memory_delta_mb:.2f}MB should be under 500MB"

        print(f"\nHigh Load Stress Test Metrics:")
        print(f"  Total Requests: {metrics.total_requests}")
        print(f"  Success Rate: {(1 - metrics.error_rate):.2%}")
        print(f"  Avg Response Time: {metrics.avg_response_time:.3f}s")
        print(f"  P95 Response Time: {metrics.p95_response_time:.3f}s")
        print(f"  Requests/Second: {metrics.requests_per_second:.2f}")
        print(f"  Memory Delta: {metrics.memory_delta_mb:.2f}MB")
        print(f"  CPU Usage: {metrics.cpu_avg_percent:.1f}%")

    async def test_burst_load_pattern(
        self, load_test_agent, mock_llm_client, basic_test_scenarios
    ):
        """Test system performance under burst load patterns."""
        # Setup mock responses
        mock_llm_client.set_responses(
            [f"Burst response: {scenario}" for scenario in basic_test_scenarios * 8]
        )
        mock_llm_client.set_delay(0.15)

        # Simulate burst pattern: quick ramp-up, sustained load, then ramp-down
        config = LoadTestConfig(
            concurrent_users=30,
            requests_per_user=5,
            ramp_up_time=2.0,  # Quick ramp-up
            test_duration=45.0,
            request_delay=0.1,  # Minimal delay for burst effect
            timeout=20.0,
        )

        runner = LoadTestRunner(load_test_agent)
        metrics = await runner.run_load_test(config, basic_test_scenarios)

        # Burst load assertions
        assert (
            metrics.error_rate < 0.15
        ), f"Error rate {metrics.error_rate:.2%} should be under 15%"
        assert (
            metrics.avg_response_time < 8.0
        ), f"Average response time {metrics.avg_response_time:.2f}s should be under 8s"
        assert (
            metrics.p99_response_time < 20.0
        ), f"P99 response time {metrics.p99_response_time:.2f}s should be under 20s"
        assert (
            metrics.requests_per_second > 1.0
        ), f"RPS {metrics.requests_per_second:.2f} should be above 1.0"

        print(f"\nBurst Load Test Metrics:")
        print(f"  Total Requests: {metrics.total_requests}")
        print(f"  Success Rate: {(1 - metrics.error_rate):.2%}")
        print(f"  Avg Response Time: {metrics.avg_response_time:.3f}s")
        print(f"  P99 Response Time: {metrics.p99_response_time:.3f}s")
        print(f"  Requests/Second: {metrics.requests_per_second:.2f}")

    async def test_sustained_load_endurance(
        self, load_test_agent, mock_llm_client, basic_test_scenarios
    ):
        """Test system endurance under sustained load over extended period."""
        # Setup mock responses for sustained test
        mock_llm_client.set_responses(
            [
                f"Sustained response: {scenario}"
                for scenario in basic_test_scenarios * 20
            ]
        )
        mock_llm_client.set_delay(0.3)

        config = LoadTestConfig(
            concurrent_users=10,
            requests_per_user=20,
            ramp_up_time=5.0,
            test_duration=180.0,  # 3 minutes sustained
            request_delay=0.5,
            timeout=25.0,
        )

        runner = LoadTestRunner(load_test_agent)

        # Monitor for memory leaks and performance degradation
        initial_memory = psutil.virtual_memory().used

        metrics = await runner.run_load_test(config, basic_test_scenarios)

        final_memory = psutil.virtual_memory().used
        memory_growth = (final_memory - initial_memory) / (1024 * 1024)

        # Endurance test assertions
        assert (
            metrics.error_rate < 0.10
        ), f"Error rate {metrics.error_rate:.2%} should be under 10%"
        assert (
            metrics.avg_response_time < 6.0
        ), f"Average response time {metrics.avg_response_time:.2f}s should be under 6s"
        assert (
            memory_growth < 150
        ), f"Memory growth {memory_growth:.2f}MB should be under 150MB"

        # Check for performance degradation over time
        early_responses = runner.response_times[: len(runner.response_times) // 3]
        late_responses = runner.response_times[-len(runner.response_times) // 3 :]

        if early_responses and late_responses:
            early_avg = statistics.mean(early_responses)
            late_avg = statistics.mean(late_responses)
            degradation = (late_avg - early_avg) / early_avg if early_avg > 0 else 0

            assert (
                degradation < 0.5
            ), f"Performance degradation {degradation:.2%} should be under 50%"

        print(f"\nSustained Load Endurance Test Metrics:")
        print(f"  Total Requests: {metrics.total_requests}")
        print(f"  Success Rate: {(1 - metrics.error_rate):.2%}")
        print(f"  Avg Response Time: {metrics.avg_response_time:.3f}s")
        print(f"  Memory Growth: {memory_growth:.2f}MB")
        print(f"  Test Duration: {metrics.total_time:.1f}s")

    async def test_mixed_workload_pattern(
        self,
        load_test_agent,
        mock_llm_client,
        basic_test_scenarios,
        complex_test_scenarios,
    ):
        """Test system performance with mixed workload patterns."""
        # Combine basic and complex scenarios
        mixed_scenarios = basic_test_scenarios[:5] + complex_test_scenarios[:5]

        # Setup mock responses
        mock_llm_client.set_responses(
            [f"Mixed response: {scenario}" for scenario in mixed_scenarios * 6]
        )
        mock_llm_client.set_delay(0.25)

        config = LoadTestConfig(
            concurrent_users=20,
            requests_per_user=6,
            ramp_up_time=8.0,
            test_duration=90.0,
            request_delay=0.4,
            timeout=20.0,
        )

        runner = LoadTestRunner(load_test_agent)
        metrics = await runner.run_load_test(config, mixed_scenarios)

        # Mixed workload assertions
        assert (
            metrics.error_rate < 0.15
        ), f"Error rate {metrics.error_rate:.2%} should be under 15%"
        assert (
            metrics.avg_response_time < 7.0
        ), f"Average response time {metrics.avg_response_time:.2f}s should be under 7s"
        assert (
            metrics.p95_response_time < 15.0
        ), f"P95 response time {metrics.p95_response_time:.2f}s should be under 15s"
        assert (
            metrics.requests_per_second > 1.0
        ), f"RPS {metrics.requests_per_second:.2f} should be above 1.0"

        print(f"\nMixed Workload Test Metrics:")
        print(f"  Total Requests: {metrics.total_requests}")
        print(f"  Success Rate: {(1 - metrics.error_rate):.2%}")
        print(f"  Avg Response Time: {metrics.avg_response_time:.3f}s")
        print(f"  P95 Response Time: {metrics.p95_response_time:.3f}s")
        print(f"  Requests/Second: {metrics.requests_per_second:.2f}")

    async def test_error_recovery_under_load(
        self, load_test_agent, mock_llm_client, basic_test_scenarios
    ):
        """Test system error recovery capabilities under load."""
        # Setup mixed responses with some errors
        responses = []
        errors = []

        for i, scenario in enumerate(basic_test_scenarios * 8):
            if i % 4 == 0:  # 25% error rate
                errors.append(OpenManusError(f"Simulated error for: {scenario}"))
                responses.append("Error response")
            else:
                errors.append(None)
                responses.append(f"Success response: {scenario}")

        mock_llm_client.set_responses(responses)
        mock_llm_client.set_errors(errors)
        mock_llm_client.set_delay(0.2)

        config = LoadTestConfig(
            concurrent_users=15,
            requests_per_user=8,
            ramp_up_time=3.0,
            test_duration=60.0,
            request_delay=0.3,
            timeout=15.0,
        )

        runner = LoadTestRunner(load_test_agent)
        metrics = await runner.run_load_test(config, basic_test_scenarios)

        # Error recovery assertions
        assert (
            metrics.error_rate < 0.40
        ), f"Error rate {metrics.error_rate:.2%} should be under 40%"
        assert (
            metrics.successful_requests > 0
        ), "Some requests should succeed despite errors"
        assert (
            metrics.total_time < 90
        ), f"Test should complete within reasonable time: {metrics.total_time:.1f}s"

        # Verify system didn't crash or hang
        assert (
            metrics.total_requests > 0
        ), "System should process requests despite errors"

        print(f"\nError Recovery Test Metrics:")
        print(f"  Total Requests: {metrics.total_requests}")
        print(f"  Successful Requests: {metrics.successful_requests}")
        print(f"  Failed Requests: {metrics.failed_requests}")
        print(f"  Error Rate: {metrics.error_rate:.2%}")
        print(f"  Avg Response Time: {metrics.avg_response_time:.3f}s")

    async def test_memory_pressure_load(self, load_test_agent, mock_llm_client):
        """Test system behavior under memory pressure during load."""
        # Create memory-intensive scenarios
        memory_scenarios = [
            "Generate a large document with 1000 words about artificial intelligence",
            "Create a comprehensive analysis of climate change with detailed statistics",
            "Write a complete tutorial on Python programming with code examples",
            "Generate a detailed report on global economic trends with data visualization",
            "Create an extensive guide on machine learning algorithms with implementations",
        ]

        # Setup responses that simulate memory usage
        mock_llm_client.set_responses(
            [
                f"Large response: {scenario} " + "x" * 1000
                for scenario in memory_scenarios * 4
            ]
        )
        mock_llm_client.set_delay(0.4)

        config = LoadTestConfig(
            concurrent_users=12,
            requests_per_user=5,
            ramp_up_time=4.0,
            test_duration=75.0,
            request_delay=0.6,
            timeout=25.0,
        )

        # Force garbage collection before test
        gc.collect()
        initial_memory = psutil.virtual_memory().used

        runner = LoadTestRunner(load_test_agent)
        metrics = await runner.run_load_test(config, memory_scenarios)

        # Force garbage collection after test
        gc.collect()
        final_memory = psutil.virtual_memory().used

        memory_retained = (final_memory - initial_memory) / (1024 * 1024)

        # Memory pressure assertions
        assert (
            metrics.error_rate < 0.20
        ), f"Error rate {metrics.error_rate:.2%} should be under 20%"
        assert metrics.memory_peak_mb > 0, "Memory usage should be tracked"
        assert (
            memory_retained < 200
        ), f"Retained memory {memory_retained:.2f}MB should be under 200MB"

        print(f"\nMemory Pressure Test Metrics:")
        print(f"  Total Requests: {metrics.total_requests}")
        print(f"  Success Rate: {(1 - metrics.error_rate):.2%}")
        print(f"  Peak Memory: {metrics.memory_peak_mb:.2f}MB")
        print(f"  Memory Delta: {metrics.memory_delta_mb:.2f}MB")
        print(f"  Retained Memory: {memory_retained:.2f}MB")

    @pytest.mark.slow
    async def test_breaking_point_analysis(
        self, load_test_agent, mock_llm_client, basic_test_scenarios
    ):
        """Test to find system breaking point by gradually increasing load."""
        # Setup mock responses
        mock_llm_client.set_responses(
            [
                f"Breaking point response: {scenario}"
                for scenario in basic_test_scenarios * 50
            ]
        )
        mock_llm_client.set_delay(0.1)

        breaking_point_found = False
        max_successful_users = 0

        # Gradually increase concurrent users to find breaking point
        for concurrent_users in [10, 20, 30, 40, 50]:
            config = LoadTestConfig(
                concurrent_users=concurrent_users,
                requests_per_user=5,
                ramp_up_time=2.0,
                test_duration=30.0,
                request_delay=0.2,
                timeout=10.0,
            )

            runner = LoadTestRunner(load_test_agent)
            metrics = await runner.run_load_test(config, basic_test_scenarios)

            print(f"\nBreaking Point Analysis - {concurrent_users} users:")
            print(f"  Success Rate: {(1 - metrics.error_rate):.2%}")
            print(f"  Avg Response Time: {metrics.avg_response_time:.3f}s")
            print(f"  Requests/Second: {metrics.requests_per_second:.2f}")

            # Check if system is still performing acceptably
            if metrics.error_rate < 0.10 and metrics.avg_response_time < 5.0:
                max_successful_users = concurrent_users
            else:
                breaking_point_found = True
                break

            # Small delay between tests
            await asyncio.sleep(2.0)

        # Breaking point analysis assertions
        assert (
            max_successful_users >= 10
        ), f"System should handle at least 10 concurrent users"

        if breaking_point_found:
            print(
                f"\nBreaking point found between {max_successful_users} and {concurrent_users} concurrent users"
            )
        else:
            print(
                f"\nSystem handled up to {max_successful_users} concurrent users successfully"
            )

        # Verify system can recover after high load
        recovery_config = LoadTestConfig(
            concurrent_users=5,
            requests_per_user=3,
            ramp_up_time=1.0,
            test_duration=15.0,
            request_delay=0.5,
            timeout=10.0,
        )

        recovery_runner = LoadTestRunner(load_test_agent)
        recovery_metrics = await recovery_runner.run_load_test(
            recovery_config, basic_test_scenarios[:3]
        )

        assert (
            recovery_metrics.error_rate < 0.10
        ), "System should recover after high load"
        print(
            f"\nRecovery Test - Success Rate: {(1 - recovery_metrics.error_rate):.2%}"
        )


@pytest.mark.performance
class TestPerformanceBenchmarking:
    """Performance benchmarking with baseline metrics."""

    @pytest.fixture
    def benchmark_scenarios(self) -> Dict[str, List[str]]:
        """Benchmark scenarios categorized by complexity."""
        return {
            "simple": [
                "What is 1 + 1?",
                "Say hello",
                "Count to 3",
            ],
            "medium": [
                "Explain what Python is",
                "List 5 programming languages",
                "What is machine learning?",
            ],
            "complex": [
                "Write a Python function to sort a list",
                "Explain the difference between SQL and NoSQL databases",
                "Create a simple web scraping example",
            ],
        }

    async def test_performance_baseline_establishment(
        self, load_test_agent, mock_llm_client, benchmark_scenarios
    ):
        """Establish performance baselines for different scenario types."""
        baselines = {}

        for complexity, scenarios in benchmark_scenarios.items():
            # Setup mock responses
            mock_llm_client.set_responses(
                [
                    f"Baseline {complexity} response: {scenario}"
                    for scenario in scenarios * 10
                ]
            )
            mock_llm_client.set_delay(
                0.1
                if complexity == "simple"
                else 0.3 if complexity == "medium" else 0.5
            )

            config = LoadTestConfig(
                concurrent_users=5,
                requests_per_user=10,
                ramp_up_time=1.0,
                test_duration=30.0,
                request_delay=0.2,
                timeout=15.0,
            )

            runner = LoadTestRunner(load_test_agent)
            metrics = await runner.run_load_test(config, scenarios)

            baselines[complexity] = {
                "avg_response_time": metrics.avg_response_time,
                "p95_response_time": metrics.p95_response_time,
                "requests_per_second": metrics.requests_per_second,
                "error_rate": metrics.error_rate,
                "memory_delta_mb": metrics.memory_delta_mb,
            }

            print(f"\n{complexity.title()} Scenario Baseline:")
            print(f"  Avg Response Time: {metrics.avg_response_time:.3f}s")
            print(f"  P95 Response Time: {metrics.p95_response_time:.3f}s")
            print(f"  Requests/Second: {metrics.requests_per_second:.2f}")
            print(f"  Error Rate: {metrics.error_rate:.2%}")
            print(f"  Memory Delta: {metrics.memory_delta_mb:.2f}MB")

        # Verify baseline expectations
        assert (
            baselines["simple"]["avg_response_time"]
            < baselines["medium"]["avg_response_time"]
        )
        assert (
            baselines["medium"]["avg_response_time"]
            < baselines["complex"]["avg_response_time"]
        )

        # All scenarios should have low error rates for baseline
        for complexity, baseline in baselines.items():
            assert (
                baseline["error_rate"] < 0.05
            ), f"{complexity} baseline error rate should be under 5%"

    async def test_performance_regression_detection(
        self, load_test_agent, mock_llm_client, benchmark_scenarios
    ):
        """Test performance regression detection against baselines."""
        # Simulate performance regression by adding delay
        mock_llm_client.set_responses(
            [
                f"Regression test response: {scenario}"
                for scenarios in benchmark_scenarios.values()
                for scenario in scenarios * 5
            ]
        )
        mock_llm_client.set_delay(0.8)  # Increased delay to simulate regression

        config = LoadTestConfig(
            concurrent_users=5,
            requests_per_user=5,
            ramp_up_time=1.0,
            test_duration=20.0,
            request_delay=0.3,
            timeout=15.0,
        )

        runner = LoadTestRunner(load_test_agent)
        metrics = await runner.run_load_test(config, benchmark_scenarios["medium"])

        # Expected baseline for medium scenarios (from previous test)
        expected_baseline = {
            "avg_response_time": 1.0,  # Expected baseline
            "p95_response_time": 2.0,
            "requests_per_second": 3.0,
        }

        # Calculate regression percentages
        response_time_regression = (
            metrics.avg_response_time - expected_baseline["avg_response_time"]
        ) / expected_baseline["avg_response_time"]

        print(f"\nPerformance Regression Analysis:")
        print(f"  Current Avg Response Time: {metrics.avg_response_time:.3f}s")
        print(f"  Expected Baseline: {expected_baseline['avg_response_time']:.3f}s")
        print(f"  Regression: {response_time_regression:.2%}")

        # This test is designed to detect regression, so we expect it
        # In a real scenario, this would trigger alerts
        if response_time_regression > 0.5:  # 50% regression threshold
            print(
                f"  ⚠️  Performance regression detected: {response_time_regression:.2%}"
            )

        # Verify the test can detect significant performance changes
        assert (
            metrics.avg_response_time > expected_baseline["avg_response_time"]
        ), "Regression test should detect performance degradation"
