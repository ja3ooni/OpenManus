"""
Performance tests for concurrent request handling and response times.

Tests the system's ability to handle multiple concurrent requests while
maintaining acceptable response times and resource usage.
"""

import asyncio
import time
from typing import Dict, List
from unittest.mock import AsyncMock, patch

import psutil
import pytest

from app.agent.manus import Manus
from app.config import Config
from app.exceptions import OpenManusError
from app.schema import AgentResponse


@pytest.mark.performance
class TestConcurrentRequestHandling:
    """Test concurrent request handling performance."""

    @pytest.fixture
    def performance_config(self, test_config):
        """Configuration optimized for performance testing."""
        config = test_config.copy()
        config.update(
            {
                "max_concurrent_requests": 20,
                "request_timeout": 30,
                "memory_limit": "1GB",
                "performance_monitoring": True,
            }
        )
        return config

    @pytest.fixture
    async def performance_agent(self, mock_llm_client, performance_config):
        """Agent configured for performance testing."""
        with patch("app.config.Config._instance", None):
            config = Config()
            config._config = performance_config

            with patch("app.agent.manus.LLMClient") as mock_llm_class:
                mock_llm_class.return_value = mock_llm_client
                agent = Manus()
                return agent

    async def test_concurrent_simple_requests(self, performance_agent, mock_llm_client):
        """Test handling multiple simple concurrent requests."""
        # Setup mock responses
        mock_llm_client.set_responses(
            ["Task completed successfully" for _ in range(10)]
        )

        # Define simple tasks
        tasks = [
            "Calculate 2+2",
            "What is the capital of France?",
            "List 3 colors",
            "Count to 5",
            "Say hello",
        ] * 2  # 10 tasks total

        # Measure performance
        start_time = time.time()
        start_memory = psutil.virtual_memory().used

        # Execute concurrent requests
        async def execute_task(task):
            return await performance_agent.run(task)

        results = await asyncio.gather(
            *[execute_task(task) for task in tasks], return_exceptions=True
        )

        end_time = time.time()
        end_memory = psutil.virtual_memory().used

        # Verify results
        successful_results = [r for r in results if isinstance(r, AgentResponse)]
        failed_results = [r for r in results if isinstance(r, Exception)]

        # Performance assertions
        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory

        assert len(successful_results) >= 8, "At least 80% of requests should succeed"
        assert (
            execution_time < 60
        ), f"Execution time {execution_time:.2f}s should be under 60s"
        assert (
            memory_delta < 100 * 1024 * 1024
        ), f"Memory usage increase {memory_delta} should be under 100MB"

        # Log performance metrics
        avg_response_time = execution_time / len(tasks)
        print(f"Concurrent requests performance:")
        print(f"  Total execution time: {execution_time:.2f}s")
        print(f"  Average response time: {avg_response_time:.2f}s")
        print(f"  Memory delta: {memory_delta / 1024 / 1024:.2f}MB")
        print(f"  Success rate: {len(successful_results)}/{len(tasks)}")

    async def test_concurrent_complex_requests(
        self, performance_agent, mock_llm_client
    ):
        """Test handling concurrent complex requests with tool usage."""
        # Setup mock responses for complex tasks
        mock_llm_client.set_responses(
            ["I'll help you with that complex task" for _ in range(5)]
        )

        # Define complex tasks that would use tools
        complex_tasks = [
            "Create a Python script to calculate fibonacci numbers",
            "Search for information about machine learning",
            "Analyze a CSV file with sales data",
            "Write a function to sort a list of numbers",
            "Generate a report on system performance",
        ]

        start_time = time.time()
        start_cpu = psutil.cpu_percent()

        # Execute concurrent complex requests
        results = await asyncio.gather(
            *[performance_agent.run(task) for task in complex_tasks],
            return_exceptions=True,
        )

        end_time = time.time()
        end_cpu = psutil.cpu_percent()

        # Verify results
        successful_results = [r for r in results if isinstance(r, AgentResponse)]

        execution_time = end_time - start_time
        avg_cpu = (start_cpu + end_cpu) / 2

        assert (
            len(successful_results) >= 4
        ), "At least 80% of complex requests should succeed"
        assert (
            execution_time < 120
        ), f"Complex requests should complete within 120s, took {execution_time:.2f}s"
        assert avg_cpu < 80, f"CPU usage {avg_cpu:.1f}% should stay under 80%"

    async def test_request_queue_handling(self, performance_agent, mock_llm_client):
        """Test request queuing under high load."""
        # Setup responses with varying delays
        mock_llm_client.set_delay(0.1)  # Small delay to simulate processing
        mock_llm_client.set_responses([f"Response {i}" for i in range(20)])

        # Create 20 concurrent requests
        tasks = [f"Task {i}" for i in range(20)]

        start_time = time.time()

        # Execute all requests concurrently
        results = await asyncio.gather(
            *[performance_agent.run(task) for task in tasks], return_exceptions=True
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Verify queue handling
        successful_results = [r for r in results if isinstance(r, AgentResponse)]

        assert (
            len(successful_results) >= 16
        ), "At least 80% of queued requests should succeed"
        assert (
            execution_time < 180
        ), f"Queued requests should complete within 180s, took {execution_time:.2f}s"

        # Check that requests were processed (not all failed immediately)
        assert (
            mock_llm_client.call_count >= 16
        ), "Most requests should have been processed"

    async def test_memory_usage_under_load(self, performance_agent, mock_llm_client):
        """Test memory usage patterns under concurrent load."""
        import gc

        # Force garbage collection before test
        gc.collect()

        # Setup responses
        mock_llm_client.set_responses(["Memory test response" for _ in range(15)])

        # Measure baseline memory
        baseline_memory = psutil.virtual_memory().used

        # Execute memory-intensive concurrent requests
        tasks = ["Generate a large text document with 1000 words" for _ in range(15)]

        results = await asyncio.gather(
            *[performance_agent.run(task) for task in tasks], return_exceptions=True
        )

        # Measure peak memory
        peak_memory = psutil.virtual_memory().used

        # Force cleanup
        del results
        gc.collect()

        # Measure memory after cleanup
        final_memory = psutil.virtual_memory().used

        memory_increase = peak_memory - baseline_memory
        memory_retained = final_memory - baseline_memory

        # Memory usage assertions
        assert (
            memory_increase < 200 * 1024 * 1024
        ), f"Memory increase {memory_increase / 1024 / 1024:.2f}MB should be under 200MB"
        assert (
            memory_retained < 50 * 1024 * 1024
        ), f"Retained memory {memory_retained / 1024 / 1024:.2f}MB should be under 50MB"

    async def test_response_time_distribution(self, performance_agent, mock_llm_client):
        """Test response time distribution for concurrent requests."""
        # Setup responses with consistent timing
        mock_llm_client.set_delay(0.05)  # 50ms delay
        mock_llm_client.set_responses(["Timed response" for _ in range(10)])

        tasks = ["Quick task" for _ in range(10)]
        response_times = []

        # Measure individual response times
        async def timed_request(task):
            start = time.time()
            result = await performance_agent.run(task)
            end = time.time()
            response_times.append(end - start)
            return result

        results = await asyncio.gather(
            *[timed_request(task) for task in tasks], return_exceptions=True
        )

        # Analyze response time distribution
        successful_times = [
            t
            for i, t in enumerate(response_times)
            if isinstance(results[i], AgentResponse)
        ]

        if successful_times:
            avg_time = sum(successful_times) / len(successful_times)
            max_time = max(successful_times)
            min_time = min(successful_times)

            # Response time assertions
            assert (
                avg_time < 5.0
            ), f"Average response time {avg_time:.2f}s should be under 5s"
            assert (
                max_time < 10.0
            ), f"Maximum response time {max_time:.2f}s should be under 10s"
            assert (
                min_time > 0.01
            ), f"Minimum response time {min_time:.2f}s should be reasonable"

            # Check for reasonable distribution (no extreme outliers)
            time_variance = max_time - min_time
            assert (
                time_variance < 8.0
            ), f"Response time variance {time_variance:.2f}s should be reasonable"

    async def test_error_handling_under_load(self, performance_agent, mock_llm_client):
        """Test error handling during concurrent requests."""
        # Setup mixed responses (some errors, some success)
        mock_llm_client.set_errors(
            [
                OpenManusError("Simulated error 1"),
                None,  # Success
                OpenManusError("Simulated error 2"),
                None,  # Success
                TimeoutError("Timeout error"),
            ]
        )
        mock_llm_client.set_responses(
            [
                "Success response 1",
                "Success response 2",
                "Success response 3",
                "Success response 4",
                "Success response 5",
            ]
        )

        tasks = [f"Task with potential error {i}" for i in range(5)]

        start_time = time.time()

        results = await asyncio.gather(
            *[performance_agent.run(task) for task in tasks], return_exceptions=True
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Analyze error handling
        successful_results = [r for r in results if isinstance(r, AgentResponse)]
        error_results = [r for r in results if isinstance(r, Exception)]

        # Verify graceful error handling
        assert (
            len(results) == 5
        ), "All requests should return a result (success or error)"
        assert (
            execution_time < 30
        ), f"Error handling should not significantly delay execution: {execution_time:.2f}s"

        # Should have some successes and some errors based on our setup
        assert len(successful_results) >= 1, "Some requests should succeed"
        assert len(error_results) >= 1, "Some requests should fail as expected"

    @pytest.mark.slow
    async def test_sustained_load_performance(self, performance_agent, mock_llm_client):
        """Test performance under sustained load over time."""
        # Setup for sustained load test
        mock_llm_client.set_delay(0.1)
        mock_llm_client.set_responses(
            [f"Sustained load response {i}" for i in range(50)]
        )

        # Run multiple batches of concurrent requests
        batch_size = 5
        num_batches = 10
        batch_results = []

        for batch in range(num_batches):
            batch_start = time.time()

            tasks = [f"Batch {batch} Task {i}" for i in range(batch_size)]

            results = await asyncio.gather(
                *[performance_agent.run(task) for task in tasks], return_exceptions=True
            )

            batch_end = time.time()
            batch_time = batch_end - batch_start

            successful_count = len([r for r in results if isinstance(r, AgentResponse)])

            batch_results.append(
                {
                    "batch": batch,
                    "time": batch_time,
                    "success_count": successful_count,
                    "total_count": batch_size,
                }
            )

            # Small delay between batches
            await asyncio.sleep(0.5)

        # Analyze sustained performance
        avg_batch_time = sum(b["time"] for b in batch_results) / len(batch_results)
        total_success = sum(b["success_count"] for b in batch_results)
        total_requests = sum(b["total_count"] for b in batch_results)

        success_rate = total_success / total_requests

        # Performance assertions for sustained load
        assert (
            avg_batch_time < 15.0
        ), f"Average batch time {avg_batch_time:.2f}s should be under 15s"
        assert (
            success_rate >= 0.8
        ), f"Success rate {success_rate:.2%} should be at least 80%"

        # Check for performance degradation over time
        early_batches = batch_results[:3]
        late_batches = batch_results[-3:]

        early_avg = sum(b["time"] for b in early_batches) / len(early_batches)
        late_avg = sum(b["time"] for b in late_batches) / len(late_batches)

        degradation = (late_avg - early_avg) / early_avg if early_avg > 0 else 0

        assert (
            degradation < 0.5
        ), f"Performance degradation {degradation:.2%} should be under 50%"


@pytest.mark.performance
class TestResponseTimeMetrics:
    """Test response time metrics and SLA compliance."""

    async def test_response_time_sla_compliance(
        self, performance_agent, mock_llm_client
    ):
        """Test that response times meet SLA requirements."""
        # Setup consistent responses
        mock_llm_client.set_responses(["SLA test response" for _ in range(20)])

        # Define SLA requirements
        sla_requirements = {
            "p50": 2.0,  # 50th percentile under 2s
            "p95": 5.0,  # 95th percentile under 5s
            "p99": 10.0,  # 99th percentile under 10s
        }

        tasks = ["SLA test task" for _ in range(20)]
        response_times = []

        # Measure response times
        for task in tasks:
            start_time = time.time()
            try:
                await performance_agent.run(task)
                end_time = time.time()
                response_times.append(end_time - start_time)
            except Exception:
                # Include failed requests as maximum time for SLA calculation
                response_times.append(sla_requirements["p99"])

        # Calculate percentiles
        response_times.sort()
        n = len(response_times)

        p50_index = int(0.50 * n)
        p95_index = int(0.95 * n)
        p99_index = int(0.99 * n)

        p50_time = response_times[p50_index] if p50_index < n else response_times[-1]
        p95_time = response_times[p95_index] if p95_index < n else response_times[-1]
        p99_time = response_times[p99_index] if p99_index < n else response_times[-1]

        # Verify SLA compliance
        assert (
            p50_time <= sla_requirements["p50"]
        ), f"P50 response time {p50_time:.2f}s exceeds SLA {sla_requirements['p50']}s"
        assert (
            p95_time <= sla_requirements["p95"]
        ), f"P95 response time {p95_time:.2f}s exceeds SLA {sla_requirements['p95']}s"
        assert (
            p99_time <= sla_requirements["p99"]
        ), f"P99 response time {p99_time:.2f}s exceeds SLA {sla_requirements['p99']}s"

    async def test_timeout_handling_performance(
        self, performance_agent, mock_llm_client
    ):
        """Test performance of timeout handling."""
        # Setup responses with timeouts
        mock_llm_client.set_errors([TimeoutError("Request timeout") for _ in range(5)])

        tasks = ["Timeout test task" for _ in range(5)]

        start_time = time.time()

        results = await asyncio.gather(
            *[performance_agent.run(task) for task in tasks], return_exceptions=True
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Verify timeout handling doesn't block system
        timeout_errors = [r for r in results if isinstance(r, TimeoutError)]

        assert len(timeout_errors) >= 3, "Most requests should timeout as expected"
        assert (
            execution_time < 60
        ), f"Timeout handling should be fast: {execution_time:.2f}s"
