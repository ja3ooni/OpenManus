"""
Tests for the performance testing and benchmarking system.
"""

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.performance.testing import (
    PerformanceTestConfig,
    PerformanceTestResult,
    PerformanceTestRunner,
    RegressionTestResult,
    TestStatus,
    TestType,
    get_performance_test_summary,
    performance_test_runner,
    run_all_performance_tests,
    run_performance_test,
    run_regression_test,
    set_performance_baseline,
)


class TestPerformanceTestConfig:
    """Test performance test configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PerformanceTestConfig(name="test_config", test_type=TestType.BASELINE)

        assert config.name == "test_config"
        assert config.test_type == TestType.BASELINE
        assert config.duration_seconds == 60
        assert config.concurrent_users == 1
        assert config.ramp_up_seconds == 10
        assert config.target_rps == 1.0
        assert config.max_response_time_ms == 30000
        assert config.max_error_rate_percent == 5.0
        assert config.enabled is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PerformanceTestConfig(
            name="custom_test",
            test_type=TestType.LOAD,
            duration_seconds=120,
            concurrent_users=5,
            target_rps=2.0,
            max_response_time_ms=15000,
            tags=["custom", "load"],
            metadata={"purpose": "testing"},
        )

        assert config.duration_seconds == 120
        assert config.concurrent_users == 5
        assert config.target_rps == 2.0
        assert config.max_response_time_ms == 15000
        assert config.tags == ["custom", "load"]
        assert config.metadata["purpose"] == "testing"


class TestPerformanceTestResult:
    """Test performance test result data model."""

    def test_result_creation(self):
        """Test creating performance test result."""
        start_time = datetime.now(timezone.utc)

        result = PerformanceTestResult(
            test_name="test_result",
            test_type=TestType.BASELINE,
            status=TestStatus.COMPLETED,
            start_time=start_time,
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
        )

        assert result.test_name == "test_result"
        assert result.test_type == TestType.BASELINE
        assert result.status == TestStatus.COMPLETED
        assert result.start_time == start_time
        assert result.total_requests == 100
        assert result.successful_requests == 95
        assert result.failed_requests == 5


class TestPerformanceTestRunner:
    """Test performance test runner functionality."""

    @pytest.fixture
    def test_runner(self):
        """Create test runner for testing."""
        return PerformanceTestRunner()

    @pytest.fixture
    def simple_test_config(self):
        """Create simple test configuration."""
        return PerformanceTestConfig(
            name="simple_test",
            test_type=TestType.BASELINE,
            duration_seconds=1,  # Short for testing
            concurrent_users=1,
            target_rps=1.0,
            warmup_requests=2,
            cooldown_seconds=0,
        )

    def test_default_test_configs(self, test_runner):
        """Test that default test configurations are loaded."""
        assert len(test_runner.test_configs) > 0
        assert "agent_baseline_performance" in test_runner.test_configs
        assert "agent_load_test" in test_runner.test_configs
        assert "tool_execution_performance" in test_runner.test_configs

    @pytest.mark.asyncio
    async def test_default_test_function(self, test_runner):
        """Test the default test function."""
        # Should complete without error most of the time
        start_time = time.time()
        await test_runner._default_test_function()
        duration = time.time() - start_time

        # Should take at least 0.1 seconds
        assert duration >= 0.1

    @pytest.mark.asyncio
    async def test_run_simple_test(self, test_runner, simple_test_config):
        """Test running a simple performance test."""
        test_runner.add_test_config(simple_test_config)

        async def mock_test_function():
            await asyncio.sleep(0.05)  # 50ms response time

        result = await test_runner.run_test("simple_test", mock_test_function)

        assert result.test_name == "simple_test"
        assert result.status == TestStatus.COMPLETED
        assert result.total_requests > 0
        assert result.avg_response_time_ms > 0
        assert result.requests_per_second > 0
        assert result.end_time is not None

    @pytest.mark.asyncio
    async def test_run_test_with_errors(self, test_runner, simple_test_config):
        """Test running test with simulated errors."""
        test_runner.add_test_config(simple_test_config)

        async def failing_test_function():
            await asyncio.sleep(0.01)
            raise Exception("Simulated test error")

        result = await test_runner.run_test("simple_test", failing_test_function)

        assert result.status == TestStatus.COMPLETED
        assert result.failed_requests > 0
        assert result.error_rate_percent > 0
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_concurrent_test_execution(self, test_runner):
        """Test concurrent test execution."""
        config = PerformanceTestConfig(
            name="concurrent_test",
            test_type=TestType.LOAD,
            duration_seconds=2,
            concurrent_users=3,
            target_rps=2.0,
            warmup_requests=0,
            cooldown_seconds=0,
        )
        test_runner.add_test_config(config)

        async def mock_test_function():
            await asyncio.sleep(0.1)

        result = await test_runner.run_test("concurrent_test", mock_test_function)

        assert result.status == TestStatus.COMPLETED
        assert (
            result.total_requests > 3
        )  # Should have multiple requests from concurrent users

    @pytest.mark.asyncio
    async def test_disabled_test(self, test_runner):
        """Test that disabled tests are skipped."""
        config = PerformanceTestConfig(
            name="disabled_test",
            test_type=TestType.BASELINE,
            enabled=False,
        )
        test_runner.add_test_config(config)

        result = await test_runner.run_test("disabled_test")

        assert result.status == TestStatus.CANCELLED
        assert result.metadata.get("reason") == "Test disabled"

    @pytest.mark.asyncio
    async def test_performance_level_assessment(self, test_runner, simple_test_config):
        """Test performance level assessment."""
        test_runner.add_test_config(simple_test_config)

        # Fast test function (should be excellent)
        async def fast_test_function():
            await asyncio.sleep(0.001)  # 1ms

        result = await test_runner.run_test("simple_test", fast_test_function)

        # Performance level should be good or excellent
        assert result.performance_level.value in ["good", "excellent"]
        assert result.meets_targets is True

    @pytest.mark.asyncio
    async def test_test_history(self, test_runner, simple_test_config):
        """Test test execution history tracking."""
        test_runner.add_test_config(simple_test_config)

        # Run test multiple times
        for i in range(3):
            await test_runner.run_test("simple_test")
            await asyncio.sleep(0.1)  # Small delay

        history = test_runner.get_test_history("simple_test", limit=5)

        assert len(history) == 3
        # Should be sorted by start time (most recent first)
        assert history[0].start_time >= history[1].start_time
        assert history[1].start_time >= history[2].start_time

    def test_add_remove_test_config(self, test_runner):
        """Test adding and removing test configurations."""
        config = PerformanceTestConfig(
            name="temp_test",
            test_type=TestType.BASELINE,
        )

        # Add config
        test_runner.add_test_config(config)
        assert "temp_test" in test_runner.test_configs

        # Remove config
        test_runner.remove_test_config("temp_test")
        assert "temp_test" not in test_runner.test_configs

    @pytest.mark.asyncio
    async def test_baseline_and_regression_testing(
        self, test_runner, simple_test_config
    ):
        """Test baseline setting and regression testing."""
        test_runner.add_test_config(simple_test_config)

        # Run baseline test
        async def baseline_test_function():
            await asyncio.sleep(0.05)  # 50ms baseline

        baseline_result = await test_runner.run_test(
            "simple_test", baseline_test_function
        )
        test_runner.set_baseline("simple_test", baseline_result)

        # Run regression test (slower)
        async def regression_test_function():
            await asyncio.sleep(0.1)  # 100ms (2x slower)

        regression_result = await test_runner.run_regression_test(
            "simple_test",
            regression_test_function,
            regression_threshold_percent=50.0,  # 50% threshold
        )

        assert regression_result.test_name == "simple_test"
        assert regression_result.baseline_result == baseline_result
        assert regression_result.response_time_change_percent > 0  # Should be slower
        # May or may not detect regression depending on exact timing

    @pytest.mark.asyncio
    async def test_run_all_tests(self, test_runner):
        """Test running all configured tests."""
        # Add a quick test config
        config = PerformanceTestConfig(
            name="quick_test",
            test_type=TestType.BASELINE,
            duration_seconds=0.5,
            warmup_requests=0,
            cooldown_seconds=0,
        )
        test_runner.add_test_config(config)

        async def quick_test_function():
            await asyncio.sleep(0.01)

        # Run only our quick test
        results = await test_runner.run_all_tests(
            quick_test_function, test_filter=["quick_test"]
        )

        assert len(results) == 1
        assert results[0].test_name == "quick_test"
        assert results[0].status == TestStatus.COMPLETED

    def test_get_test_summary(self, test_runner):
        """Test getting test summary."""
        # Add some mock results
        result1 = PerformanceTestResult(
            test_name="test1",
            test_type=TestType.BASELINE,
            status=TestStatus.COMPLETED,
            start_time=datetime.now(timezone.utc),
            avg_response_time_ms=100.0,
            requests_per_second=2.0,
        )

        result2 = PerformanceTestResult(
            test_name="test1",
            test_type=TestType.BASELINE,
            status=TestStatus.COMPLETED,
            start_time=datetime.now(timezone.utc),
            avg_response_time_ms=120.0,
            requests_per_second=1.8,
        )

        test_runner.test_results.extend([result1, result2])

        summary = test_runner.get_test_summary()

        assert summary["total_tests_executed"] == 2
        assert summary["unique_tests"] == 1
        assert "test1" in summary["test_summaries"]
        assert summary["test_summaries"]["test1"]["execution_count"] == 2


class TestRegressionTesting:
    """Test regression testing functionality."""

    def test_percentage_change_calculation(self):
        """Test percentage change calculation."""
        runner = PerformanceTestRunner()

        # Normal case
        change = runner._calculate_percentage_change(100.0, 120.0)
        assert change == 20.0

        # Decrease
        change = runner._calculate_percentage_change(100.0, 80.0)
        assert change == -20.0

        # Zero baseline
        change = runner._calculate_percentage_change(0.0, 50.0)
        assert change == 100.0

        # Both zero
        change = runner._calculate_percentage_change(0.0, 0.0)
        assert change == 0.0


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_run_performance_test(self):
        """Test run_performance_test convenience function."""
        # Add a test config to the global runner
        config = PerformanceTestConfig(
            name="convenience_test",
            test_type=TestType.BASELINE,
            duration_seconds=0.5,
            warmup_requests=0,
            cooldown_seconds=0,
        )
        performance_test_runner.add_test_config(config)

        async def test_function():
            await asyncio.sleep(0.01)

        result = await run_performance_test("convenience_test", test_function)

        assert result.test_name == "convenience_test"
        assert result.status == TestStatus.COMPLETED

    def test_set_performance_baseline(self):
        """Test set_performance_baseline convenience function."""
        result = PerformanceTestResult(
            test_name="baseline_test",
            test_type=TestType.BASELINE,
            status=TestStatus.COMPLETED,
            start_time=datetime.now(timezone.utc),
        )

        set_performance_baseline("baseline_test", result)

        assert "baseline_test" in performance_test_runner.baseline_results
        assert performance_test_runner.baseline_results["baseline_test"] == result

    def test_get_performance_test_summary(self):
        """Test get_performance_test_summary convenience function."""
        summary = get_performance_test_summary()

        assert isinstance(summary, dict)
        assert "total_tests_executed" in summary


class TestPerformanceTestIntegration:
    """Test performance testing integration scenarios."""

    @pytest.mark.asyncio
    async def test_complete_test_lifecycle(self):
        """Test complete performance test lifecycle."""
        runner = PerformanceTestRunner()

        # Create test configuration
        config = PerformanceTestConfig(
            name="lifecycle_test",
            test_type=TestType.BASELINE,
            duration_seconds=1,
            concurrent_users=2,
            target_rps=1.0,
            warmup_requests=1,
            cooldown_seconds=0,
        )
        runner.add_test_config(config)

        # Define test function
        request_count = 0

        async def test_function():
            nonlocal request_count
            request_count += 1
            await asyncio.sleep(0.05)

            # Simulate occasional errors
            if request_count % 10 == 0:
                raise Exception("Simulated error")

        # Run test
        result = await runner.run_test("lifecycle_test", test_function)

        # Verify results
        assert result.status == TestStatus.COMPLETED
        assert result.total_requests > 0
        assert result.avg_response_time_ms > 0
        assert result.requests_per_second > 0
        assert result.duration_seconds > 0

        # Set as baseline
        runner.set_baseline("lifecycle_test", result)

        # Run regression test
        async def slower_test_function():
            await asyncio.sleep(0.1)  # Slower than baseline

        regression_result = await runner.run_regression_test(
            "lifecycle_test", slower_test_function, regression_threshold_percent=20.0
        )

        assert regression_result.test_name == "lifecycle_test"
        assert regression_result.baseline_result == result
        assert regression_result.response_time_change_percent > 0

        # Get test history
        history = runner.get_test_history("lifecycle_test")
        assert len(history) >= 2  # Baseline + regression test

        # Get summary
        summary = runner.get_test_summary()
        assert summary["unique_tests"] >= 1
        assert "lifecycle_test" in summary["test_summaries"]

    @pytest.mark.asyncio
    async def test_stress_test_scenario(self):
        """Test stress testing scenario."""
        runner = PerformanceTestRunner()

        config = PerformanceTestConfig(
            name="stress_test",
            test_type=TestType.STRESS,
            duration_seconds=2,
            concurrent_users=5,
            target_rps=3.0,
            max_response_time_ms=1000,
            max_error_rate_percent=20.0,
            warmup_requests=0,
            cooldown_seconds=0,
        )
        runner.add_test_config(config)

        # Simulate stressed system
        async def stressed_test_function():
            # Variable response time
            delay = 0.1 + (time.time() % 0.2)  # 0.1-0.3 seconds
            await asyncio.sleep(delay)

            # Higher error rate under stress
            if time.time() % 10 < 3:  # 30% error rate
                raise Exception("System overloaded")

        result = await runner.run_test("stress_test", stressed_test_function)

        assert result.status == TestStatus.COMPLETED
        assert result.error_rate_percent > 0  # Should have some errors
        # Performance level might be poor due to high error rate


if __name__ == "__main__":
    pytest.main([__file__])
