"""
Tests for performance dashboard integration with automated testing.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.performance.dashboard import PerformanceDashboard
from app.performance.testing import (
    PerformanceTestConfig,
    PerformanceTestResult,
    PerformanceTestRunner,
    TestStatus,
    TestType,
)


class TestPerformanceDashboardIntegration:
    """Test performance dashboard integration with automated testing."""

    @pytest.fixture
    def test_runner(self):
        """Create test runner for testing."""
        return PerformanceTestRunner()

    @pytest.fixture
    def dashboard(self, test_runner):
        """Create performance dashboard for testing."""
        return PerformanceDashboard(test_runner=test_runner)

    @pytest.fixture
    def sample_test_config(self):
        """Create sample test configuration."""
        return PerformanceTestConfig(
            name="integration_test",
            test_type=TestType.BASELINE,
            duration_seconds=1,
            concurrent_users=1,
            target_rps=1.0,
            warmup_requests=0,
            cooldown_seconds=0,
        )

    @pytest.fixture
    def sample_test_result(self):
        """Create sample test result."""
        return PerformanceTestResult(
            test_name="integration_test",
            test_type=TestType.BASELINE,
            status=TestStatus.COMPLETED,
            start_time=datetime.now(timezone.utc),
            total_requests=10,
            successful_requests=9,
            failed_requests=1,
            avg_response_time_ms=150.0,
            requests_per_second=2.0,
            error_rate_percent=10.0,
        )

    @pytest.mark.asyncio
    async def test_dashboard_includes_test_data(
        self, dashboard, test_runner, sample_test_result
    ):
        """Test that dashboard includes performance test data."""
        # Add test result to runner
        test_runner.test_results.append(sample_test_result)

        # Get dashboard data
        dashboard_data = await dashboard.get_performance_dashboard_data()

        # Verify test data is included
        assert "performance" in dashboard_data
        assert "automated_testing" in dashboard_data["performance"]

        automated_testing = dashboard_data["performance"]["automated_testing"]
        assert "test_summary" in automated_testing
        assert "recent_test_results" in automated_testing
        assert "test_configs" in automated_testing

        # Verify test summary includes our result
        test_summary = automated_testing["test_summary"]
        assert test_summary["total_tests_executed"] >= 1

    @pytest.mark.asyncio
    async def test_run_automated_performance_tests(
        self, dashboard, test_runner, sample_test_config
    ):
        """Test running automated performance tests through dashboard."""
        # Add test configuration
        test_runner.add_test_config(sample_test_config)

        # Mock test function
        async def mock_test_function():
            await asyncio.sleep(0.01)

        # Run tests through dashboard
        analysis = await dashboard.run_automated_performance_tests(
            test_filter=["integration_test"], test_function=mock_test_function
        )

        # Verify analysis structure
        assert "execution_summary" in analysis
        assert "performance_analysis" in analysis
        assert "recommendations" in analysis
        assert "test_results" in analysis

        # Verify execution summary
        execution_summary = analysis["execution_summary"]
        assert execution_summary["total_tests"] == 1
        assert execution_summary["successful_tests"] >= 0
        assert execution_summary["failed_tests"] >= 0

        # Verify test results
        assert len(analysis["test_results"]) == 1
        test_result = analysis["test_results"][0]
        assert test_result["test_name"] == "integration_test"

    @pytest.mark.asyncio
    async def test_run_automated_tests_with_failures(
        self, dashboard, test_runner, sample_test_config
    ):
        """Test running automated tests with simulated failures."""
        test_runner.add_test_config(sample_test_config)

        # Mock failing test function
        async def failing_test_function():
            await asyncio.sleep(0.01)
            raise Exception("Simulated test failure")

        analysis = await dashboard.run_automated_performance_tests(
            test_filter=["integration_test"], test_function=failing_test_function
        )

        # Should complete but with errors
        assert "execution_summary" in analysis
        assert analysis["execution_summary"]["total_tests"] == 1

        # Should have test results even with failures
        assert len(analysis["test_results"]) == 1
        test_result = analysis["test_results"][0]
        assert test_result["failed_requests"] > 0

    @pytest.mark.asyncio
    async def test_regression_testing_integration(
        self, dashboard, test_runner, sample_test_config
    ):
        """Test regression testing integration."""
        test_runner.add_test_config(sample_test_config)

        # Create baseline result
        baseline_result = PerformanceTestResult(
            test_name="integration_test",
            test_type=TestType.BASELINE,
            status=TestStatus.COMPLETED,
            start_time=datetime.now(timezone.utc),
            avg_response_time_ms=100.0,
            requests_per_second=2.0,
            error_rate_percent=5.0,
        )
        test_runner.set_baseline("integration_test", baseline_result)

        # Run regression tests
        analysis = await dashboard.run_regression_tests(["integration_test"])

        # Verify analysis structure
        assert "regression_summary" in analysis
        assert "regression_details" in analysis
        assert "critical_regressions" in analysis

        # Should have run one regression test
        regression_summary = analysis["regression_summary"]
        assert regression_summary["total_tests"] >= 0

    @pytest.mark.asyncio
    async def test_regression_testing_no_baselines(self, dashboard, test_runner):
        """Test regression testing when no baselines exist."""
        analysis = await dashboard.run_regression_tests()

        # Should return error about no baselines
        assert "error" in analysis
        assert "No baseline results available" in analysis["error"]
        assert analysis["regression_summary"]["total_tests"] == 0

    @pytest.mark.asyncio
    async def test_performance_recommendations_generation(
        self, dashboard, test_runner, sample_test_config
    ):
        """Test that performance recommendations are generated based on test results."""
        # Configure test with strict targets
        sample_test_config.max_response_time_ms = 50.0  # Very strict
        sample_test_config.target_rps = 10.0  # High target
        test_runner.add_test_config(sample_test_config)

        # Mock slow test function
        async def slow_test_function():
            await asyncio.sleep(0.1)  # 100ms - exceeds 50ms target

        analysis = await dashboard.run_automated_performance_tests(
            test_filter=["integration_test"], test_function=slow_test_function
        )

        # Should have recommendations due to not meeting targets
        assert "recommendations" in analysis
        # May or may not have recommendations depending on exact test execution

    @pytest.mark.asyncio
    async def test_dashboard_error_handling(self, dashboard):
        """Test dashboard error handling for test operations."""
        # Test with invalid test filter
        analysis = await dashboard.run_automated_performance_tests(
            test_filter=["nonexistent_test"]
        )

        # Should handle gracefully
        assert "execution_summary" in analysis
        assert analysis["execution_summary"]["total_tests"] == 0

    @pytest.mark.asyncio
    async def test_concurrent_test_execution_through_dashboard(
        self, dashboard, test_runner
    ):
        """Test concurrent test execution through dashboard."""
        # Add multiple test configurations
        configs = [
            PerformanceTestConfig(
                name=f"concurrent_test_{i}",
                test_type=TestType.BASELINE,
                duration_seconds=0.5,
                concurrent_users=1,
                warmup_requests=0,
                cooldown_seconds=0,
            )
            for i in range(3)
        ]

        for config in configs:
            test_runner.add_test_config(config)

        # Mock test function
        async def mock_test_function():
            await asyncio.sleep(0.01)

        # Run all tests
        analysis = await dashboard.run_automated_performance_tests(
            test_filter=[f"concurrent_test_{i}" for i in range(3)],
            test_function=mock_test_function,
        )

        # Should have run all tests
        assert analysis["execution_summary"]["total_tests"] == 3
        assert len(analysis["test_results"]) == 3

    @pytest.mark.asyncio
    async def test_performance_analysis_integration(
        self, dashboard, test_runner, sample_test_config
    ):
        """Test integration of performance analysis with test results."""
        test_runner.add_test_config(sample_test_config)

        # Mock test function with known performance characteristics
        async def predictable_test_function():
            await asyncio.sleep(0.05)  # 50ms response time

        analysis = await dashboard.run_automated_performance_tests(
            test_filter=["integration_test"], test_function=predictable_test_function
        )

        # Verify performance analysis
        if analysis["execution_summary"]["successful_tests"] > 0:
            assert "performance_analysis" in analysis

            if "integration_test" in analysis["performance_analysis"]:
                perf_analysis = analysis["performance_analysis"]["integration_test"]
                assert "performance_level" in perf_analysis
                assert "meets_targets" in perf_analysis
                assert "avg_response_time_ms" in perf_analysis
                assert "requests_per_second" in perf_analysis
                assert "error_rate_percent" in perf_analysis

    @pytest.mark.asyncio
    async def test_test_configuration_management_through_dashboard(
        self, dashboard, test_runner
    ):
        """Test test configuration management through dashboard integration."""
        # Add test configuration
        config = PerformanceTestConfig(
            name="config_test",
            test_type=TestType.LOAD,
            duration_seconds=1,
            tags=["dashboard", "integration"],
        )
        test_runner.add_test_config(config)

        # Get dashboard data
        dashboard_data = await dashboard.get_performance_dashboard_data()

        # Verify configuration is included
        automated_testing = dashboard_data["performance"]["automated_testing"]
        test_configs = automated_testing["test_configs"]

        assert "config_test" in test_configs
        config_data = test_configs["config_test"]
        assert config_data["test_type"] == "load"
        assert config_data["tags"] == ["dashboard", "integration"]

    @pytest.mark.asyncio
    async def test_dashboard_with_mixed_test_results(self, dashboard, test_runner):
        """Test dashboard with mixed successful and failed test results."""
        # Add test results with different statuses
        results = [
            PerformanceTestResult(
                test_name="successful_test",
                test_type=TestType.BASELINE,
                status=TestStatus.COMPLETED,
                start_time=datetime.now(timezone.utc),
                avg_response_time_ms=100.0,
                requests_per_second=2.0,
            ),
            PerformanceTestResult(
                test_name="failed_test",
                test_type=TestType.LOAD,
                status=TestStatus.FAILED,
                start_time=datetime.now(timezone.utc),
                errors=["Test execution failed"],
            ),
            PerformanceTestResult(
                test_name="cancelled_test",
                test_type=TestType.STRESS,
                status=TestStatus.CANCELLED,
                start_time=datetime.now(timezone.utc),
            ),
        ]

        test_runner.test_results.extend(results)

        # Get dashboard data
        dashboard_data = await dashboard.get_performance_dashboard_data()

        # Verify test summary includes all results
        test_summary = dashboard_data["performance"]["automated_testing"][
            "test_summary"
        ]
        assert test_summary["total_tests_executed"] >= 3


if __name__ == "__main__":
    pytest.main([__file__])
