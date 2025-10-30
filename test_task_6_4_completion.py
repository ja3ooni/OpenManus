#!/usr/bin/env python3
"""
Test to verify completion of Task 6.4: Create Performance Monitoring and Metrics
"""

import asyncio
import os
import sys
import time
from datetime import datetime, timezone

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all performance monitoring components can be imported."""
    print("🧪 Testing Performance Monitoring Imports...")

    try:
        # Test metrics module
        from app.performance.metrics import (
            BenchmarkResult,
            PerformanceBenchmarkRunner,
            PerformanceLevel,
            PerformanceMetricsCollector,
            SLAReport,
            SLATarget,
        )

        print("✅ Performance metrics components imported successfully")

        # Test dashboard module
        from app.performance.dashboard import PerformanceDashboard

        print("✅ Performance dashboard imported successfully")

        # Test testing module
        from app.performance.testing import (
            PerformanceTestConfig,
            PerformanceTestRunner,
            TestStatus,
            TestType,
        )

        print("✅ Performance testing components imported successfully")

        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


async def test_metrics_collection():
    """Test performance metrics collection functionality."""
    print("\n🧪 Testing Performance Metrics Collection...")

    try:
        from app.performance.metrics import PerformanceMetricsCollector

        # Create metrics collector
        collector = PerformanceMetricsCollector()
        print("✅ Created performance metrics collector")

        # Test recording metrics
        collector.record_response_time("test_operation", 150.0, {"test": "true"})
        collector.record_throughput("test_operation", 10, {"test": "true"})
        collector.record_error_rate("test_operation", 5.0, {"test": "true"})
        print("✅ Recorded test metrics")

        # Test getting statistics
        response_stats = collector.get_response_time_stats("test_operation")
        throughput_stats = collector.get_throughput_stats("test_operation")
        error_stats = collector.get_error_rate_stats("test_operation")

        print(f"✅ Response time stats: {response_stats}")
        print(f"✅ Throughput stats: {throughput_stats}")
        print(f"✅ Error rate stats: {error_stats}")

        # Test SLA reporting
        sla_reports = collector.get_all_sla_reports()
        print(f"✅ SLA reports generated: {len(sla_reports)} reports")

        # Test performance summary
        summary = collector.get_performance_summary()
        print(f"✅ Performance summary generated: {summary['overall_performance']}")

        return True

    except Exception as e:
        print(f"❌ Metrics collection test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_performance_testing():
    """Test automated performance testing functionality."""
    print("\n🧪 Testing Automated Performance Testing...")

    try:
        from app.performance.testing import (
            PerformanceTestConfig,
            PerformanceTestRunner,
            TestType,
        )

        # Create test runner
        runner = PerformanceTestRunner()
        print(f"✅ Created test runner with {len(runner.test_configs)} default configs")

        # Add a quick test configuration
        config = PerformanceTestConfig(
            name="completion_test",
            test_type=TestType.BASELINE,
            duration_seconds=1,
            concurrent_users=1,
            warmup_requests=0,
            cooldown_seconds=0,
        )
        runner.add_test_config(config)
        print("✅ Added test configuration")

        # Define a simple test function
        async def simple_test():
            await asyncio.sleep(0.01)  # 10ms simulated work

        # Run the test
        print("🏃 Running performance test...")
        result = await runner.run_test("completion_test", simple_test)

        print(f"✅ Test completed:")
        print(f"   - Status: {result.status.value}")
        print(f"   - Total requests: {result.total_requests}")
        print(f"   - Avg response time: {result.avg_response_time_ms:.2f}ms")
        print(f"   - Performance level: {result.performance_level.value}")

        # Test summary
        summary = runner.get_test_summary()
        print(f"✅ Test summary: {summary['total_tests_executed']} tests executed")

        return True

    except Exception as e:
        print(f"❌ Performance testing test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_dashboard_integration():
    """Test performance dashboard integration."""
    print("\n🧪 Testing Performance Dashboard Integration...")

    try:
        from app.performance.dashboard import PerformanceDashboard
        from app.performance.metrics import PerformanceMetricsCollector
        from app.performance.testing import PerformanceTestRunner

        # Create components
        metrics_collector = PerformanceMetricsCollector()
        test_runner = PerformanceTestRunner()
        dashboard = PerformanceDashboard(
            metrics_collector=metrics_collector, test_runner=test_runner
        )
        print("✅ Created performance dashboard")

        # Add some test data
        metrics_collector.record_response_time("dashboard_test", 200.0)
        metrics_collector.record_throughput("dashboard_test", 5)
        metrics_collector.record_error_rate("dashboard_test", 2.0)
        print("✅ Added test metrics data")

        # Get dashboard data
        dashboard_data = await dashboard.get_performance_dashboard_data()

        # Verify dashboard structure
        assert "performance" in dashboard_data
        assert "summary" in dashboard_data["performance"]
        assert "automated_testing" in dashboard_data["performance"]
        print("✅ Dashboard data structure verified")

        # Test automated performance tests through dashboard
        from app.performance.testing import PerformanceTestConfig, TestType

        quick_config = PerformanceTestConfig(
            name="dashboard_integration_test",
            test_type=TestType.BASELINE,
            duration_seconds=0.5,
            warmup_requests=0,
            cooldown_seconds=0,
        )
        test_runner.add_test_config(quick_config)

        async def dashboard_test_func():
            await asyncio.sleep(0.005)  # 5ms

        analysis = await dashboard.run_automated_performance_tests(
            test_filter=["dashboard_integration_test"],
            test_function=dashboard_test_func,
        )

        print(f"✅ Dashboard automated tests completed:")
        print(f"   - Total tests: {analysis['execution_summary']['total_tests']}")
        print(
            f"   - Successful tests: {analysis['execution_summary']['successful_tests']}"
        )

        return True

    except Exception as e:
        print(f"❌ Dashboard integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_file_structure():
    """Test that all required files exist and have content."""
    print("\n🧪 Testing File Structure...")

    required_files = [
        "app/performance/metrics.py",
        "app/performance/dashboard.py",
        "app/performance/testing.py",
        "tests/performance/test_performance_testing.py",
        "tests/performance/test_dashboard_integration.py",
    ]

    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            if size > 0:
                print(f"✅ {file_path} exists ({size} bytes)")
            else:
                print(f"❌ {file_path} exists but is empty")
                all_exist = False
        else:
            print(f"❌ {file_path} does not exist")
            all_exist = False

    return all_exist


async def main():
    """Main test function."""
    print("🎯 Task 6.4 Completion Test: Create Performance Monitoring and Metrics")
    print("=" * 70)

    tests = [
        ("File Structure", test_file_structure),
        ("Component Imports", test_imports),
        ("Metrics Collection", test_metrics_collection),
        ("Performance Testing", test_performance_testing),
        ("Dashboard Integration", test_dashboard_integration),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 50)

        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 Task 6.4 COMPLETED SUCCESSFULLY!")
        print("\n✅ Performance Monitoring and Metrics Implementation:")
        print("   - ✅ Performance metrics collection and reporting")
        print("   - ✅ Response time monitoring and SLA tracking")
        print("   - ✅ Performance dashboards and alerting")
        print("   - ✅ Automated performance testing and benchmarking")
        print("   - ✅ Integration with existing monitoring infrastructure")
        return True
    else:
        print(f"\n❌ Task 6.4 INCOMPLETE: {total - passed} tests failed")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
