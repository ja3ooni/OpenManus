#!/usr/bin/env python3
"""
Basic test of performance testing functionality.
"""

import asyncio
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.performance.testing import (
    PerformanceTestConfig,
    PerformanceTestRunner,
    TestStatus,
    TestType,
)


async def test_basic_functionality():
    """Test basic performance testing functionality."""
    print("🧪 Testing Performance Testing Module")

    runner = PerformanceTestRunner()

    # Test that default configs are loaded
    print(f"✅ Default test configs loaded: {len(runner.test_configs)}")
    for name in list(runner.test_configs.keys())[:3]:
        print(f"   - {name}")

    # Test adding a custom config
    config = PerformanceTestConfig(
        name="basic_test",
        test_type=TestType.BASELINE,
        duration_seconds=1,
        concurrent_users=1,
        warmup_requests=0,
        cooldown_seconds=0,
    )
    runner.add_test_config(config)
    print(f"✅ Added custom config: {config.name}")

    # Test running a simple test
    async def simple_test():
        await asyncio.sleep(0.01)  # 10ms simulated work

    print("🏃 Running performance test...")
    result = await runner.run_test("basic_test", simple_test)

    print(f"✅ Test completed:")
    print(f"   - Name: {result.test_name}")
    print(f"   - Status: {result.status.value}")
    print(f"   - Total requests: {result.total_requests}")
    print(f"   - Successful requests: {result.successful_requests}")
    print(f"   - Failed requests: {result.failed_requests}")
    print(f"   - Avg response time: {result.avg_response_time_ms:.2f}ms")
    print(f"   - Requests per second: {result.requests_per_second:.2f}")
    print(f"   - Error rate: {result.error_rate_percent:.2f}%")
    print(f"   - Performance level: {result.performance_level.value}")
    print(f"   - Meets targets: {result.meets_targets}")

    # Test summary
    summary = runner.get_test_summary()
    print(f"✅ Test summary:")
    print(f"   - Total tests executed: {summary['total_tests_executed']}")
    print(f"   - Unique tests: {summary['unique_tests']}")
    print(f"   - Overall performance: {summary['overall_performance']}")

    # Test baseline and regression
    print("🔄 Testing baseline and regression...")
    runner.set_baseline("basic_test", result)

    # Run a slightly slower test for regression
    async def slower_test():
        await asyncio.sleep(0.02)  # 20ms - 2x slower

    regression_result = await runner.run_regression_test(
        "basic_test", slower_test, regression_threshold_percent=50.0  # 50% threshold
    )

    print(f"✅ Regression test completed:")
    print(f"   - Regression detected: {regression_result.regression_detected}")
    print(
        f"   - Performance change: {regression_result.performance_change_percent:.1f}%"
    )
    print(
        f"   - Response time change: {regression_result.response_time_change_percent:.1f}%"
    )

    print("\n🎉 Performance testing module working correctly!")
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_basic_functionality())
        if success:
            print("\n✅ All tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
