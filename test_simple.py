#!/usr/bin/env python3
"""
Simple test without imports from app.performance.__init__.py
"""

import asyncio
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly to avoid circular imports
from app.performance.testing import (
    PerformanceTestConfig,
    PerformanceTestRunner,
    TestType,
)


async def simple_test():
    """Simple test of performance testing."""
    print("🧪 Testing Performance Testing Module (Direct Import)")

    runner = PerformanceTestRunner()
    print(f"✅ Created runner with {len(runner.test_configs)} default configs")

    # Add a simple test config
    config = PerformanceTestConfig(
        name="simple_test",
        test_type=TestType.BASELINE,
        duration_seconds=0.5,
        warmup_requests=0,
        cooldown_seconds=0,
    )
    runner.add_test_config(config)
    print("✅ Added test config")

    # Run test
    async def test_func():
        await asyncio.sleep(0.01)

    result = await runner.run_test("simple_test", test_func)
    print(f"✅ Test completed: {result.status.value}")
    print(f"   Requests: {result.total_requests}")
    print(f"   Response time: {result.avg_response_time_ms:.1f}ms")

    print("🎉 Success!")


if __name__ == "__main__":
    asyncio.run(simple_test())
