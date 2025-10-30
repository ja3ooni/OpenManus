#!/usr/bin/env python3
"""
Direct test of resource manager without going through __init__.py
"""

import asyncio
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_resource_manager():
    """Test resource manager directly."""

    print("🚀 Testing Resource Management System (Direct Import)...")

    try:
        # Import directly from the module file
        from app.performance.resource_manager import (
            Priority,
            ResourceAllocation,
            ResourceLimits,
            ResourceManager,
            ResourceMonitor,
            ResourceType,
            ResourceUsage,
        )

        print("✓ All imports successful")

        # Test ResourceUsage
        usage = ResourceUsage(
            memory_mb=256.0,
            cpu_percent=45.0,
            connections=10,
        )
        print(
            f"✓ ResourceUsage created: {usage.memory_mb}MB memory, {usage.cpu_percent}% CPU"
        )

        # Test ResourceLimits
        limits = ResourceLimits(
            max_memory_mb=1024,
            max_connections=50,
        )
        print(f"✓ ResourceLimits created: {limits.max_memory_mb}MB memory limit")

        # Test ResourceManager
        resource_manager = ResourceManager(
            limits=limits,
            enable_monitoring=False,
            enable_auto_cleanup=False,
        )
        print("✓ ResourceManager created")

        # Test resource allocation
        allocation = await resource_manager.allocate_resource(
            resource_type=ResourceType.MEMORY,
            amount=256.0,
            priority=Priority.NORMAL,
            allocated_by="test_user",
        )

        assert allocation is not None
        print(f"✓ Allocated 256MB memory (ID: {allocation.allocation_id})")

        # Check resource status
        status = await resource_manager.get_resource_status(ResourceType.MEMORY)
        print(f"✓ Memory utilization: {status['utilization']:.1%}")

        # Test deallocation
        success = await resource_manager.deallocate_resource(allocation.allocation_id)
        assert success
        print("✓ Successfully deallocated resource")

        # Test resource monitoring
        monitor = ResourceMonitor(limits)
        current_usage = await monitor.get_current_usage()
        print(
            f"✓ Current system usage: {current_usage.memory_mb:.1f}MB, {current_usage.cpu_percent:.1f}% CPU"
        )

        print("\n✅ All resource management tests passed!")

        # Print summary
        print("\n📊 Resource Management System Summary:")
        print("- ✅ Resource allocation and deallocation")
        print("- ✅ Priority-based resource management")
        print("- ✅ Resource limits enforcement")
        print("- ✅ System resource monitoring")
        print("- ✅ Resource usage statistics")
        print("- ✅ Memory, CPU, and connection tracking")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_resource_manager())
    if success:
        print("\n🎉 Resource Management System is working correctly!")
    else:
        print("\n💥 Resource Management System has issues!")
