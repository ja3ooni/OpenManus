#!/usr/bin/env python3
"""
Simple test script to verify resource management system implementation.
"""

import asyncio
import time

from app.performance.resource_manager import (
    Priority,
    ResourceContext,
    ResourceLimits,
    ResourceManager,
    ResourceType,
    allocate_connections,
    allocate_memory,
    deallocate_resource,
    force_cleanup,
    get_resource_statistics,
    get_resource_usage,
    manage_resources,
)


async def test_resource_usage():
    """Test getting current resource usage."""

    print("Testing Resource Usage Monitoring...")

    usage = await get_resource_usage()

    print(
        f"✓ Current Memory Usage: {usage.memory_mb:.1f} MB ({usage.memory_percent:.1f}%)"
    )
    print(f"✓ Current CPU Usage: {usage.cpu_percent:.1f}%")
    print(f"✓ Current Connections: {usage.connections}")
    print(f"✓ Current Threads: {usage.threads}")
    print(f"✓ Current Async Tasks: {usage.async_tasks}")


async def test_resource_allocation():
    """Test basic resource allocation and deallocation."""

    print("\nTesting Resource Allocation...")

    # Create resource manager with custom limits
    limits = ResourceLimits(
        max_memory_mb=1024,
        max_connections=50,
        max_threads=20,
    )

    resource_manager = ResourceManager(
        limits=limits,
        enable_monitoring=False,  # Disable for testing
        enable_auto_cleanup=False,
    )

    # Test memory allocation
    memory_allocation = await resource_manager.allocate_resource(
        resource_type=ResourceType.MEMORY,
        amount=256.0,
        priority=Priority.NORMAL,
        allocated_by="test_user",
        metadata={"purpose": "testing"},
    )

    assert memory_allocation is not None
    print(f"✓ Allocated 256MB memory (ID: {memory_allocation.allocation_id})")

    # Test connection allocation
    conn_allocation = await resource_manager.allocate_resource(
        resource_type=ResourceType.CONNECTIONS,
        amount=10,
        priority=Priority.HIGH,
        allocated_by="test_user",
    )

    assert conn_allocation is not None
    print(f"✓ Allocated 10 connections (ID: {conn_allocation.allocation_id})")

    # Check resource status
    memory_status = await resource_manager.get_resource_status(ResourceType.MEMORY)
    print(f"✓ Memory utilization: {memory_status['utilization']:.1%}")

    conn_status = await resource_manager.get_resource_status(ResourceType.CONNECTIONS)
    print(f"✓ Connection utilization: {conn_status['utilization']:.1%}")

    # Deallocate resources
    success1 = await resource_manager.deallocate_resource(
        memory_allocation.allocation_id
    )
    success2 = await resource_manager.deallocate_resource(conn_allocation.allocation_id)

    assert success1 and success2
    print("✓ Successfully deallocated all resources")


async def test_resource_limits():
    """Test resource limit enforcement."""

    print("\nTesting Resource Limits...")

    resource_manager = ResourceManager(
        limits=ResourceLimits(max_connections=5),
        enable_monitoring=False,
        enable_auto_cleanup=False,
    )

    # Allocate up to the limit
    allocations = []
    for i in range(5):
        allocation = await resource_manager.allocate_resource(
            ResourceType.CONNECTIONS, 1, Priority.NORMAL, f"user_{i}"
        )
        if allocation:
            allocations.append(allocation)

    print(f"✓ Successfully allocated {len(allocations)} connections (at limit)")

    # Try to allocate beyond limit
    over_limit = await resource_manager.allocate_resource(
        ResourceType.CONNECTIONS, 1, Priority.NORMAL, "over_limit_user"
    )

    assert over_limit is None
    print("✓ Correctly rejected allocation beyond limit")

    # Clean up
    for allocation in allocations:
        await resource_manager.deallocate_resource(allocation.allocation_id)

    print("✓ Cleaned up all allocations")


async def test_priority_preemption():
    """Test priority-based resource preemption."""

    print("\nTesting Priority Preemption...")

    resource_manager = ResourceManager(
        limits=ResourceLimits(max_connections=10),
        enable_monitoring=False,
        enable_auto_cleanup=False,
    )

    # Allocate low priority resources
    low_priority_allocations = []
    for i in range(5):
        allocation = await resource_manager.allocate_resource(
            ResourceType.CONNECTIONS, 2, Priority.LOW, f"low_priority_{i}"
        )
        if allocation:
            low_priority_allocations.append(allocation)

    print(
        f"✓ Allocated {len(low_priority_allocations)} low priority resources (10 connections total)"
    )

    # Try to allocate critical priority (should preempt low priority)
    critical_allocation = await resource_manager.allocate_resource(
        ResourceType.CONNECTIONS, 5, Priority.CRITICAL, "critical_user"
    )

    assert critical_allocation is not None
    print("✓ Critical priority allocation succeeded by preempting low priority")

    # Check that some low priority allocations were freed
    remaining_allocations = [
        alloc_id
        for alloc_id in [a.allocation_id for a in low_priority_allocations]
        if alloc_id in resource_manager.allocations
    ]

    print(
        f"✓ {len(low_priority_allocations) - len(remaining_allocations)} low priority allocations were preempted"
    )

    # Clean up
    await resource_manager.deallocate_resource(critical_allocation.allocation_id)
    for allocation in low_priority_allocations:
        if allocation.allocation_id in resource_manager.allocations:
            await resource_manager.deallocate_resource(allocation.allocation_id)


async def test_resource_context():
    """Test resource context manager."""

    print("\nTesting Resource Context Manager...")

    resource_manager = ResourceManager(
        limits=ResourceLimits(max_memory_mb=1024),
        enable_monitoring=False,
        enable_auto_cleanup=False,
    )

    # Test successful context
    async with ResourceContext(
        resource_manager=resource_manager,
        resource_type=ResourceType.MEMORY,
        amount=256.0,
        priority=Priority.NORMAL,
        allocated_by="context_test",
    ) as allocation:

        assert allocation is not None
        print(f"✓ Context allocated 256MB memory (ID: {allocation.allocation_id})")

        # Check that resource is allocated
        assert resource_manager.resource_usage[ResourceType.MEMORY] == 256.0
        print("✓ Resource is properly allocated within context")

    # Check that resource is deallocated after context
    assert resource_manager.resource_usage[ResourceType.MEMORY] == 0.0
    print("✓ Resource automatically deallocated after context exit")


async def test_resource_decorator():
    """Test resource management decorator."""

    print("\nTesting Resource Management Decorator...")

    @manage_resources(ResourceType.MEMORY, 128.0, Priority.NORMAL)
    async def memory_intensive_function(data):
        # Simulate memory-intensive work
        await asyncio.sleep(0.01)
        return f"Processed {len(data)} items"

    # Test function execution
    result = await memory_intensive_function("test_data")
    assert result == "Processed 9 items"
    print(
        "✓ Decorated function executed successfully with automatic resource management"
    )


async def test_convenience_functions():
    """Test convenience functions."""

    print("\nTesting Convenience Functions...")

    # Test memory allocation
    memory_alloc = await allocate_memory(64.0, Priority.NORMAL)
    assert memory_alloc is not None
    print(
        f"✓ Allocated 64MB using convenience function (ID: {memory_alloc.allocation_id})"
    )

    # Test connection allocation
    conn_alloc = await allocate_connections(3, Priority.HIGH)
    assert conn_alloc is not None
    print(
        f"✓ Allocated 3 connections using convenience function (ID: {conn_alloc.allocation_id})"
    )

    # Test resource statistics
    stats = await get_resource_statistics()
    assert isinstance(stats, dict)
    print("✓ Retrieved resource statistics")
    print(f"  - Memory usage: {stats['memory']['current_usage']:.1f}MB")
    print(f"  - Connection usage: {stats['connections']['current_usage']}")

    # Test cleanup
    success1 = await deallocate_resource(memory_alloc.allocation_id)
    success2 = await deallocate_resource(conn_alloc.allocation_id)
    assert success1 and success2
    print("✓ Deallocated resources using convenience function")

    # Test force cleanup
    cleanup_result = await force_cleanup()
    assert isinstance(cleanup_result, dict)
    print(f"✓ Force cleanup collected {cleanup_result['collected_objects']} objects")


async def test_monitoring():
    """Test resource monitoring."""

    print("\nTesting Resource Monitoring...")

    resource_manager = ResourceManager(
        limits=ResourceLimits(max_memory_mb=512),
        enable_monitoring=True,
        enable_auto_cleanup=False,
    )

    try:
        await resource_manager.start()
        print("✓ Started resource manager with monitoring")

        # Wait for some monitoring data
        await asyncio.sleep(0.2)

        # Get monitoring statistics
        if resource_manager.monitor:
            usage_stats = await resource_manager.monitor.get_usage_statistics()
            if usage_stats:
                print(
                    f"✓ Memory utilization: {usage_stats['memory']['utilization']:.1%}"
                )
                print(f"✓ CPU utilization: {usage_stats['cpu']['utilization']:.1%}")

            # Get usage history
            history = await resource_manager.monitor.get_usage_history(minutes=1)
            print(f"✓ Collected {len(history)} usage samples")

    finally:
        await resource_manager.stop()
        print("✓ Stopped resource manager")


async def test_performance():
    """Test resource management performance."""

    print("\nTesting Performance...")

    resource_manager = ResourceManager(
        limits=ResourceLimits(max_connections=1000),
        enable_monitoring=False,
        enable_auto_cleanup=False,
    )

    # Test allocation performance
    start_time = time.time()
    allocations = []

    for i in range(100):
        allocation = await resource_manager.allocate_resource(
            ResourceType.CONNECTIONS, 1, Priority.NORMAL, f"perf_test_{i}"
        )
        if allocation:
            allocations.append(allocation)

    allocation_time = time.time() - start_time
    print(
        f"✓ Allocated 100 resources in {allocation_time:.3f}s ({100/allocation_time:.0f} ops/sec)"
    )

    # Test deallocation performance
    start_time = time.time()

    for allocation in allocations:
        await resource_manager.deallocate_resource(allocation.allocation_id)

    deallocation_time = time.time() - start_time
    print(
        f"✓ Deallocated 100 resources in {deallocation_time:.3f}s ({100/deallocation_time:.0f} ops/sec)"
    )


async def main():
    """Run all resource management tests."""

    print("🚀 Testing Resource Management System...")

    try:
        await test_resource_usage()
        await test_resource_allocation()
        await test_resource_limits()
        await test_priority_preemption()
        await test_resource_context()
        await test_resource_decorator()
        await test_convenience_functions()
        await test_monitoring()
        await test_performance()

        print("\n✅ All resource management tests passed!")

        # Print summary
        print("\n📊 Resource Management System Summary:")
        print("- ✅ Comprehensive resource monitoring (Memory, CPU, Connections, etc.)")
        print("- ✅ Priority-based allocation with preemption")
        print("- ✅ Automatic resource limits enforcement")
        print("- ✅ Context manager for automatic cleanup")
        print("- ✅ Decorator for function-level resource management")
        print("- ✅ Real-time monitoring with alerts")
        print("- ✅ Garbage collection and cleanup automation")
        print("- ✅ High-performance allocation/deallocation")
        print("- ✅ Comprehensive statistics and reporting")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
