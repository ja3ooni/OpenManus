"""
Tests for the resource management system.
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.performance.resource_manager import (
    Priority,
    ResourceAllocation,
    ResourceContext,
    ResourceLimits,
    ResourceManager,
    ResourceMonitor,
    ResourceStatus,
    ResourceType,
    ResourceUsage,
    allocate_connections,
    allocate_memory,
    deallocate_resource,
    force_cleanup,
    get_resource_statistics,
    get_resource_usage,
    manage_resources,
)


class TestResourceUsage:
    """Test resource usage data model."""

    def test_resource_usage_creation(self):
        """Test creating resource usage object."""

        usage = ResourceUsage(
            memory_mb=512.0,
            memory_percent=25.0,
            cpu_percent=45.0,
            connections=10,
            file_handles=50,
            threads=5,
            async_tasks=20,
        )

        assert usage.memory_mb == 512.0
        assert usage.memory_percent == 25.0
        assert usage.cpu_percent == 45.0
        assert usage.connections == 10
        assert usage.file_handles == 50
        assert usage.threads == 5
        assert usage.async_tasks == 20
        assert isinstance(usage.timestamp, datetime)

    def test_resource_usage_to_dict(self):
        """Test converting resource usage to dictionary."""

        usage = ResourceUsage(
            memory_mb=256.0,
            cpu_percent=30.0,
            connections=5,
        )

        usage_dict = usage.to_dict()

        assert usage_dict["memory_mb"] == 256.0
        assert usage_dict["cpu_percent"] == 30.0
        assert usage_dict["connections"] == 5
        assert "timestamp" in usage_dict
        assert isinstance(usage_dict["timestamp"], str)


class TestResourceLimits:
    """Test resource limits configuration."""

    def test_default_limits(self):
        """Test default resource limits."""

        limits = ResourceLimits()

        assert limits.max_memory_mb == 2048
        assert limits.max_cpu_percent == 80.0
        assert limits.max_connections == 1000
        assert limits.max_file_handles == 1000
        assert limits.max_threads == 100
        assert limits.max_async_tasks == 1000

        # Check thresholds
        assert limits.memory_warning_threshold == 0.8
        assert limits.memory_critical_threshold == 0.9

    def test_custom_limits(self):
        """Test custom resource limits."""

        limits = ResourceLimits(
            max_memory_mb=4096,
            max_cpu_percent=90.0,
            max_connections=2000,
            memory_warning_threshold=0.7,
        )

        assert limits.max_memory_mb == 4096
        assert limits.max_cpu_percent == 90.0
        assert limits.max_connections == 2000
        assert limits.memory_warning_threshold == 0.7


class TestResourceAllocation:
    """Test resource allocation data model."""

    def test_resource_allocation_creation(self):
        """Test creating resource allocation."""

        allocation = ResourceAllocation(
            allocation_id="test_123",
            resource_type=ResourceType.MEMORY,
            amount=512.0,
            priority=Priority.HIGH,
            allocated_at=datetime.now(timezone.utc),
            allocated_by="test_user",
            metadata={"purpose": "testing"},
        )

        assert allocation.allocation_id == "test_123"
        assert allocation.resource_type == ResourceType.MEMORY
        assert allocation.amount == 512.0
        assert allocation.priority == Priority.HIGH
        assert allocation.allocated_by == "test_user"
        assert allocation.metadata["purpose"] == "testing"


class TestResourceMonitor:
    """Test resource monitoring functionality."""

    @pytest.fixture
    def resource_limits(self):
        """Create resource limits for testing."""
        return ResourceLimits(
            max_memory_mb=1024,
            max_cpu_percent=80.0,
            max_connections=100,
        )

    @pytest.fixture
    def resource_monitor(self, resource_limits):
        """Create resource monitor for testing."""
        return ResourceMonitor(
            limits=resource_limits,
            monitoring_interval=0.1,  # Fast for testing
            history_size=10,
        )

    @pytest.mark.asyncio
    async def test_get_current_usage(self, resource_monitor):
        """Test getting current resource usage."""

        usage = await resource_monitor.get_current_usage()

        assert isinstance(usage, ResourceUsage)
        assert usage.memory_mb >= 0
        assert usage.memory_percent >= 0
        assert usage.cpu_percent >= 0
        assert usage.connections >= 0
        assert usage.threads >= 0
        assert isinstance(usage.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, resource_monitor):
        """Test starting and stopping monitoring."""

        # Initially not monitoring
        assert not resource_monitor._monitoring

        # Start monitoring
        await resource_monitor.start_monitoring()
        assert resource_monitor._monitoring
        assert resource_monitor._monitor_task is not None

        # Wait a bit for some samples
        await asyncio.sleep(0.3)

        # Should have some usage history
        assert len(resource_monitor.usage_history) > 0

        # Stop monitoring
        await resource_monitor.stop_monitoring()
        assert not resource_monitor._monitoring

    @pytest.mark.asyncio
    async def test_alert_callbacks(self, resource_monitor):
        """Test resource alert callbacks."""

        alert_messages = []

        async def alert_callback(message, usage):
            alert_messages.append(message)

        resource_monitor.add_alert_callback(alert_callback)

        # Mock high memory usage
        with patch.object(resource_monitor, "get_current_usage") as mock_usage:
            mock_usage.return_value = ResourceUsage(
                memory_mb=950.0,  # Above warning threshold (80% of 1024MB)
                cpu_percent=70.0,
                connections=5,
            )

            await resource_monitor.start_monitoring()
            await asyncio.sleep(0.2)  # Wait for monitoring cycle
            await resource_monitor.stop_monitoring()

        # Should have received alert
        assert len(alert_messages) > 0
        assert any("WARNING" in msg for msg in alert_messages)

    @pytest.mark.asyncio
    async def test_usage_history(self, resource_monitor):
        """Test usage history tracking."""

        await resource_monitor.start_monitoring()
        await asyncio.sleep(0.3)  # Collect some samples
        await resource_monitor.stop_monitoring()

        # Get recent history
        history = await resource_monitor.get_usage_history(minutes=1)
        assert len(history) > 0

        # All entries should be recent
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=1)
        for usage in history:
            assert usage.timestamp >= cutoff_time

    @pytest.mark.asyncio
    async def test_usage_statistics(self, resource_monitor):
        """Test usage statistics calculation."""

        # Add some mock usage data
        for i in range(5):
            usage = ResourceUsage(
                memory_mb=100.0 + i * 10,
                cpu_percent=20.0 + i * 5,
                connections=i,
            )
            resource_monitor.usage_history.append(usage)

        stats = await resource_monitor.get_usage_statistics()

        assert "memory" in stats
        assert "cpu" in stats
        assert "connections" in stats

        # Check memory stats
        memory_stats = stats["memory"]
        assert memory_stats["current_mb"] == 140.0  # Last value
        assert memory_stats["max_mb"] == 140.0
        assert memory_stats["min_mb"] == 100.0
        assert memory_stats["limit_mb"] == 1024
        assert 0 <= memory_stats["utilization"] <= 1


class TestResourceManager:
    """Test resource manager functionality."""

    @pytest.fixture
    def resource_limits(self):
        """Create resource limits for testing."""
        return ResourceLimits(
            max_memory_mb=1024,
            max_cpu_percent=80.0,
            max_connections=10,
            max_threads=5,
        )

    @pytest.fixture
    def resource_manager(self, resource_limits):
        """Create resource manager for testing."""
        return ResourceManager(
            limits=resource_limits,
            enable_monitoring=False,  # Disable for testing
            enable_auto_cleanup=False,  # Disable for testing
        )

    @pytest.mark.asyncio
    async def test_allocate_deallocate_resource(self, resource_manager):
        """Test basic resource allocation and deallocation."""

        # Allocate memory
        allocation = await resource_manager.allocate_resource(
            resource_type=ResourceType.MEMORY,
            amount=256.0,
            priority=Priority.NORMAL,
            allocated_by="test",
        )

        assert allocation is not None
        assert allocation.resource_type == ResourceType.MEMORY
        assert allocation.amount == 256.0
        assert allocation.priority == Priority.NORMAL

        # Check resource usage
        assert resource_manager.resource_usage[ResourceType.MEMORY] == 256.0

        # Deallocate
        success = await resource_manager.deallocate_resource(allocation.allocation_id)
        assert success is True

        # Check resource usage is back to zero
        assert resource_manager.resource_usage[ResourceType.MEMORY] == 0.0

    @pytest.mark.asyncio
    async def test_allocation_limit_enforcement(self, resource_manager):
        """Test that allocation limits are enforced."""

        # Try to allocate more than the limit
        allocation = await resource_manager.allocate_resource(
            resource_type=ResourceType.CONNECTIONS,
            amount=15,  # More than limit of 10
            priority=Priority.NORMAL,
            allocated_by="test",
        )

        # Should fail
        assert allocation is None
        assert resource_manager.resource_usage[ResourceType.CONNECTIONS] == 0

    @pytest.mark.asyncio
    async def test_critical_priority_preemption(self, resource_manager):
        """Test that critical priority can preempt low priority allocations."""

        # Allocate low priority resources up to limit
        low_priority_allocations = []
        for i in range(5):
            allocation = await resource_manager.allocate_resource(
                resource_type=ResourceType.CONNECTIONS,
                amount=2,
                priority=Priority.LOW,
                allocated_by=f"low_priority_{i}",
            )
            if allocation:
                low_priority_allocations.append(allocation)

        # Should have allocated 10 connections (5 * 2)
        assert resource_manager.resource_usage[ResourceType.CONNECTIONS] == 10

        # Try to allocate critical priority (should preempt low priority)
        critical_allocation = await resource_manager.allocate_resource(
            resource_type=ResourceType.CONNECTIONS,
            amount=5,
            priority=Priority.CRITICAL,
            allocated_by="critical_user",
        )

        # Should succeed by freeing low priority resources
        assert critical_allocation is not None
        assert critical_allocation.priority == Priority.CRITICAL

    @pytest.mark.asyncio
    async def test_resource_status(self, resource_manager):
        """Test getting resource status."""

        # Allocate some resources
        await resource_manager.allocate_resource(
            ResourceType.MEMORY, 512.0, Priority.NORMAL, "test"
        )
        await resource_manager.allocate_resource(
            ResourceType.CONNECTIONS, 8, Priority.HIGH, "test"
        )

        # Get memory status
        memory_status = await resource_manager.get_resource_status(ResourceType.MEMORY)
        assert memory_status["resource_type"] == "memory"
        assert memory_status["current_usage"] == 512.0
        assert memory_status["limit"] == 1024
        assert memory_status["utilization"] == 0.5
        assert memory_status["status"] == ResourceStatus.AVAILABLE.value

        # Get connections status (should be throttled at 80% usage)
        conn_status = await resource_manager.get_resource_status(
            ResourceType.CONNECTIONS
        )
        assert conn_status["current_usage"] == 8
        assert conn_status["utilization"] == 0.8
        assert conn_status["status"] == ResourceStatus.THROTTLED.value

    @pytest.mark.asyncio
    async def test_all_resource_status(self, resource_manager):
        """Test getting status of all resources."""

        # Allocate some resources
        await resource_manager.allocate_resource(
            ResourceType.MEMORY, 256.0, Priority.NORMAL, "test"
        )

        status = await resource_manager.get_all_resource_status()

        # Should have status for all resource types
        for resource_type in ResourceType:
            assert resource_type.value in status

        # Memory should show usage
        assert status["memory"]["current_usage"] == 256.0

    @pytest.mark.asyncio
    async def test_cleanup_callbacks(self, resource_manager):
        """Test cleanup callback registration and execution."""

        cleanup_called = False

        async def cleanup_callback():
            nonlocal cleanup_called
            cleanup_called = True

        # Allocate resource
        allocation = await resource_manager.allocate_resource(
            ResourceType.MEMORY, 256.0, Priority.LOW, "test"
        )

        # Register cleanup callback
        resource_manager.register_cleanup_callback(
            allocation.allocation_id, cleanup_callback
        )

        # Force cleanup of low priority resources
        await resource_manager._free_low_priority_resources(ResourceType.MEMORY, 100.0)

        # Cleanup should have been called
        assert cleanup_called is True

    @pytest.mark.asyncio
    async def test_garbage_collection(self, resource_manager):
        """Test forced garbage collection."""

        result = await resource_manager.force_garbage_collection()

        assert "collected_objects" in result
        assert "cleaned_references" in result
        assert "remaining_references" in result
        assert isinstance(result["collected_objects"], int)


class TestResourceContext:
    """Test resource context manager."""

    @pytest.fixture
    def resource_manager(self):
        """Create resource manager for testing."""
        return ResourceManager(
            limits=ResourceLimits(max_memory_mb=1024),
            enable_monitoring=False,
            enable_auto_cleanup=False,
        )

    @pytest.mark.asyncio
    async def test_resource_context_success(self, resource_manager):
        """Test successful resource context usage."""

        async with ResourceContext(
            resource_manager=resource_manager,
            resource_type=ResourceType.MEMORY,
            amount=256.0,
            priority=Priority.NORMAL,
        ) as allocation:

            assert allocation is not None
            assert allocation.resource_type == ResourceType.MEMORY
            assert allocation.amount == 256.0

            # Resource should be allocated
            assert resource_manager.resource_usage[ResourceType.MEMORY] == 256.0

        # Resource should be deallocated after context exit
        assert resource_manager.resource_usage[ResourceType.MEMORY] == 0.0

    @pytest.mark.asyncio
    async def test_resource_context_failure(self, resource_manager):
        """Test resource context when allocation fails."""

        async with ResourceContext(
            resource_manager=resource_manager,
            resource_type=ResourceType.MEMORY,
            amount=2048.0,  # More than limit
            priority=Priority.NORMAL,
        ) as allocation:

            # Allocation should fail
            assert allocation is None

        # No resources should be allocated
        assert resource_manager.resource_usage[ResourceType.MEMORY] == 0.0


class TestResourceDecorator:
    """Test resource management decorator."""

    @pytest.mark.asyncio
    async def test_manage_resources_decorator(self):
        """Test resource management decorator."""

        @manage_resources(ResourceType.MEMORY, 256.0, Priority.NORMAL)
        async def test_function():
            return "success"

        # Function should execute successfully
        result = await test_function()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_manage_resources_decorator_failure(self):
        """Test resource management decorator when allocation fails."""

        @manage_resources(ResourceType.MEMORY, 999999.0, Priority.NORMAL)  # Huge amount
        async def test_function():
            return "success"

        # Function should raise RuntimeError due to allocation failure
        with pytest.raises(RuntimeError, match="Failed to allocate"):
            await test_function()


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_get_resource_usage(self):
        """Test getting resource usage."""

        usage = await get_resource_usage()

        assert isinstance(usage, ResourceUsage)
        assert usage.memory_mb >= 0
        assert usage.cpu_percent >= 0

    @pytest.mark.asyncio
    async def test_get_resource_statistics(self):
        """Test getting resource statistics."""

        stats = await get_resource_statistics()

        assert isinstance(stats, dict)
        # Should have status for all resource types
        for resource_type in ResourceType:
            assert resource_type.value in stats

    @pytest.mark.asyncio
    async def test_allocate_memory(self):
        """Test memory allocation convenience function."""

        allocation = await allocate_memory(128.0, Priority.NORMAL)

        assert allocation is not None
        assert allocation.resource_type == ResourceType.MEMORY
        assert allocation.amount == 128.0

        # Clean up
        await deallocate_resource(allocation.allocation_id)

    @pytest.mark.asyncio
    async def test_allocate_connections(self):
        """Test connection allocation convenience function."""

        allocation = await allocate_connections(5, Priority.NORMAL)

        assert allocation is not None
        assert allocation.resource_type == ResourceType.CONNECTIONS
        assert allocation.amount == 5

        # Clean up
        await deallocate_resource(allocation.allocation_id)

    @pytest.mark.asyncio
    async def test_force_cleanup(self):
        """Test force cleanup convenience function."""

        result = await force_cleanup()

        assert isinstance(result, dict)
        assert "collected_objects" in result


class TestResourceManagerIntegration:
    """Test resource manager integration scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_allocations(self):
        """Test concurrent resource allocations."""

        resource_manager = ResourceManager(
            limits=ResourceLimits(max_connections=20),
            enable_monitoring=False,
            enable_auto_cleanup=False,
        )

        async def allocate_worker(worker_id):
            allocation = await resource_manager.allocate_resource(
                ResourceType.CONNECTIONS, 2, Priority.NORMAL, f"worker_{worker_id}"
            )
            if allocation:
                await asyncio.sleep(0.1)  # Simulate work
                await resource_manager.deallocate_resource(allocation.allocation_id)
                return True
            return False

        # Run multiple workers concurrently
        tasks = [allocate_worker(i) for i in range(15)]
        results = await asyncio.gather(*tasks)

        # Most should succeed (some might fail due to limits)
        successful = sum(results)
        assert successful >= 5  # At least some should succeed

        # All resources should be deallocated
        assert resource_manager.resource_usage[ResourceType.CONNECTIONS] == 0

    @pytest.mark.asyncio
    async def test_resource_lifecycle_with_monitoring(self):
        """Test complete resource lifecycle with monitoring."""

        resource_manager = ResourceManager(
            limits=ResourceLimits(max_memory_mb=512),
            enable_monitoring=True,
            enable_auto_cleanup=False,
        )

        try:
            await resource_manager.start()

            # Allocate resources
            allocation1 = await resource_manager.allocate_resource(
                ResourceType.MEMORY, 128.0, Priority.NORMAL, "test1"
            )
            allocation2 = await resource_manager.allocate_resource(
                ResourceType.MEMORY, 256.0, Priority.HIGH, "test2"
            )

            assert allocation1 is not None
            assert allocation2 is not None

            # Wait for monitoring to collect data
            await asyncio.sleep(0.2)

            # Get status
            status = await resource_manager.get_all_resource_status()
            assert status["memory"]["current_usage"] == 384.0

            # Clean up
            await resource_manager.deallocate_resource(allocation1.allocation_id)
            await resource_manager.deallocate_resource(allocation2.allocation_id)

        finally:
            await resource_manager.stop()


if __name__ == "__main__":
    pytest.main([__file__])
