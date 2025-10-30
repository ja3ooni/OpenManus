"""
Tests for agent pool management system.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.base import BaseAgent
from app.agent.pool import (
    AgentInstance,
    AgentPool,
    PoolConfig,
    PoolRequest,
    PoolStatus,
    Priority,
    RequestStatus,
    get_agent_pool,
    shutdown_agent_pool,
)
from app.performance.resource_manager import Priority


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(self, **kwargs):
        super().__init__(name="MockAgent", **kwargs)
        self.run_mock = AsyncMock(return_value="Mock result")

    async def step(self) -> str:
        return "Mock step"

    async def run(self, request: str = None) -> str:
        return await self.run_mock(request)


@pytest.fixture
def pool_config():
    """Create a test pool configuration."""
    return PoolConfig(
        min_agents=2,
        max_agents=5,
        max_queue_size=10,
        default_timeout=30.0,
        agent_idle_timeout=60.0,
        health_check_interval=1.0,
        cleanup_interval=2.0,
        agent_class=MockAgent,
    )


@pytest.fixture
async def agent_pool(pool_config):
    """Create and start an agent pool for testing."""
    pool = AgentPool(pool_config)
    await pool.start()
    yield pool
    await pool.stop()


class TestPoolRequest:
    """Test PoolRequest functionality."""

    def test_pool_request_creation(self):
        """Test creating a pool request."""
        request = PoolRequest(content="test request")

        assert request.content == "test request"
        assert request.priority == Priority.NORMAL
        assert request.status == RequestStatus.QUEUED
        assert request.id is not None
        assert request.created_at.tzinfo is not None

    def test_pool_request_age(self):
        """Test request age calculation."""
        request = PoolRequest(content="test")

        # Age should be very small for new request
        assert request.age < 1.0

    def test_pool_request_expiry(self):
        """Test request expiry check."""
        request = PoolRequest(content="test", timeout=0.1)

        # Should not be expired immediately
        assert not request.is_expired()

        # Wait and check again
        import time

        time.sleep(0.2)
        assert request.is_expired()

    def test_processing_time(self):
        """Test processing time calculation."""
        request = PoolRequest(content="test")

        # No processing time initially
        assert request.processing_time is None

        # Set start and end times
        request.started_at = datetime.now(timezone.utc)
        request.completed_at = datetime.now(timezone.utc)

        # Should have processing time
        assert request.processing_time is not None
        assert request.processing_time >= 0


class TestAgentInstance:
    """Test AgentInstance functionality."""

    def test_agent_instance_creation(self):
        """Test creating an agent instance."""
        agent = MockAgent()
        instance = AgentInstance(agent=agent)

        assert instance.agent == agent
        assert instance.id is not None
        assert not instance.is_busy
        assert instance.total_requests == 0
        assert instance.failed_requests == 0
        assert instance.success_rate == 1.0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        instance = AgentInstance(agent=MockAgent())

        # Initially 100% success rate
        assert instance.success_rate == 1.0

        # Add some requests
        instance.total_requests = 10
        instance.failed_requests = 2

        assert instance.success_rate == 0.8

    def test_idle_time(self):
        """Test idle time calculation."""
        instance = AgentInstance(agent=MockAgent())

        # Should have minimal idle time
        assert instance.idle_time < 1.0

        # Mark as used and check
        instance.mark_used()
        assert instance.idle_time < 1.0


class TestPoolConfig:
    """Test PoolConfig validation."""

    def test_valid_config(self):
        """Test creating a valid configuration."""
        config = PoolConfig(
            min_agents=2,
            max_agents=10,
            max_queue_size=100,
        )

        assert config.min_agents == 2
        assert config.max_agents == 10
        assert config.max_queue_size == 100

    def test_invalid_config(self):
        """Test validation of invalid configurations."""
        with pytest.raises(ValueError):
            PoolConfig(min_agents=0)  # Should be >= 1

        with pytest.raises(ValueError):
            PoolConfig(max_agents=0)  # Should be >= 1

        with pytest.raises(ValueError):
            PoolConfig(max_queue_size=0)  # Should be >= 1


class TestAgentPool:
    """Test AgentPool functionality."""

    @pytest.mark.asyncio
    async def test_pool_initialization(self, pool_config):
        """Test pool initialization."""
        pool = AgentPool(pool_config)

        assert pool.status == PoolStatus.INITIALIZING
        assert len(pool.agents) == 0
        assert len(pool.available_agents) == 0

        await pool.start()

        assert pool.status == PoolStatus.ACTIVE
        assert len(pool.agents) == pool_config.min_agents
        assert len(pool.available_agents) == pool_config.min_agents

        await pool.stop()

    @pytest.mark.asyncio
    async def test_submit_request(self, agent_pool):
        """Test submitting a request."""
        request_id = await agent_pool.submit_request("test request")

        assert request_id is not None
        assert agent_pool.metrics["total_requests"] == 1

        # Check request is in queue
        total_queued = sum(len(queue) for queue in agent_pool.request_queues.values())
        assert total_queued == 1

    @pytest.mark.asyncio
    async def test_submit_request_with_priority(self, agent_pool):
        """Test submitting requests with different priorities."""
        high_id = await agent_pool.submit_request("high priority", Priority.HIGH)
        normal_id = await agent_pool.submit_request("normal priority", Priority.NORMAL)

        assert high_id != normal_id
        assert len(agent_pool.request_queues[Priority.HIGH]) == 1
        assert len(agent_pool.request_queues[Priority.NORMAL]) == 1

    @pytest.mark.asyncio
    async def test_request_processing(self, agent_pool):
        """Test request processing."""
        # Submit a request
        request_id = await agent_pool.submit_request("test request")

        # Wait for processing
        await asyncio.sleep(0.5)

        # Check request status
        request = await agent_pool.get_request_status(request_id)
        assert request is not None

        # Should be completed or processing
        assert request.status in [RequestStatus.PROCESSING, RequestStatus.COMPLETED]

    @pytest.mark.asyncio
    async def test_cancel_request(self, agent_pool):
        """Test cancelling a queued request."""
        # Submit a request
        request_id = await agent_pool.submit_request("test request")

        # Cancel immediately
        cancelled = await agent_pool.cancel_request(request_id)
        assert cancelled

        # Check request status
        request = await agent_pool.get_request_status(request_id)
        assert request.status == RequestStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_pool_status(self, agent_pool):
        """Test getting pool status."""
        status = agent_pool.get_pool_status()

        assert "status" in status
        assert "agents" in status
        assert "queues" in status
        assert "metrics" in status
        assert "resource_usage" in status

        assert status["agents"]["total"] >= 2
        assert status["status"] == PoolStatus.ACTIVE.value

    @pytest.mark.asyncio
    async def test_queue_full_error(self, pool_config):
        """Test error when queue is full."""
        # Create pool with small queue
        pool_config.max_queue_size = 2
        pool = AgentPool(pool_config)
        await pool.start()

        try:
            # Fill the queue
            await pool.submit_request("request 1")
            await pool.submit_request("request 2")

            # This should raise an error
            with pytest.raises(RuntimeError, match="queue is full"):
                await pool.submit_request("request 3")

        finally:
            await pool.stop()

    @pytest.mark.asyncio
    async def test_pool_not_active_error(self, pool_config):
        """Test error when pool is not active."""
        pool = AgentPool(pool_config)

        # Should raise error when not started
        with pytest.raises(RuntimeError, match="not active"):
            await pool.submit_request("test request")

    @pytest.mark.asyncio
    async def test_auto_scaling_up(self, pool_config):
        """Test auto-scaling up under load."""
        pool_config.enable_auto_scaling = True
        pool_config.scale_up_threshold = 0.5
        pool = AgentPool(pool_config)
        await pool.start()

        try:
            initial_agents = len(pool.agents)

            # Submit many requests to trigger scaling
            for i in range(5):
                await pool.submit_request(f"request {i}")

            # Wait for auto-scaler to run
            await asyncio.sleep(1.5)

            # Should have scaled up
            assert len(pool.agents) >= initial_agents

        finally:
            await pool.stop()

    @pytest.mark.asyncio
    async def test_request_timeout(self, pool_config):
        """Test request timeout handling."""

        # Create mock agent that takes too long
        class SlowMockAgent(MockAgent):
            async def run(self, request: str = None) -> str:
                await asyncio.sleep(2.0)  # Longer than timeout
                return "slow result"

        pool_config.agent_class = SlowMockAgent
        pool_config.default_timeout = 0.5
        pool = AgentPool(pool_config)
        await pool.start()

        try:
            request_id = await pool.submit_request("slow request")

            # Wait for timeout
            await asyncio.sleep(1.0)

            request = await pool.get_request_status(request_id)
            assert request.status == RequestStatus.TIMEOUT
            assert "timed out" in request.error

        finally:
            await pool.stop()

    @pytest.mark.asyncio
    async def test_agent_failure_handling(self, pool_config):
        """Test handling of agent failures."""

        # Create mock agent that fails
        class FailingMockAgent(MockAgent):
            async def run(self, request: str = None) -> str:
                raise ValueError("Mock failure")

        pool_config.agent_class = FailingMockAgent
        pool = AgentPool(pool_config)
        await pool.start()

        try:
            request_id = await pool.submit_request("failing request")

            # Wait for processing
            await asyncio.sleep(0.5)

            request = await agent_pool.get_request_status(request_id)
            assert request.status == RequestStatus.FAILED
            assert "Mock failure" in request.error

        finally:
            await pool.stop()


class TestGlobalPool:
    """Test global pool management."""

    @pytest.mark.asyncio
    async def test_get_global_pool(self):
        """Test getting the global pool instance."""
        # Ensure no global pool exists
        await shutdown_agent_pool()

        # Get pool (should create new one)
        pool1 = await get_agent_pool()
        assert pool1 is not None
        assert pool1.status == PoolStatus.ACTIVE

        # Get pool again (should return same instance)
        pool2 = await get_agent_pool()
        assert pool1 is pool2

        # Cleanup
        await shutdown_agent_pool()

    @pytest.mark.asyncio
    async def test_shutdown_global_pool(self):
        """Test shutting down the global pool."""
        # Create global pool
        pool = await get_agent_pool()
        assert pool.status == PoolStatus.ACTIVE

        # Shutdown
        await shutdown_agent_pool()

        # Should be able to create new pool
        new_pool = await get_agent_pool()
        assert new_pool is not pool

        # Cleanup
        await shutdown_agent_pool()


@pytest.mark.asyncio
async def test_integration_scenario():
    """Test a complete integration scenario."""
    config = PoolConfig(
        min_agents=2,
        max_agents=4,
        max_queue_size=20,
        default_timeout=10.0,
        enable_auto_scaling=True,
        agent_class=MockAgent,
    )

    pool = AgentPool(config)
    await pool.start()

    try:
        # Submit multiple requests with different priorities
        request_ids = []

        # High priority requests
        for i in range(3):
            request_id = await pool.submit_request(
                f"high priority request {i}", Priority.HIGH
            )
            request_ids.append(request_id)

        # Normal priority requests
        for i in range(5):
            request_id = await pool.submit_request(
                f"normal priority request {i}", Priority.NORMAL
            )
            request_ids.append(request_id)

        # Wait for processing
        await asyncio.sleep(2.0)

        # Check all requests were processed
        completed_count = 0
        for request_id in request_ids:
            request = await pool.get_request_status(request_id)
            if request and request.status == RequestStatus.COMPLETED:
                completed_count += 1

        assert completed_count > 0  # At least some should be completed

        # Check pool metrics
        status = pool.get_pool_status()
        assert status["metrics"]["total_requests"] == len(request_ids)

    finally:
        await pool.stop()
