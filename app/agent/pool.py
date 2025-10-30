"""
Agent pool management for concurrent operations.

This module provides agent pool management capabilities including:
- Agent pool management for concurrent operations
- Request queuing with priority handling
- Load balancing and request routing capabilities
- Graceful degradation under high load conditions
"""

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
from weakref import WeakSet

from pydantic import BaseModel, Field

from app.agent.base import BaseAgent
from app.logger import logger


class DefaultAgent(BaseAgent):
    """Default agent implementation for pool testing."""

    def __init__(self, **kwargs):
        super().__init__(name="DefaultAgent", **kwargs)

    async def step(self) -> str:
        return "Default agent step completed"


class Priority(Enum):
    """Priority levels for request handling."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class PoolStatus(Enum):
    """Status of the agent pool."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


class RequestStatus(Enum):
    """Status of a request in the pool."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class PoolRequest:
    """Represents a request in the agent pool."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    priority: Priority = Priority.NORMAL
    timeout: float = 300.0  # 5 minutes default
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: RequestStatus = RequestStatus.QUEUED
    result: Optional[str] = None
    error: Optional[str] = None
    agent_id: Optional[str] = None

    def __post_init__(self):
        """Ensure created_at is timezone-aware."""
        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)

    @property
    def age(self) -> float:
        """Get the age of the request in seconds."""
        now = datetime.now(timezone.utc)
        return (now - self.created_at).total_seconds()

    @property
    def processing_time(self) -> Optional[float]:
        """Get the processing time in seconds if completed."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def is_expired(self) -> bool:
        """Check if the request has exceeded its timeout."""
        return self.age > self.timeout


@dataclass
class AgentInstance:
    """Represents an agent instance in the pool."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent: BaseAgent = field(default_factory=DefaultAgent)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_busy: bool = False
    current_request_id: Optional[str] = None
    total_requests: int = 0
    failed_requests: int = 0

    def __post_init__(self):
        """Ensure timestamps are timezone-aware."""
        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)
        if self.last_used.tzinfo is None:
            self.last_used = self.last_used.replace(tzinfo=timezone.utc)

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of this agent instance."""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests

    @property
    def idle_time(self) -> float:
        """Get the idle time in seconds."""
        now = datetime.now(timezone.utc)
        return (now - self.last_used).total_seconds()

    def mark_used(self):
        """Mark the agent as recently used."""
        self.last_used = datetime.now(timezone.utc)


class PoolConfig(BaseModel):
    """Configuration for the agent pool."""

    min_agents: int = Field(
        default=2, ge=1, description="Minimum number of agents in pool"
    )
    max_agents: int = Field(
        default=10, ge=1, description="Maximum number of agents in pool"
    )
    max_queue_size: int = Field(default=1000, ge=1, description="Maximum queue size")
    default_timeout: float = Field(
        default=300.0, gt=0, description="Default request timeout in seconds"
    )
    agent_idle_timeout: float = Field(
        default=600.0, gt=0, description="Agent idle timeout in seconds"
    )
    scale_up_threshold: float = Field(
        default=0.8,
        gt=0,
        le=1,
        description="Queue utilization threshold for scaling up",
    )
    scale_down_threshold: float = Field(
        default=0.3,
        gt=0,
        le=1,
        description="Queue utilization threshold for scaling down",
    )
    health_check_interval: float = Field(
        default=30.0, gt=0, description="Health check interval in seconds"
    )
    cleanup_interval: float = Field(
        default=60.0, gt=0, description="Cleanup interval in seconds"
    )
    enable_auto_scaling: bool = Field(
        default=True, description="Enable automatic scaling"
    )
    enable_load_balancing: bool = Field(
        default=True, description="Enable load balancing"
    )
    agent_class: Type[BaseAgent] = Field(
        default=DefaultAgent, description="Agent class to instantiate"
    )

    class Config:
        arbitrary_types_allowed = True


class AgentPool:
    """
    Agent pool for managing concurrent operations with load balancing and auto-scaling.

    Features:
    - Dynamic agent pool management
    - Priority-based request queuing
    - Load balancing across available agents
    - Auto-scaling based on load
    - Graceful degradation under high load
    - Health monitoring and cleanup
    """

    def __init__(self, config: Optional[PoolConfig] = None):
        """Initialize the agent pool."""
        self.config = config or PoolConfig()
        self.status = PoolStatus.INITIALIZING

        # Agent management
        self.agents: Dict[str, AgentInstance] = {}
        self.available_agents: Set[str] = set()
        self.busy_agents: Set[str] = set()

        # Request management
        self.request_queues: Dict[Priority, deque] = {
            Priority.CRITICAL: deque(),
            Priority.HIGH: deque(),
            Priority.NORMAL: deque(),
            Priority.LOW: deque(),
        }
        self.active_requests: Dict[str, PoolRequest] = {}
        self.completed_requests: deque = deque(
            maxlen=1000
        )  # Keep last 1000 for metrics

        # Monitoring and metrics
        try:
            from app.performance.resource_manager import ResourceManager

            self.resource_manager = ResourceManager()
        except ImportError:
            # Fallback if resource manager is not available
            self.resource_manager = None

        self.metrics = {
            "total_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "timeout_requests": 0,
            "cancelled_requests": 0,
            "average_processing_time": 0.0,
            "queue_sizes": {priority: 0 for priority in Priority},
        }

        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()

        logger.info(f"Initialized agent pool with config: {self.config}")

    async def start(self) -> None:
        """Start the agent pool and background tasks."""
        logger.info("Starting agent pool...")

        try:
            # Initialize minimum agents
            await self._initialize_agents()

            # Start background tasks
            await self._start_background_tasks()

            self.status = PoolStatus.ACTIVE
            logger.info(
                f"Agent pool started successfully with {len(self.agents)} agents"
            )

        except Exception as e:
            self.status = PoolStatus.STOPPED
            logger.error(f"Failed to start agent pool: {e}")
            raise

    async def stop(self) -> None:
        """Stop the agent pool and cleanup resources."""
        logger.info("Stopping agent pool...")
        self.status = PoolStatus.SHUTTING_DOWN

        # Signal shutdown to background tasks
        self._shutdown_event.set()

        # Cancel all background tasks
        for task in self._background_tasks:
            task.cancel()

        # Wait for background tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Cancel all active requests
        for request in self.active_requests.values():
            request.status = RequestStatus.CANCELLED

        # Clear queues
        for queue in self.request_queues.values():
            queue.clear()

        # Cleanup agents
        self.agents.clear()
        self.available_agents.clear()
        self.busy_agents.clear()

        self.status = PoolStatus.STOPPED
        logger.info("Agent pool stopped successfully")

    async def submit_request(
        self,
        content: str,
        priority: Priority = Priority.NORMAL,
        timeout: Optional[float] = None,
    ) -> str:
        """
        Submit a request to the agent pool.

        Args:
            content: The request content
            priority: Request priority
            timeout: Request timeout in seconds

        Returns:
            Request ID for tracking

        Raises:
            RuntimeError: If pool is not active or queue is full
        """
        if self.status != PoolStatus.ACTIVE:
            raise RuntimeError(f"Agent pool is not active (status: {self.status})")

        # Check queue capacity
        total_queued = sum(len(queue) for queue in self.request_queues.values())
        if total_queued >= self.config.max_queue_size:
            raise RuntimeError("Agent pool queue is full")

        # Create request
        request = PoolRequest(
            content=content,
            priority=priority,
            timeout=timeout or self.config.default_timeout,
        )

        # Add to appropriate queue
        self.request_queues[priority].append(request)
        self.metrics["total_requests"] += 1
        self.metrics["queue_sizes"][priority] += 1

        logger.debug(f"Submitted request {request.id} with priority {priority}")
        return request.id

    async def get_request_status(self, request_id: str) -> Optional[PoolRequest]:
        """Get the status of a request."""
        # Check active requests
        if request_id in self.active_requests:
            return self.active_requests[request_id]

        # Check completed requests
        for request in self.completed_requests:
            if request.id == request_id:
                return request

        # Check queued requests
        for queue in self.request_queues.values():
            for request in queue:
                if request.id == request_id:
                    return request

        return None

    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a request if it's still queued."""
        # Remove from queues
        for priority, queue in self.request_queues.items():
            for i, request in enumerate(queue):
                if request.id == request_id:
                    request.status = RequestStatus.CANCELLED
                    del queue[i]
                    self.metrics["cancelled_requests"] += 1
                    self.metrics["queue_sizes"][priority] -= 1
                    logger.debug(f"Cancelled queued request {request_id}")
                    return True

        # Cannot cancel active requests
        return False

    def get_pool_status(self) -> Dict[str, Any]:
        """Get comprehensive pool status information."""
        return {
            "status": self.status.value,
            "agents": {
                "total": len(self.agents),
                "available": len(self.available_agents),
                "busy": len(self.busy_agents),
            },
            "queues": {
                priority.name: len(queue)
                for priority, queue in self.request_queues.items()
            },
            "active_requests": len(self.active_requests),
            "metrics": self.metrics.copy(),
            "resource_usage": await self._get_resource_usage_dict(),
        }

    async def _initialize_agents(self) -> None:
        """Initialize the minimum number of agents."""
        for _ in range(self.config.min_agents):
            await self._create_agent()

    async def _create_agent(self) -> str:
        """Create a new agent instance."""
        agent_instance = AgentInstance(agent=self.config.agent_class())
        self.agents[agent_instance.id] = agent_instance
        self.available_agents.add(agent_instance.id)

        logger.debug(f"Created agent {agent_instance.id}")
        return agent_instance.id

    async def _remove_agent(self, agent_id: str) -> None:
        """Remove an agent instance."""
        if agent_id in self.agents:
            self.available_agents.discard(agent_id)
            self.busy_agents.discard(agent_id)
            del self.agents[agent_id]
            logger.debug(f"Removed agent {agent_id}")

    async def _start_background_tasks(self) -> None:
        """Start background tasks for pool management."""
        tasks = [
            self._request_processor(),
            self._health_monitor(),
            self._cleanup_task(),
        ]

        if self.config.enable_auto_scaling:
            tasks.append(self._auto_scaler())

        for coro in tasks:
            task = asyncio.create_task(coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _request_processor(self) -> None:
        """Process requests from the queue."""
        logger.debug("Started request processor")

        while not self._shutdown_event.is_set():
            try:
                # Get next request with priority
                request = await self._get_next_request()
                if not request:
                    await asyncio.sleep(0.1)
                    continue

                # Get available agent
                agent_id = await self._get_available_agent()
                if not agent_id:
                    # Put request back in queue if no agents available
                    self.request_queues[request.priority].appendleft(request)
                    await asyncio.sleep(0.5)
                    continue

                # Process request
                asyncio.create_task(self._process_request(request, agent_id))

            except Exception as e:
                logger.error(f"Error in request processor: {e}")
                await asyncio.sleep(1.0)

    async def _get_next_request(self) -> Optional[PoolRequest]:
        """Get the next request from queues based on priority."""
        # Process in priority order
        for priority in [
            Priority.CRITICAL,
            Priority.HIGH,
            Priority.NORMAL,
            Priority.LOW,
        ]:
            queue = self.request_queues[priority]
            if queue:
                request = queue.popleft()
                self.metrics["queue_sizes"][priority] -= 1

                # Check if request has expired
                if request.is_expired():
                    request.status = RequestStatus.TIMEOUT
                    self.completed_requests.append(request)
                    self.metrics["timeout_requests"] += 1
                    continue

                return request

        return None

    async def _get_available_agent(self) -> Optional[str]:
        """Get an available agent using load balancing."""
        if not self.available_agents:
            return None

        if not self.config.enable_load_balancing:
            return next(iter(self.available_agents))

        # Load balancing: choose agent with lowest load
        best_agent_id = None
        best_score = float("inf")

        for agent_id in self.available_agents:
            agent = self.agents[agent_id]
            # Score based on success rate and idle time
            score = (1.0 - agent.success_rate) * 100 + max(0, 300 - agent.idle_time)
            if score < best_score:
                best_score = score
                best_agent_id = agent_id

        return best_agent_id

    async def _process_request(self, request: PoolRequest, agent_id: str) -> None:
        """Process a request with the assigned agent."""
        agent_instance = self.agents[agent_id]

        try:
            # Mark agent as busy
            self.available_agents.discard(agent_id)
            self.busy_agents.add(agent_id)
            agent_instance.is_busy = True
            agent_instance.current_request_id = request.id

            # Update request status
            request.status = RequestStatus.PROCESSING
            request.started_at = datetime.now(timezone.utc)
            request.agent_id = agent_id
            self.active_requests[request.id] = request

            logger.debug(f"Processing request {request.id} with agent {agent_id}")

            # Process with timeout
            try:
                result = await asyncio.wait_for(
                    agent_instance.agent.run(request.content), timeout=request.timeout
                )

                request.result = result
                request.status = RequestStatus.COMPLETED
                self.metrics["completed_requests"] += 1

            except asyncio.TimeoutError:
                request.status = RequestStatus.TIMEOUT
                request.error = f"Request timed out after {request.timeout} seconds"
                self.metrics["timeout_requests"] += 1
                agent_instance.failed_requests += 1

            except Exception as e:
                request.status = RequestStatus.FAILED
                request.error = str(e)
                self.metrics["failed_requests"] += 1
                agent_instance.failed_requests += 1
                logger.error(f"Request {request.id} failed: {e}")

            # Update completion time and metrics
            request.completed_at = datetime.now(timezone.utc)
            if request.processing_time:
                # Update average processing time
                total_completed = (
                    self.metrics["completed_requests"] + self.metrics["failed_requests"]
                )
                if total_completed > 0:
                    current_avg = self.metrics["average_processing_time"]
                    self.metrics["average_processing_time"] = (
                        current_avg * (total_completed - 1) + request.processing_time
                    ) / total_completed

        finally:
            # Mark agent as available
            self.busy_agents.discard(agent_id)
            self.available_agents.add(agent_id)
            agent_instance.is_busy = False
            agent_instance.current_request_id = None
            agent_instance.total_requests += 1
            agent_instance.mark_used()

            # Move request to completed
            self.active_requests.pop(request.id, None)
            self.completed_requests.append(request)

            logger.debug(f"Completed request {request.id} with status {request.status}")

    async def _health_monitor(self) -> None:
        """Monitor pool health and update status."""
        logger.debug("Started health monitor")

        while not self._shutdown_event.is_set():
            try:
                # Check resource usage
                if self.resource_manager and hasattr(self.resource_manager, "monitor"):
                    usage = await self.resource_manager.monitor.get_current_usage()
                else:
                    # Fallback usage data
                    import psutil

                    process = psutil.Process()
                    usage = type(
                        "Usage",
                        (),
                        {
                            "memory_percent": process.memory_percent(),
                            "cpu_percent": process.cpu_percent(),
                            "to_dict": lambda: {
                                "memory_percent": process.memory_percent(),
                                "cpu_percent": process.cpu_percent(),
                            },
                        },
                    )()

                # Update pool status based on conditions
                if usage.memory_percent > 90 or usage.cpu_percent > 90:
                    if self.status == PoolStatus.ACTIVE:
                        self.status = PoolStatus.OVERLOADED
                        logger.warning(
                            "Pool status changed to OVERLOADED due to high resource usage"
                        )
                elif usage.memory_percent > 80 or usage.cpu_percent > 80:
                    if self.status == PoolStatus.ACTIVE:
                        self.status = PoolStatus.DEGRADED
                        logger.warning(
                            "Pool status changed to DEGRADED due to elevated resource usage"
                        )
                else:
                    if self.status in [PoolStatus.DEGRADED, PoolStatus.OVERLOADED]:
                        self.status = PoolStatus.ACTIVE
                        logger.info("Pool status restored to ACTIVE")

                await asyncio.sleep(self.config.health_check_interval)

            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(self.config.health_check_interval)

    async def _cleanup_task(self) -> None:
        """Cleanup idle agents and expired requests."""
        logger.debug("Started cleanup task")

        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.now(timezone.utc)

                # Remove idle agents (keep minimum)
                if len(self.agents) > self.config.min_agents:
                    idle_agents = [
                        agent_id
                        for agent_id, agent in self.agents.items()
                        if (
                            agent_id in self.available_agents
                            and (current_time - agent.last_used).total_seconds()
                            > self.config.agent_idle_timeout
                        )
                    ]

                    # Remove excess idle agents
                    agents_to_remove = min(
                        len(idle_agents), len(self.agents) - self.config.min_agents
                    )

                    for agent_id in idle_agents[:agents_to_remove]:
                        await self._remove_agent(agent_id)

                # Cleanup old completed requests (keep last 1000)
                # This is handled by deque maxlen, but we can add additional cleanup here

                await asyncio.sleep(self.config.cleanup_interval)

            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(self.config.cleanup_interval)

    async def _auto_scaler(self) -> None:
        """Auto-scale the agent pool based on load."""
        logger.debug("Started auto-scaler")

        while not self._shutdown_event.is_set():
            try:
                # Calculate queue utilization
                total_queued = sum(len(queue) for queue in self.request_queues.values())
                queue_utilization = total_queued / self.config.max_queue_size

                # Calculate agent utilization
                agent_utilization = (
                    len(self.busy_agents) / len(self.agents) if self.agents else 0
                )

                # Scale up if needed
                if (
                    queue_utilization > self.config.scale_up_threshold
                    or agent_utilization > self.config.scale_up_threshold
                ):
                    if len(self.agents) < self.config.max_agents:
                        await self._create_agent()
                        logger.info(
                            f"Scaled up: created new agent (total: {len(self.agents)})"
                        )

                # Scale down if needed
                elif (
                    queue_utilization < self.config.scale_down_threshold
                    and agent_utilization < self.config.scale_down_threshold
                ):
                    if len(self.agents) > self.config.min_agents:
                        # Find an idle agent to remove
                        for agent_id in list(self.available_agents):
                            agent = self.agents[agent_id]
                            if agent.idle_time > 60:  # Idle for more than 1 minute
                                await self._remove_agent(agent_id)
                                logger.info(
                                    f"Scaled down: removed idle agent (total: {len(self.agents)})"
                                )
                                break

                await asyncio.sleep(30.0)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in auto-scaler: {e}")
                await asyncio.sleep(30.0)

    async def _get_resource_usage_dict(self) -> Dict[str, Any]:
        """Get resource usage as dictionary."""
        if self.resource_manager and hasattr(self.resource_manager, "monitor"):
            usage = await self.resource_manager.monitor.get_current_usage()
            return usage.to_dict()
        else:
            # Fallback implementation
            try:
                import psutil

                process = psutil.Process()
                return {
                    "memory_percent": process.memory_percent(),
                    "cpu_percent": process.cpu_percent(),
                    "connections": 0,
                    "file_handles": 0,
                    "threads": process.num_threads(),
                    "async_tasks": 0,
                }
            except Exception:
                return {
                    "memory_percent": 0.0,
                    "cpu_percent": 0.0,
                    "connections": 0,
                    "file_handles": 0,
                    "threads": 0,
                    "async_tasks": 0,
                }


# Global agent pool instance
_global_pool: Optional[AgentPool] = None


async def get_agent_pool(config: Optional[PoolConfig] = None) -> AgentPool:
    """Get or create the global agent pool instance."""
    global _global_pool

    if _global_pool is None:
        _global_pool = AgentPool(config)
        await _global_pool.start()

    return _global_pool


async def shutdown_agent_pool() -> None:
    """Shutdown the global agent pool."""
    global _global_pool

    if _global_pool is not None:
        await _global_pool.stop()
        _global_pool = None
