"""
Enhanced MCP Connection Manager with robust connection handling and automatic reconnection.
"""

import asyncio
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import urlparse

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import ListToolsResult

from app.logger import logger
from app.resilience import CircuitBreaker, RetryManager


class ConnectionState(Enum):
    """Connection states for MCP servers."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass
class ServerConfig:
    """Configuration for an MCP server."""

    server_id: str
    connection_type: str  # "stdio" or "sse"
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    enabled: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    health_check_interval: int = 30
    connection_timeout: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectionInfo:
    """Information about a server connection."""

    config: ServerConfig
    state: ConnectionState = ConnectionState.DISCONNECTED
    session: Optional[ClientSession] = None
    exit_stack: Optional[AsyncExitStack] = None
    last_connected: Optional[float] = None
    last_error: Optional[str] = None
    retry_count: int = 0
    circuit_breaker: Optional[CircuitBreaker] = None
    health_check_task: Optional[asyncio.Task] = None


class MCPConnectionManager:
    """
    Enhanced MCP connection manager with robust connection handling,
    automatic reconnection, health monitoring, and load balancing.
    """

    def __init__(self):
        self.connections: Dict[str, ConnectionInfo] = {}
        self.retry_manager = RetryManager()
        self._shutdown_event = asyncio.Event()
        self._connection_callbacks: List[Callable[[str, ConnectionState], None]] = []
        self._discovery_callbacks: List[Callable[[str, List[str]], None]] = []

    def add_connection_callback(
        self, callback: Callable[[str, ConnectionState], None]
    ) -> None:
        """Add a callback for connection state changes."""
        self._connection_callbacks.append(callback)

    def add_discovery_callback(
        self, callback: Callable[[str, List[str]], None]
    ) -> None:
        """Add a callback for tool discovery events."""
        self._discovery_callbacks.append(callback)

    async def register_server(self, config: ServerConfig) -> None:
        """Register a new MCP server configuration."""
        if config.server_id in self.connections:
            logger.warning(
                f"Server {config.server_id} already registered, updating configuration"
            )
            await self.disconnect_server(config.server_id)

        # Create circuit breaker for this server
        circuit_breaker = CircuitBreaker(
            failure_threshold=3, recovery_timeout=60, expected_exception=Exception
        )

        connection_info = ConnectionInfo(config=config, circuit_breaker=circuit_breaker)

        self.connections[config.server_id] = connection_info
        logger.info(f"Registered MCP server: {config.server_id}")

        # Auto-connect if enabled
        if config.enabled:
            await self.connect_server(config.server_id)

    async def connect_server(self, server_id: str) -> bool:
        """Connect to a specific MCP server with retry logic."""
        if server_id not in self.connections:
            logger.error(f"Server {server_id} not registered")
            return False

        connection_info = self.connections[server_id]
        config = connection_info.config

        if not config.enabled:
            logger.info(f"Server {server_id} is disabled, skipping connection")
            return False

        if connection_info.state in [
            ConnectionState.CONNECTED,
            ConnectionState.CONNECTING,
        ]:
            logger.info(f"Server {server_id} already connected or connecting")
            return connection_info.state == ConnectionState.CONNECTED

        await self._update_connection_state(server_id, ConnectionState.CONNECTING)

        try:
            # Use circuit breaker to prevent repeated failures
            success = await connection_info.circuit_breaker.call(
                self._establish_connection, server_id
            )

            if success:
                await self._update_connection_state(
                    server_id, ConnectionState.CONNECTED
                )
                connection_info.last_connected = time.time()
                connection_info.retry_count = 0

                # Start health check monitoring
                await self._start_health_monitoring(server_id)

                # Discover tools
                await self._discover_tools(server_id)

                logger.info(f"Successfully connected to MCP server: {server_id}")
                return True
            else:
                await self._update_connection_state(server_id, ConnectionState.FAILED)
                return False

        except Exception as e:
            logger.error(f"Failed to connect to server {server_id}: {e}")
            connection_info.last_error = str(e)
            await self._update_connection_state(server_id, ConnectionState.FAILED)
            return False

    async def _establish_connection(self, server_id: str) -> bool:
        """Establish the actual connection to the MCP server."""
        connection_info = self.connections[server_id]
        config = connection_info.config

        # Clean up any existing connection
        await self._cleanup_connection(server_id)

        exit_stack = AsyncExitStack()
        connection_info.exit_stack = exit_stack

        try:
            if config.connection_type == "sse":
                if not config.url:
                    raise ValueError("URL is required for SSE connection")

                # Validate URL format
                parsed_url = urlparse(config.url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    raise ValueError(f"Invalid URL format: {config.url}")

                streams_context = sse_client(url=config.url)
                streams = await asyncio.wait_for(
                    exit_stack.enter_async_context(streams_context),
                    timeout=config.connection_timeout,
                )
                session = await exit_stack.enter_async_context(ClientSession(*streams))

            elif config.connection_type == "stdio":
                if not config.command:
                    raise ValueError("Command is required for stdio connection")

                server_params = StdioServerParameters(
                    command=config.command, args=config.args or []
                )
                stdio_transport = await asyncio.wait_for(
                    exit_stack.enter_async_context(stdio_client(server_params)),
                    timeout=config.connection_timeout,
                )
                read, write = stdio_transport
                session = await exit_stack.enter_async_context(
                    ClientSession(read, write)
                )

            else:
                raise ValueError(
                    f"Unsupported connection type: {config.connection_type}"
                )

            # Initialize the session
            await asyncio.wait_for(
                session.initialize(), timeout=config.connection_timeout
            )
            connection_info.session = session

            return True

        except Exception as e:
            # Clean up on failure
            try:
                await exit_stack.aclose()
            except Exception as cleanup_error:
                logger.warning(f"Error during connection cleanup: {cleanup_error}")

            connection_info.exit_stack = None
            raise e

    async def disconnect_server(self, server_id: str) -> None:
        """Disconnect from a specific MCP server."""
        if server_id not in self.connections:
            logger.warning(f"Server {server_id} not found")
            return

        connection_info = self.connections[server_id]

        # Stop health monitoring
        if connection_info.health_check_task:
            connection_info.health_check_task.cancel()
            try:
                await connection_info.health_check_task
            except asyncio.CancelledError:
                pass
            connection_info.health_check_task = None

        # Clean up connection
        await self._cleanup_connection(server_id)
        await self._update_connection_state(server_id, ConnectionState.DISCONNECTED)

        logger.info(f"Disconnected from MCP server: {server_id}")

    async def _cleanup_connection(self, server_id: str) -> None:
        """Clean up connection resources for a server."""
        connection_info = self.connections[server_id]

        if connection_info.exit_stack:
            try:
                await connection_info.exit_stack.aclose()
            except Exception as e:
                logger.warning(f"Error closing exit stack for {server_id}: {e}")
            finally:
                connection_info.exit_stack = None
                connection_info.session = None

    async def reconnect_server(self, server_id: str) -> bool:
        """Reconnect to a specific MCP server."""
        if server_id not in self.connections:
            logger.error(f"Server {server_id} not registered")
            return False

        connection_info = self.connections[server_id]

        if connection_info.retry_count >= connection_info.config.max_retries:
            logger.error(f"Max retries exceeded for server {server_id}")
            await self._update_connection_state(server_id, ConnectionState.FAILED)
            return False

        await self._update_connection_state(server_id, ConnectionState.RECONNECTING)
        connection_info.retry_count += 1

        # Exponential backoff
        delay = connection_info.config.retry_delay * (
            2 ** (connection_info.retry_count - 1)
        )
        await asyncio.sleep(delay)

        return await self.connect_server(server_id)

    async def _start_health_monitoring(self, server_id: str) -> None:
        """Start health check monitoring for a server."""
        connection_info = self.connections[server_id]

        if connection_info.health_check_task:
            connection_info.health_check_task.cancel()

        connection_info.health_check_task = asyncio.create_task(
            self._health_check_loop(server_id)
        )

    async def _health_check_loop(self, server_id: str) -> None:
        """Health check loop for a server connection."""
        connection_info = self.connections[server_id]
        config = connection_info.config

        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(config.health_check_interval)

                if connection_info.state != ConnectionState.CONNECTED:
                    break

                # Perform health check by listing tools
                if connection_info.session:
                    await asyncio.wait_for(
                        connection_info.session.list_tools(), timeout=5.0
                    )
                    logger.debug(f"Health check passed for server {server_id}")
                else:
                    raise Exception("Session is None")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Health check failed for server {server_id}: {e}")
                connection_info.last_error = str(e)

                # Attempt reconnection
                if not self._shutdown_event.is_set():
                    asyncio.create_task(self.reconnect_server(server_id))
                break

    async def _discover_tools(self, server_id: str) -> None:
        """Discover available tools from a server."""
        connection_info = self.connections[server_id]

        if not connection_info.session:
            return

        try:
            response = await connection_info.session.list_tools()
            tool_names = [tool.name for tool in response.tools]

            # Notify discovery callbacks
            for callback in self._discovery_callbacks:
                try:
                    callback(server_id, tool_names)
                except Exception as e:
                    logger.error(f"Error in discovery callback: {e}")

            logger.info(
                f"Discovered {len(tool_names)} tools from server {server_id}: {tool_names}"
            )

        except Exception as e:
            logger.error(f"Failed to discover tools from server {server_id}: {e}")

    async def _update_connection_state(
        self, server_id: str, new_state: ConnectionState
    ) -> None:
        """Update connection state and notify callbacks."""
        if server_id not in self.connections:
            return

        old_state = self.connections[server_id].state
        self.connections[server_id].state = new_state

        if old_state != new_state:
            logger.info(
                f"Server {server_id} state changed: {old_state.value} -> {new_state.value}"
            )

            # Notify connection callbacks
            for callback in self._connection_callbacks:
                try:
                    callback(server_id, new_state)
                except Exception as e:
                    logger.error(f"Error in connection callback: {e}")

    def get_connected_servers(self) -> List[str]:
        """Get list of currently connected server IDs."""
        return [
            server_id
            for server_id, info in self.connections.items()
            if info.state == ConnectionState.CONNECTED
        ]

    def get_server_session(self, server_id: str) -> Optional[ClientSession]:
        """Get the session for a specific server."""
        if server_id in self.connections:
            return self.connections[server_id].session
        return None

    def get_connection_info(self, server_id: str) -> Optional[ConnectionInfo]:
        """Get connection information for a server."""
        return self.connections.get(server_id)

    async def list_all_tools(self) -> Dict[str, ListToolsResult]:
        """List tools from all connected servers."""
        results = {}

        for server_id, connection_info in self.connections.items():
            if (
                connection_info.state == ConnectionState.CONNECTED
                and connection_info.session
            ):
                try:
                    response = await connection_info.session.list_tools()
                    results[server_id] = response
                except Exception as e:
                    logger.error(f"Failed to list tools from server {server_id}: {e}")

        return results

    async def shutdown(self) -> None:
        """Shutdown all connections and cleanup resources."""
        logger.info("Shutting down MCP connection manager")
        self._shutdown_event.set()

        # Disconnect all servers
        disconnect_tasks = []
        for server_id in list(self.connections.keys()):
            disconnect_tasks.append(self.disconnect_server(server_id))

        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)

        self.connections.clear()
        logger.info("MCP connection manager shutdown complete")

    def get_load_balanced_server(
        self, servers: Optional[List[str]] = None
    ) -> Optional[str]:
        """Get a server for load balancing based on connection health and load."""
        available_servers = servers or self.get_connected_servers()

        if not available_servers:
            return None

        # Simple round-robin for now - can be enhanced with more sophisticated algorithms
        # Filter to only connected servers
        connected_servers = [
            server_id
            for server_id in available_servers
            if self.connections.get(server_id, {}).state == ConnectionState.CONNECTED
        ]

        if not connected_servers:
            return None

        # For now, return the first connected server
        # TODO: Implement more sophisticated load balancing
        return connected_servers[0]
