from contextlib import AsyncExitStack
from typing import Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import ListToolsResult, TextContent

from app.logger import logger
from app.mcp.connection_manager import (
    ConnectionState,
    MCPConnectionManager,
    ServerConfig,
)
from app.tool.base import BaseTool, ToolResult
from app.tool.tool_collection import ToolCollection


class MCPClientTool(BaseTool):
    """Represents a tool proxy that can be called on the MCP server from the client side."""

    session: Optional[ClientSession] = None
    server_id: str = ""  # Add server identifier
    original_name: str = ""

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool by making a remote call to the MCP server."""
        if not self.session:
            return ToolResult(error="Not connected to MCP server")

        try:
            logger.info(f"Executing tool: {self.original_name}")
            result = await self.session.call_tool(self.original_name, kwargs)
            content_str = ", ".join(
                item.text for item in result.content if isinstance(item, TextContent)
            )
            return ToolResult(output=content_str or "No output returned.")
        except Exception as e:
            return ToolResult(error=f"Error executing tool: {str(e)}")


class MCPClients(ToolCollection):
    """
    A collection of tools that connects to multiple MCP servers and manages available tools through the Model Context Protocol.
    Enhanced with robust connection management, automatic reconnection, and health monitoring.
    """

    sessions: Dict[str, ClientSession] = {}
    exit_stacks: Dict[str, AsyncExitStack] = {}
    description: str = "MCP client tools for server interaction"

    def __init__(self):
        super().__init__()  # Initialize with empty tools list
        self.name = "mcp"  # Keep name for backward compatibility
        self.connection_manager = MCPConnectionManager()

        # Register callbacks for connection events
        self.connection_manager.add_connection_callback(
            self._on_connection_state_changed
        )
        self.connection_manager.add_discovery_callback(self._on_tools_discovered)

    async def connect_sse(self, server_url: str, server_id: str = "") -> None:
        """Connect to an MCP server using SSE transport."""
        if not server_url:
            raise ValueError("Server URL is required.")

        server_id = server_id or server_url

        # Use the new connection manager
        config = ServerConfig(
            server_id=server_id, connection_type="sse", url=server_url, enabled=True
        )

        await self.connection_manager.register_server(config)
        success = await self.connection_manager.connect_server(server_id)

        if not success:
            raise RuntimeError(f"Failed to connect to MCP server {server_id}")

        # Update legacy sessions dict for backward compatibility
        session = self.connection_manager.get_server_session(server_id)
        if session:
            self.sessions[server_id] = session

    async def connect_stdio(
        self, command: str, args: List[str], server_id: str = ""
    ) -> None:
        """Connect to an MCP server using stdio transport."""
        if not command:
            raise ValueError("Server command is required.")

        server_id = server_id or command

        # Use the new connection manager
        config = ServerConfig(
            server_id=server_id,
            connection_type="stdio",
            command=command,
            args=args,
            enabled=True,
        )

        await self.connection_manager.register_server(config)
        success = await self.connection_manager.connect_server(server_id)

        if not success:
            raise RuntimeError(f"Failed to connect to MCP server {server_id}")

        # Update legacy sessions dict for backward compatibility
        session = self.connection_manager.get_server_session(server_id)
        if session:
            self.sessions[server_id] = session

    def _sanitize_tool_name(self, name: str) -> str:
        """Sanitize tool name to match MCPClientTool requirements."""
        import re

        # Replace invalid characters with underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)

        # Remove consecutive underscores
        sanitized = re.sub(r"_+", "_", sanitized)

        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")

        # Truncate to 64 characters if needed
        if len(sanitized) > 64:
            sanitized = sanitized[:64]

        return sanitized

    async def list_tools(self) -> ListToolsResult:
        """List all available tools."""
        all_tools_results = await self.connection_manager.list_all_tools()
        tools_result = ListToolsResult(tools=[])

        for server_id, result in all_tools_results.items():
            tools_result.tools.extend(result.tools)

        return tools_result

    def _on_connection_state_changed(
        self, server_id: str, new_state: ConnectionState
    ) -> None:
        """Handle connection state changes."""
        logger.info(
            f"MCP server {server_id} connection state changed to {new_state.value}"
        )

        if new_state == ConnectionState.CONNECTED:
            # Update legacy sessions dict
            session = self.connection_manager.get_server_session(server_id)
            if session:
                self.sessions[server_id] = session
        elif new_state in [ConnectionState.DISCONNECTED, ConnectionState.FAILED]:
            # Clean up legacy references
            self.sessions.pop(server_id, None)
            self.exit_stacks.pop(server_id, None)

            # Remove tools from this server
            self.tool_map = {
                k: v
                for k, v in self.tool_map.items()
                if getattr(v, "server_id", None) != server_id
            }
            self.tools = tuple(self.tool_map.values())

    def _on_tools_discovered(self, server_id: str, tool_names: List[str]) -> None:
        """Handle tool discovery events."""
        logger.info(f"Discovered tools from server {server_id}: {tool_names}")

        # Get the session for this server
        session = self.connection_manager.get_server_session(server_id)
        if not session:
            logger.warning(f"No session available for server {server_id}")
            return

        # Create tool objects for discovered tools
        try:
            # Get tool details from the server
            import asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, schedule the tool creation
                asyncio.create_task(self._create_tools_for_server(server_id, session))
            else:
                # If not in async context, run synchronously
                loop.run_until_complete(
                    self._create_tools_for_server(server_id, session)
                )
        except Exception as e:
            logger.error(f"Error creating tools for server {server_id}: {e}")

    async def _create_tools_for_server(
        self, server_id: str, session: ClientSession
    ) -> None:
        """Create tool objects for a server's tools."""
        try:
            response = await session.list_tools()

            for tool in response.tools:
                original_name = tool.name
                tool_name = f"mcp_{server_id}_{original_name}"
                tool_name = self._sanitize_tool_name(tool_name)

                server_tool = MCPClientTool(
                    name=tool_name,
                    description=tool.description,
                    parameters=tool.inputSchema,
                    session=session,
                    server_id=server_id,
                    original_name=original_name,
                )
                self.tool_map[tool_name] = server_tool

            # Update tools tuple
            self.tools = tuple(self.tool_map.values())
            logger.info(f"Created {len(response.tools)} tools for server {server_id}")

        except Exception as e:
            logger.error(f"Failed to create tools for server {server_id}: {e}")

    async def reconnect_server(self, server_id: str) -> bool:
        """Reconnect to a specific MCP server."""
        return await self.connection_manager.reconnect_server(server_id)

    def get_connected_servers(self) -> List[str]:
        """Get list of currently connected server IDs."""
        return self.connection_manager.get_connected_servers()

    def get_server_health(self, server_id: str) -> Optional[Dict]:
        """Get health information for a server."""
        connection_info = self.connection_manager.get_connection_info(server_id)
        if not connection_info:
            return None

        return {
            "server_id": server_id,
            "state": connection_info.state.value,
            "last_connected": connection_info.last_connected,
            "last_error": connection_info.last_error,
            "retry_count": connection_info.retry_count,
            "config": {
                "connection_type": connection_info.config.connection_type,
                "enabled": connection_info.config.enabled,
                "max_retries": connection_info.config.max_retries,
            },
        }

    async def disconnect(self, server_id: str = "") -> None:
        """Disconnect from a specific MCP server or all servers if no server_id provided."""
        if server_id:
            await self.connection_manager.disconnect_server(server_id)
            # Clean up legacy references
            self.sessions.pop(server_id, None)
            self.exit_stacks.pop(server_id, None)

            # Remove tools associated with this server
            self.tool_map = {
                k: v
                for k, v in self.tool_map.items()
                if getattr(v, "server_id", None) != server_id
            }
            self.tools = tuple(self.tool_map.values())
        else:
            # Disconnect from all servers
            await self.connection_manager.shutdown()
            self.sessions.clear()
            self.exit_stacks.clear()
            self.tool_map = {}
            self.tools = tuple()
            logger.info("Disconnected from all MCP servers")
