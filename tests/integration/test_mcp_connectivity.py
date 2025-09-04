"""
Integration tests for MCP server connectivity and tool registration.

This module contains comprehensive tests for MCP (Model Context Protocol)
server connections, tool discovery, registration, and lifecycle management.
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from app.agent.manus import Manus
from app.config import Config
from app.exceptions import MCPConnectionError, ToolRegistrationError
from app.tool.mcp import MCPClients, MCPClientTool
from tests.base import IntegrationTestCase, MCPTestCase


class TestMCPServerConnectivity(MCPTestCase):
    """Tests for MCP server connectivity."""

    def setup_method(self):
        """Set up MCP connectivity test environment."""
        super().setup_method()
        self.mock_config = self.create_mock("app.config.config")
        self.mock_config.mcp_config.servers = {}

    @pytest.mark.asyncio
    async def test_sse_server_connection_success(self):
        """Test successful SSE server connection."""
        mcp_clients = MCPClients()

        # Mock SSE connection components
        with patch("app.tool.mcp.sse_client") as mock_sse_client:
            with patch("app.tool.mcp.ClientSession") as mock_session_class:

                # Setup mocks
                mock_streams = Mock()
                mock_sse_client.return_value.__aenter__ = AsyncMock(
                    return_value=mock_streams
                )
                mock_sse_client.return_value.__aexit__ = AsyncMock(return_value=None)

                mock_session = Mock()
                mock_session.initialize = AsyncMock()
                mock_session.list_tools = AsyncMock(return_value=Mock(tools=[]))
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                # Test connection
                await mcp_clients.connect_sse("http://test-server.com", "test_server")

                # Verify connection established
                assert "test_server" in mcp_clients.sessions
                mock_session.initialize.assert_called_once()
                mock_session.list_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_stdio_server_connection_success(self):
        """Test successful stdio server connection."""
        mcp_clients = MCPClients()

        with patch("app.tool.mcp.stdio_client") as mock_stdio_client:
            with patch("app.tool.mcp.ClientSession") as mock_session_class:

                # Setup mocks
                mock_transport = (Mock(), Mock())  # read, write streams
                mock_stdio_client.return_value.__aenter__ = AsyncMock(
                    return_value=mock_transport
                )
                mock_stdio_client.return_value.__aexit__ = AsyncMock(return_value=None)

                mock_session = Mock()
                mock_session.initialize = AsyncMock()
                mock_session.list_tools = AsyncMock(return_value=Mock(tools=[]))
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                # Test connection
                await mcp_clients.connect_stdio(
                    "python", ["-m", "test_server"], "stdio_server"
                )

                # Verify connection established
                assert "stdio_server" in mcp_clients.sessions
                mock_session.initialize.assert_called_once()
                mock_session.list_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_failure_handling(self):
        """Test connection failure handling."""
        mcp_clients = MCPClients()

        with patch("app.tool.mcp.sse_client") as mock_sse_client:
            # Simulate connection failure
            mock_sse_client.side_effect = ConnectionError("Connection refused")

            with pytest.raises(ConnectionError):
                await mcp_clients.connect_sse(
                    "http://invalid-server.com", "invalid_server"
                )

            # Verify no session was created
            assert "invalid_server" not in mcp_clients.sessions

    @pytest.mark.asyncio
    async def test_connection_cleanup_on_failure(self):
        """Test proper cleanup when connection fails during initialization."""
        mcp_clients = MCPClients()

        with patch("app.tool.mcp.sse_client") as mock_sse_client:
            with patch("app.tool.mcp.ClientSession") as mock_session_class:

                # Setup partial success then failure
                mock_streams = Mock()
                mock_sse_client.return_value.__aenter__ = AsyncMock(
                    return_value=mock_streams
                )
                mock_sse_client.return_value.__aexit__ = AsyncMock(return_value=None)

                mock_session = Mock()
                mock_session.initialize = AsyncMock(
                    side_effect=Exception("Initialization failed")
                )
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                with pytest.raises(Exception, match="Initialization failed"):
                    await mcp_clients.connect_sse(
                        "http://test-server.com", "test_server"
                    )

                # Verify cleanup occurred
                assert "test_server" not in mcp_clients.sessions

    @pytest.mark.asyncio
    async def test_multiple_server_connections(self):
        """Test connecting to multiple MCP servers."""
        mcp_clients = MCPClients()

        servers = [
            ("http://server1.com", "server1"),
            ("http://server2.com", "server2"),
            ("http://server3.com", "server3"),
        ]

        with patch("app.tool.mcp.sse_client") as mock_sse_client:
            with patch("app.tool.mcp.ClientSession") as mock_session_class:

                # Setup mocks for successful connections
                mock_streams = Mock()
                mock_sse_client.return_value.__aenter__ = AsyncMock(
                    return_value=mock_streams
                )
                mock_sse_client.return_value.__aexit__ = AsyncMock(return_value=None)

                mock_session = Mock()
                mock_session.initialize = AsyncMock()
                mock_session.list_tools = AsyncMock(return_value=Mock(tools=[]))
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                # Connect to all servers
                for url, server_id in servers:
                    await mcp_clients.connect_sse(url, server_id)

                # Verify all connections
                assert len(mcp_clients.sessions) == 3
                for _, server_id in servers:
                    assert server_id in mcp_clients.sessions

    @pytest.mark.asyncio
    async def test_server_disconnection(self):
        """Test server disconnection."""
        mcp_clients = MCPClients()

        # Setup initial connection
        await self.setup_mock_mcp_server("test_server", ["tool1", "tool2"])

        # Simulate connection in mcp_clients
        mock_exit_stack = Mock()
        mock_exit_stack.aclose = AsyncMock()
        mcp_clients.exit_stacks["test_server"] = mock_exit_stack
        mcp_clients.sessions["test_server"] = Mock()

        # Add some tools
        tool1 = MCPClientTool(
            name="mcp_test_server_tool1",
            description="Test tool 1",
            server_id="test_server",
            original_name="tool1",
        )
        mcp_clients.tool_map["mcp_test_server_tool1"] = tool1
        mcp_clients.tools = (tool1,)

        # Test disconnection
        await mcp_clients.disconnect("test_server")

        # Verify cleanup
        assert "test_server" not in mcp_clients.sessions
        assert "test_server" not in mcp_clients.exit_stacks
        assert len(mcp_clients.tool_map) == 0
        mock_exit_stack.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_all_servers(self):
        """Test disconnecting from all servers."""
        mcp_clients = MCPClients()

        # Setup multiple connections
        server_ids = ["server1", "server2", "server3"]
        for server_id in server_ids:
            mock_exit_stack = Mock()
            mock_exit_stack.aclose = AsyncMock()
            mcp_clients.exit_stacks[server_id] = mock_exit_stack
            mcp_clients.sessions[server_id] = Mock()

        # Test disconnect all
        await mcp_clients.disconnect()

        # Verify all cleaned up
        assert len(mcp_clients.sessions) == 0
        assert len(mcp_clients.exit_stacks) == 0
        assert len(mcp_clients.tool_map) == 0

    @pytest.mark.asyncio
    async def test_reconnection_after_disconnect(self):
        """Test reconnection after disconnection."""
        mcp_clients = MCPClients()

        with patch("app.tool.mcp.sse_client") as mock_sse_client:
            with patch("app.tool.mcp.ClientSession") as mock_session_class:

                # Setup mocks
                mock_streams = Mock()
                mock_sse_client.return_value.__aenter__ = AsyncMock(
                    return_value=mock_streams
                )
                mock_sse_client.return_value.__aexit__ = AsyncMock(return_value=None)

                mock_session = Mock()
                mock_session.initialize = AsyncMock()
                mock_session.list_tools = AsyncMock(return_value=Mock(tools=[]))
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                # Initial connection
                await mcp_clients.connect_sse("http://test-server.com", "test_server")
                assert "test_server" in mcp_clients.sessions

                # Disconnect
                await mcp_clients.disconnect("test_server")
                assert "test_server" not in mcp_clients.sessions

                # Reconnect
                await mcp_clients.connect_sse("http://test-server.com", "test_server")
                assert "test_server" in mcp_clients.sessions


class TestMCPToolRegistration(MCPTestCase):
    """Tests for MCP tool registration and discovery."""

    def setup_method(self):
        """Set up tool registration test environment."""
        super().setup_method()
        self.mcp_clients = MCPClients()

    @pytest.mark.asyncio
    async def test_tool_discovery_and_registration(self):
        """Test tool discovery and registration process."""
        # Mock tool information
        mock_tools = [
            Mock(
                name="search_web",
                description="Search the web",
                inputSchema={"type": "object"},
            ),
            Mock(
                name="read_file",
                description="Read a file",
                inputSchema={"type": "object"},
            ),
            Mock(
                name="write_file",
                description="Write to a file",
                inputSchema={"type": "object"},
            ),
        ]

        with patch("app.tool.mcp.sse_client") as mock_sse_client:
            with patch("app.tool.mcp.ClientSession") as mock_session_class:

                # Setup mocks
                mock_streams = Mock()
                mock_sse_client.return_value.__aenter__ = AsyncMock(
                    return_value=mock_streams
                )
                mock_sse_client.return_value.__aexit__ = AsyncMock(return_value=None)

                mock_session = Mock()
                mock_session.initialize = AsyncMock()
                mock_session.list_tools = AsyncMock(return_value=Mock(tools=mock_tools))
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                # Connect and register tools
                await self.mcp_clients.connect_sse(
                    "http://test-server.com", "test_server"
                )

                # Verify tools were registered
                assert len(self.mcp_clients.tool_map) == 3

                # Check tool names are properly formatted
                expected_names = [
                    "mcp_test_server_search_web",
                    "mcp_test_server_read_file",
                    "mcp_test_server_write_file",
                ]

                for expected_name in expected_names:
                    assert expected_name in self.mcp_clients.tool_map
                    tool = self.mcp_clients.tool_map[expected_name]
                    assert isinstance(tool, MCPClientTool)
                    assert tool.server_id == "test_server"

    @pytest.mark.asyncio
    async def test_tool_name_sanitization(self):
        """Test tool name sanitization for invalid characters."""
        # Mock tools with problematic names
        mock_tools = [
            Mock(
                name="tool-with-dashes", description="Tool with dashes", inputSchema={}
            ),
            Mock(name="tool.with.dots", description="Tool with dots", inputSchema={}),
            Mock(
                name="tool with spaces", description="Tool with spaces", inputSchema={}
            ),
            Mock(
                name="tool@with#special$chars",
                description="Tool with special chars",
                inputSchema={},
            ),
        ]

        with patch("app.tool.mcp.sse_client") as mock_sse_client:
            with patch("app.tool.mcp.ClientSession") as mock_session_class:

                # Setup mocks
                mock_streams = Mock()
                mock_sse_client.return_value.__aenter__ = AsyncMock(
                    return_value=mock_streams
                )
                mock_sse_client.return_value.__aexit__ = AsyncMock(return_value=None)

                mock_session = Mock()
                mock_session.initialize = AsyncMock()
                mock_session.list_tools = AsyncMock(return_value=Mock(tools=mock_tools))
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                # Connect and register tools
                await self.mcp_clients.connect_sse(
                    "http://test-server.com", "test_server"
                )

                # Verify tools were registered with sanitized names
                registered_names = list(self.mcp_clients.tool_map.keys())

                # All names should be valid identifiers
                for name in registered_names:
                    # Should not contain problematic characters
                    assert "@" not in name
                    assert "#" not in name
                    assert "$" not in name
                    assert " " not in name
                    assert "." not in name

    @pytest.mark.asyncio
    async def test_duplicate_tool_handling(self):
        """Test handling of duplicate tool names from different servers."""
        # Mock tools with same names from different servers
        mock_tools = [
            Mock(
                name="common_tool",
                description="Common tool from server 1",
                inputSchema={},
            )
        ]

        with patch("app.tool.mcp.sse_client") as mock_sse_client:
            with patch("app.tool.mcp.ClientSession") as mock_session_class:

                # Setup mocks
                mock_streams = Mock()
                mock_sse_client.return_value.__aenter__ = AsyncMock(
                    return_value=mock_streams
                )
                mock_sse_client.return_value.__aexit__ = AsyncMock(return_value=None)

                mock_session = Mock()
                mock_session.initialize = AsyncMock()
                mock_session.list_tools = AsyncMock(return_value=Mock(tools=mock_tools))
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                # Connect first server
                await self.mcp_clients.connect_sse("http://server1.com", "server1")

                # Connect second server with same tool name
                await self.mcp_clients.connect_sse("http://server2.com", "server2")

                # Verify both tools are registered with different prefixed names
                assert "mcp_server1_common_tool" in self.mcp_clients.tool_map
                assert "mcp_server2_common_tool" in self.mcp_clients.tool_map

                # Verify they have different server IDs
                tool1 = self.mcp_clients.tool_map["mcp_server1_common_tool"]
                tool2 = self.mcp_clients.tool_map["mcp_server2_common_tool"]
                assert tool1.server_id == "server1"
                assert tool2.server_id == "server2"

    @pytest.mark.asyncio
    async def test_tool_registration_failure_recovery(self):
        """Test recovery from tool registration failures."""
        # Mock tools where some fail to register
        mock_tools = [
            Mock(
                name="good_tool",
                description="Good tool",
                inputSchema={"type": "object"},
            ),
            Mock(
                name="", description="Tool with empty name", inputSchema={}
            ),  # Invalid
            Mock(
                name="another_good_tool",
                description="Another good tool",
                inputSchema={"type": "object"},
            ),
        ]

        with patch("app.tool.mcp.sse_client") as mock_sse_client:
            with patch("app.tool.mcp.ClientSession") as mock_session_class:

                # Setup mocks
                mock_streams = Mock()
                mock_sse_client.return_value.__aenter__ = AsyncMock(
                    return_value=mock_streams
                )
                mock_sse_client.return_value.__aexit__ = AsyncMock(return_value=None)

                mock_session = Mock()
                mock_session.initialize = AsyncMock()
                mock_session.list_tools = AsyncMock(return_value=Mock(tools=mock_tools))
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                # Connect - should handle invalid tool gracefully
                await self.mcp_clients.connect_sse(
                    "http://test-server.com", "test_server"
                )

                # Verify valid tools were registered despite invalid one
                valid_tools = [
                    name
                    for name in self.mcp_clients.tool_map.keys()
                    if "good_tool" in name
                ]
                assert len(valid_tools) >= 2  # Should have registered the valid tools

    @pytest.mark.asyncio
    async def test_tool_metadata_preservation(self):
        """Test that tool metadata is properly preserved during registration."""
        # Mock tool with rich metadata
        mock_tool = Mock(
            name="complex_tool",
            description="A complex tool with rich metadata",
            inputSchema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "First parameter"},
                    "param2": {"type": "integer", "minimum": 0, "maximum": 100},
                },
                "required": ["param1"],
            },
        )

        with patch("app.tool.mcp.sse_client") as mock_sse_client:
            with patch("app.tool.mcp.ClientSession") as mock_session_class:

                # Setup mocks
                mock_streams = Mock()
                mock_sse_client.return_value.__aenter__ = AsyncMock(
                    return_value=mock_streams
                )
                mock_sse_client.return_value.__aexit__ = AsyncMock(return_value=None)

                mock_session = Mock()
                mock_session.initialize = AsyncMock()
                mock_session.list_tools = AsyncMock(
                    return_value=Mock(tools=[mock_tool])
                )
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                # Connect and register tool
                await self.mcp_clients.connect_sse(
                    "http://test-server.com", "test_server"
                )

                # Verify metadata preservation
                registered_tool = self.mcp_clients.tool_map[
                    "mcp_test_server_complex_tool"
                ]
                assert (
                    registered_tool.description == "A complex tool with rich metadata"
                )
                assert registered_tool.parameters == mock_tool.inputSchema
                assert registered_tool.original_name == "complex_tool"
                assert registered_tool.server_id == "test_server"


class TestMCPAgentIntegration(IntegrationTestCase):
    """Integration tests for MCP functionality within Manus agent."""

    def setup_method(self):
        """Set up MCP agent integration test environment."""
        super().setup_method()
        self.mock_config = self.create_mock("app.config.config")
        self.mock_config.workspace_root = str(self.workspace)

    @pytest.mark.asyncio
    async def test_agent_mcp_initialization_from_config(self):
        """Test agent MCP initialization from configuration."""
        # Setup mock configuration
        mock_server_config = Mock()
        mock_server_config.type = "sse"
        mock_server_config.url = "http://configured-server.com"

        self.mock_config.mcp_config.servers = {"configured_server": mock_server_config}

        agent = Manus()

        with patch.object(
            agent, "connect_mcp_server", new_callable=AsyncMock
        ) as mock_connect:
            await agent.initialize_mcp_servers()

            mock_connect.assert_called_once_with(
                "http://configured-server.com", "configured_server"
            )

    @pytest.mark.asyncio
    async def test_agent_tool_availability_after_mcp_connection(self):
        """Test that MCP tools become available in agent after connection."""
        agent = Manus()

        # Mock MCP tool registration
        mock_mcp_tool = MCPClientTool(
            name="mcp_test_server_available_tool",
            description="Available MCP tool",
            server_id="test_server",
            original_name="available_tool",
        )

        with patch.object(agent.mcp_clients, "connect_sse", new_callable=AsyncMock):
            # Simulate tool registration
            agent.mcp_clients.tool_map["mcp_test_server_available_tool"] = mock_mcp_tool
            agent.mcp_clients.tools = (mock_mcp_tool,)

            await agent.connect_mcp_server("http://test.com", "test_server")

            # Verify tool is available in agent
            available_tool_names = [tool.name for tool in agent.available_tools.tools]
            assert "mcp_test_server_available_tool" in available_tool_names

    @pytest.mark.asyncio
    async def test_agent_cleanup_disconnects_mcp_servers(self):
        """Test that agent cleanup properly disconnects MCP servers."""
        agent = Manus()
        agent._initialized = True
        agent.connected_servers["test_server"] = "http://test.com"

        with patch.object(
            agent, "disconnect_mcp_server", new_callable=AsyncMock
        ) as mock_disconnect:
            await agent.cleanup()

            mock_disconnect.assert_called_once()
            assert not agent._initialized

    @pytest.mark.asyncio
    async def test_agent_mcp_error_isolation(self):
        """Test that MCP errors don't crash the entire agent."""
        agent = Manus()

        with patch.object(
            agent.mcp_clients, "connect_sse", new_callable=AsyncMock
        ) as mock_connect:
            mock_connect.side_effect = Exception("MCP connection failed")

            # Should not raise exception, just log error
            await agent.initialize_mcp_servers()

            # Agent should still be functional
            assert agent.state == AgentState.IDLE
            assert len(agent.connected_servers) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
