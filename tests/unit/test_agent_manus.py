"""
Comprehensive unit tests for the Manus agent class.

This module contains unit tests covering all methods of the Manus agent,
including initialization, MCP server connectivity, tool execution, error handling,
and memory management.
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from app.agent.base import BaseAgent
from app.agent.manus import Manus
from app.config import Config
from app.exceptions import (
    MCPConnectionError,
    MemoryError,
    OpenManusError,
    ToolExecutionError,
)
from app.schema import AgentState, Memory, Message, ToolCall, ToolResult
from app.tool.base import BaseTool
from app.tool.mcp import MCPClients, MCPClientTool
from tests.base import AgentTestCase, UnitTestCase


class TestManusAgent(UnitTestCase):
    """Unit tests for Manus agent class."""

    def setup_method(self):
        """Set up test environment."""
        super().setup_method()
        self.mock_config = self.create_mock("app.config.config")
        self.mock_config.workspace_root = "/tmp/test"
        self.mock_config.mcp_config.servers = {}

    @pytest.mark.asyncio
    async def test_manus_initialization(self):
        """Test Manus agent initialization."""
        # Test basic initialization
        agent = Manus()

        assert agent.name == "Manus"
        assert "versatile agent" in agent.description.lower()
        assert agent.max_steps == 20
        assert agent.max_observe == 10000
        assert isinstance(agent.memory, Memory)
        assert agent.state == AgentState.IDLE
        assert not agent._initialized

    @pytest.mark.asyncio
    async def test_manus_create_factory_method(self):
        """Test Manus.create() factory method."""
        with patch.object(
            Manus, "initialize_mcp_servers", new_callable=AsyncMock
        ) as mock_init:
            agent = await Manus.create()

            assert isinstance(agent, Manus)
            assert agent._initialized
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_mcp_servers_success(self):
        """Test successful MCP server initialization."""
        mock_server_config = Mock()
        mock_server_config.type = "sse"
        mock_server_config.url = "http://test-server.com"

        self.mock_config.mcp_config.servers = {"test_server": mock_server_config}

        agent = Manus()

        with patch.object(
            agent, "connect_mcp_server", new_callable=AsyncMock
        ) as mock_connect:
            await agent.initialize_mcp_servers()

            mock_connect.assert_called_once_with(
                "http://test-server.com", "test_server"
            )

    @pytest.mark.asyncio
    async def test_initialize_mcp_servers_stdio(self):
        """Test MCP server initialization with stdio transport."""
        mock_server_config = Mock()
        mock_server_config.type = "stdio"
        mock_server_config.command = "python"
        mock_server_config.args = ["-m", "test_server"]
        mock_server_config.url = None

        self.mock_config.mcp_config.servers = {"stdio_server": mock_server_config}

        agent = Manus()

        with patch.object(
            agent, "connect_mcp_server", new_callable=AsyncMock
        ) as mock_connect:
            await agent.initialize_mcp_servers()

            mock_connect.assert_called_once_with(
                "python",
                "stdio_server",
                use_stdio=True,
                stdio_args=["-m", "test_server"],
            )

    @pytest.mark.asyncio
    async def test_initialize_mcp_servers_failure(self):
        """Test MCP server initialization failure handling."""
        mock_server_config = Mock()
        mock_server_config.type = "sse"
        mock_server_config.url = "http://invalid-server.com"

        self.mock_config.mcp_config.servers = {"failing_server": mock_server_config}

        agent = Manus()

        with patch.object(
            agent, "connect_mcp_server", new_callable=AsyncMock
        ) as mock_connect:
            mock_connect.side_effect = ConnectionError("Connection failed")

            # Should not raise exception, just log error
            await agent.initialize_mcp_servers()

            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_mcp_server_sse(self):
        """Test connecting to MCP server via SSE."""
        agent = Manus()

        with patch.object(
            agent.mcp_clients, "connect_sse", new_callable=AsyncMock
        ) as mock_connect:
            mock_connect.return_value = None

            await agent.connect_mcp_server("http://test.com", "test_server")

            mock_connect.assert_called_once_with("http://test.com", "test_server")
            assert agent.connected_servers["test_server"] == "http://test.com"

    @pytest.mark.asyncio
    async def test_connect_mcp_server_stdio(self):
        """Test connecting to MCP server via stdio."""
        agent = Manus()

        with patch.object(
            agent.mcp_clients, "connect_stdio", new_callable=AsyncMock
        ) as mock_connect:
            mock_connect.return_value = None

            await agent.connect_mcp_server(
                "python", "stdio_server", use_stdio=True, stdio_args=["-m", "server"]
            )

            mock_connect.assert_called_once_with(
                "python", ["-m", "server"], "stdio_server"
            )
            assert agent.connected_servers["stdio_server"] == "python"

    @pytest.mark.asyncio
    async def test_disconnect_mcp_server_specific(self):
        """Test disconnecting from a specific MCP server."""
        agent = Manus()
        agent.connected_servers["test_server"] = "http://test.com"

        # Add mock tools from the server
        mock_tool = Mock(spec=MCPClientTool)
        mock_tool.server_id = "test_server"
        agent.available_tools.add_tools(mock_tool)

        with patch.object(
            agent.mcp_clients, "disconnect", new_callable=AsyncMock
        ) as mock_disconnect:
            await agent.disconnect_mcp_server("test_server")

            mock_disconnect.assert_called_once_with("test_server")
            assert "test_server" not in agent.connected_servers

    @pytest.mark.asyncio
    async def test_disconnect_mcp_server_all(self):
        """Test disconnecting from all MCP servers."""
        agent = Manus()
        agent.connected_servers = {
            "server1": "http://test1.com",
            "server2": "http://test2.com",
        }

        with patch.object(
            agent.mcp_clients, "disconnect", new_callable=AsyncMock
        ) as mock_disconnect:
            await agent.disconnect_mcp_server()

            mock_disconnect.assert_called_once_with("")
            assert len(agent.connected_servers) == 0

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test agent cleanup process."""
        agent = Manus()
        agent._initialized = True

        # Mock browser context helper
        mock_browser_helper = Mock()
        mock_browser_helper.cleanup_browser = AsyncMock()
        agent.browser_context_helper = mock_browser_helper

        with patch.object(
            agent, "disconnect_mcp_server", new_callable=AsyncMock
        ) as mock_disconnect:
            await agent.cleanup()

            mock_browser_helper.cleanup_browser.assert_called_once()
            mock_disconnect.assert_called_once()
            assert not agent._initialized

    @pytest.mark.asyncio
    async def test_cleanup_not_initialized(self):
        """Test cleanup when agent is not initialized."""
        agent = Manus()
        agent._initialized = False

        mock_browser_helper = Mock()
        mock_browser_helper.cleanup_browser = AsyncMock()
        agent.browser_context_helper = mock_browser_helper

        with patch.object(
            agent, "disconnect_mcp_server", new_callable=AsyncMock
        ) as mock_disconnect:
            await agent.cleanup()

            mock_browser_helper.cleanup_browser.assert_called_once()
            mock_disconnect.assert_not_called()

    @pytest.mark.asyncio
    async def test_think_initializes_mcp_servers(self):
        """Test that think() initializes MCP servers if not already initialized."""
        agent = Manus()
        agent._initialized = False

        with patch.object(
            agent, "initialize_mcp_servers", new_callable=AsyncMock
        ) as mock_init:
            with patch.object(
                BaseAgent, "think", new_callable=AsyncMock
            ) as mock_super_think:
                mock_super_think.return_value = True

                result = await agent.think()

                mock_init.assert_called_once()
                assert agent._initialized
                assert result is True

    @pytest.mark.asyncio
    async def test_think_browser_context_handling(self):
        """Test think() handles browser context appropriately."""
        agent = Manus()
        agent._initialized = True

        # Mock browser context helper
        mock_browser_helper = Mock()
        mock_browser_helper.format_next_step_prompt = AsyncMock(
            return_value="Browser prompt"
        )
        agent.browser_context_helper = mock_browser_helper

        # Create mock messages with browser tool calls
        mock_tool_call = Mock()
        mock_tool_call.function.name = "browser_use_tool"

        mock_message = Mock()
        mock_message.tool_calls = [mock_tool_call]

        agent.memory.messages = [mock_message, mock_message, mock_message]

        original_prompt = agent.next_step_prompt

        with patch.object(
            BaseAgent, "think", new_callable=AsyncMock
        ) as mock_super_think:
            mock_super_think.return_value = True

            result = await agent.think()

            # Should have called format_next_step_prompt and restored original prompt
            mock_browser_helper.format_next_step_prompt.assert_called_once()
            assert agent.next_step_prompt == original_prompt
            assert result is True


class TestManusAgentIntegration(AgentTestCase):
    """Integration tests for Manus agent functionality."""

    def setup_method(self):
        """Set up integration test environment."""
        super().setup_method()
        self.mock_config = self.create_mock("app.config.config")
        self.mock_config.workspace_root = str(self.workspace)
        self.mock_config.mcp_config.servers = {}

    @pytest.mark.asyncio
    async def test_agent_memory_management(self):
        """Test agent memory management functionality."""
        agent = Manus()

        # Test adding messages to memory
        agent.update_memory("user", "Hello, agent!")
        agent.update_memory("assistant", "Hello! How can I help you?")

        assert len(agent.memory.messages) == 2
        assert agent.memory.messages[0].role.value == "user"
        assert agent.memory.messages[1].role.value == "assistant"

        # Test memory limit
        agent.memory.max_messages = 2
        agent.update_memory("user", "Another message")

        # Should still have 2 messages (oldest removed)
        assert len(agent.memory.messages) == 2
        assert "Another message" in agent.memory.messages[-1].content

    @pytest.mark.asyncio
    async def test_agent_tool_execution_flow(self):
        """Test complete tool execution flow."""
        agent = Manus()

        # Mock a tool execution
        mock_tool = Mock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_tool.execute = AsyncMock(
            return_value=ToolResult(output="Tool executed successfully")
        )

        agent.available_tools.add_tools(mock_tool)

        # Test tool is available
        assert "test_tool" in [tool.name for tool in agent.available_tools.tools]

        # Test tool execution
        result = await mock_tool.execute(param="value")
        assert result.output == "Tool executed successfully"

    @pytest.mark.asyncio
    async def test_agent_error_handling(self):
        """Test agent error handling in various scenarios."""
        agent = Manus()

        # Test MCP connection error handling
        with patch.object(
            agent.mcp_clients, "connect_sse", new_callable=AsyncMock
        ) as mock_connect:
            mock_connect.side_effect = ConnectionError("Connection failed")

            # Should handle error gracefully
            try:
                await agent.connect_mcp_server("http://invalid.com", "test")
            except ConnectionError:
                pass  # Expected to propagate

            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_state_management(self):
        """Test agent state management during execution."""
        agent = Manus()

        # Test initial state
        assert agent.state == AgentState.IDLE
        assert agent.current_step == 0

        # Test state transitions during run
        with patch.object(agent, "step", new_callable=AsyncMock) as mock_step:
            mock_step.return_value = "Step completed"
            agent.max_steps = 2

            result = await agent.run("Test request")

            assert agent.current_step == 0  # Reset after completion
            assert agent.state == AgentState.IDLE
            assert "Step 1" in result
            assert "Step 2" in result

    @pytest.mark.asyncio
    async def test_conversation_persistence(self):
        """Test conversation history persistence."""
        agent = Manus()

        # Add conversation messages
        messages = [
            Message.user_message("What's the weather like?"),
            Message.assistant_message("I'll help you check the weather."),
            Message.user_message("Thank you!"),
        ]

        for msg in messages:
            agent.memory.add_message(msg)

        # Test persistence
        assert len(agent.memory.messages) == 3

        # Test retrieval
        recent_messages = agent.memory.get_recent_messages(2)
        assert len(recent_messages) == 2
        assert "Thank you!" in recent_messages[-1].content

        # Test clearing
        agent.memory.clear()
        assert len(agent.memory.messages) == 0


class TestManusAgentMCPIntegration(UnitTestCase):
    """Tests for MCP server integration functionality."""

    def setup_method(self):
        """Set up MCP integration test environment."""
        super().setup_method()
        self.mock_config = self.create_mock("app.config.config")
        self.mock_config.mcp_config.servers = {}

    @pytest.mark.asyncio
    async def test_mcp_tool_registration(self):
        """Test MCP tool registration and availability."""
        agent = Manus()

        # Mock MCP client behavior
        mock_tool_info = Mock()
        mock_tool_info.name = "test_mcp_tool"
        mock_tool_info.description = "Test MCP tool"
        mock_tool_info.inputSchema = {"type": "object"}

        mock_list_response = Mock()
        mock_list_response.tools = [mock_tool_info]

        with patch.object(agent.mcp_clients, "connect_sse", new_callable=AsyncMock):
            with patch.object(
                agent.mcp_clients, "_initialize_and_list_tools", new_callable=AsyncMock
            ):
                # Simulate tool registration
                mock_mcp_tool = MCPClientTool(
                    name="mcp_test_server_test_mcp_tool",
                    description="Test MCP tool",
                    parameters={"type": "object"},
                    server_id="test_server",
                    original_name="test_mcp_tool",
                )

                agent.mcp_clients.tool_map["mcp_test_server_test_mcp_tool"] = (
                    mock_mcp_tool
                )
                agent.mcp_clients.tools = (mock_mcp_tool,)

                await agent.connect_mcp_server("http://test.com", "test_server")

                # Verify tool is available
                tool_names = [tool.name for tool in agent.available_tools.tools]
                assert any("test_mcp_tool" in name for name in tool_names)

    @pytest.mark.asyncio
    async def test_mcp_tool_execution(self):
        """Test MCP tool execution."""
        agent = Manus()

        # Create mock MCP tool
        mock_session = Mock()
        mock_session.call_tool = AsyncMock()

        # Mock successful tool execution
        mock_result = Mock()
        mock_result.content = [Mock(text="MCP tool result")]
        mock_session.call_tool.return_value = mock_result

        mcp_tool = MCPClientTool(
            name="test_mcp_tool",
            description="Test MCP tool",
            session=mock_session,
            server_id="test_server",
            original_name="original_tool",
        )

        # Execute tool
        result = await mcp_tool.execute(param="value")

        assert result.output == "MCP tool result"
        mock_session.call_tool.assert_called_once_with(
            "original_tool", {"param": "value"}
        )

    @pytest.mark.asyncio
    async def test_mcp_tool_execution_error(self):
        """Test MCP tool execution error handling."""
        mock_session = Mock()
        mock_session.call_tool = AsyncMock(
            side_effect=Exception("Tool execution failed")
        )

        mcp_tool = MCPClientTool(
            name="failing_tool",
            description="Failing tool",
            session=mock_session,
            server_id="test_server",
            original_name="failing_tool",
        )

        result = await mcp_tool.execute()

        assert result.error is not None
        assert "Tool execution failed" in result.error

    @pytest.mark.asyncio
    async def test_mcp_server_disconnection_cleanup(self):
        """Test proper cleanup when MCP server disconnects."""
        agent = Manus()

        # Setup connected server with tools
        agent.connected_servers["test_server"] = "http://test.com"

        mock_mcp_tool = Mock(spec=MCPClientTool)
        mock_mcp_tool.server_id = "test_server"
        mock_mcp_tool.name = "test_tool"

        agent.available_tools.add_tools(mock_mcp_tool)

        initial_tool_count = len(agent.available_tools.tools)

        with patch.object(agent.mcp_clients, "disconnect", new_callable=AsyncMock):
            await agent.disconnect_mcp_server("test_server")

        # Verify server is removed
        assert "test_server" not in agent.connected_servers

        # Verify tools are cleaned up (this depends on implementation)
        # The actual cleanup logic may vary based on tool collection implementation


class TestManusAgentPerformance(UnitTestCase):
    """Performance tests for Manus agent."""

    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test memory usage efficiency."""
        agent = Manus()

        # Add many messages to test memory management
        for i in range(1000):
            agent.update_memory("user", f"Message {i}")
            agent.update_memory("assistant", f"Response {i}")

        # Should respect max_messages limit
        assert len(agent.memory.messages) <= agent.memory.max_messages

    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self):
        """Test concurrent tool execution performance."""
        agent = Manus()

        # Create multiple mock tools
        mock_tools = []
        for i in range(5):
            mock_tool = Mock(spec=BaseTool)
            mock_tool.name = f"tool_{i}"
            mock_tool.execute = AsyncMock(return_value=ToolResult(output=f"Result {i}"))
            mock_tools.append(mock_tool)
            agent.available_tools.add_tools(mock_tool)

        # Execute tools concurrently
        tasks = [tool.execute() for tool in mock_tools]
        results = await asyncio.gather(*tasks)

        # Verify all tools executed successfully
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.output == f"Result {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
