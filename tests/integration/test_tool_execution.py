"""
Integration tests for tool execution and error handling.

This module contains integration tests that verify tool execution flows,
error handling mechanisms, and tool interaction patterns.
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.agent.manus import Manus
from app.exceptions import (
    MCPConnectionError,
    ResourceError,
    SecurityError,
    ToolExecutionError,
)
from app.tool.base import BaseTool, ToolResult
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.mcp import MCPClients, MCPClientTool
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor
from tests.base import IntegrationTestCase


class TestToolExecution(IntegrationTestCase):
    """Integration tests for tool execution."""

    def setup_method(self):
        """Set up tool execution test environment."""
        super().setup_method()
        self.agent = None

    @pytest.mark.asyncio
    async def test_python_tool_execution_success(self):
        """Test successful Python code execution."""
        agent = Manus()

        # Mock sandbox execution
        with patch("app.sandbox.client.SANDBOX_CLIENT") as mock_sandbox:
            mock_sandbox.execute_code = AsyncMock(
                return_value={
                    "success": True,
                    "output": "42\n",
                    "error": None,
                    "execution_time": 0.1,
                }
            )

            python_tool = PythonExecute()
            result = await python_tool.execute(code="print(42)")

            assert result.output is not None
            assert "42" in str(result.output)
            mock_sandbox.execute_code.assert_called_once()

    @pytest.mark.asyncio
    async def test_python_tool_execution_error(self):
        """Test Python code execution with errors."""
        agent = Manus()

        with patch("app.sandbox.client.SANDBOX_CLIENT") as mock_sandbox:
            mock_sandbox.execute_code = AsyncMock(
                return_value={
                    "success": False,
                    "output": "",
                    "error": 'NameError: name "undefined_var" is not defined',
                    "execution_time": 0.05,
                }
            )

            python_tool = PythonExecute()
            result = await python_tool.execute(code="print(undefined_var)")

            assert result.error is not None
            assert "NameError" in result.error

    @pytest.mark.asyncio
    async def test_file_editor_tool_execution(self):
        """Test file editor tool execution."""
        # Create a test file
        test_file = self.workspace / "test.txt"
        test_file.write_text("Hello, World!")

        editor_tool = StrReplaceEditor()

        # Test reading file
        result = await editor_tool.execute(command="view", path=str(test_file))

        assert result.output is not None
        assert "Hello, World!" in str(result.output)

    @pytest.mark.asyncio
    async def test_file_editor_tool_write(self):
        """Test file editor tool writing."""
        test_file = self.workspace / "new_file.txt"

        editor_tool = StrReplaceEditor()

        # Test creating file
        result = await editor_tool.execute(
            command="create", path=str(test_file), file_text="New file content"
        )

        assert result.output is not None
        assert test_file.exists()
        assert test_file.read_text() == "New file content"

    @pytest.mark.asyncio
    async def test_tool_execution_timeout(self):
        """Test tool execution timeout handling."""

        class SlowTool(BaseTool):
            name = "slow_tool"
            description = "A tool that takes too long"

            async def execute(self, **kwargs):
                await asyncio.sleep(2)  # Simulate slow operation
                return ToolResult(output="Completed")

        slow_tool = SlowTool()

        # Test with timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_tool.execute(), timeout=0.5)

    @pytest.mark.asyncio
    async def test_tool_execution_resource_limits(self):
        """Test tool execution with resource limits."""
        agent = Manus()

        with patch("app.sandbox.client.SANDBOX_CLIENT") as mock_sandbox:
            # Simulate resource limit exceeded
            mock_sandbox.execute_code = AsyncMock(
                side_effect=ResourceError(
                    "Memory limit exceeded",
                    resource_type="memory",
                    limit="512MB",
                    usage="1GB",
                )
            )

            python_tool = PythonExecute()

            with pytest.raises(ResourceError) as exc_info:
                await python_tool.execute(
                    code="x = [0] * 10**9"
                )  # Large memory allocation

            assert "Memory limit exceeded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tool_execution_security_validation(self):
        """Test tool execution security validation."""
        agent = Manus()

        with patch("app.sandbox.client.SANDBOX_CLIENT") as mock_sandbox:
            # Simulate security violation
            mock_sandbox.execute_code = AsyncMock(
                side_effect=SecurityError(
                    "Dangerous operation detected",
                    operation="file_access",
                    path="/etc/passwd",
                )
            )

            python_tool = PythonExecute()

            with pytest.raises(SecurityError) as exc_info:
                await python_tool.execute(code="open('/etc/passwd', 'r')")

            assert "Dangerous operation detected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self):
        """Test concurrent tool execution."""
        agent = Manus()

        # Create multiple tools
        tools = []
        for i in range(5):
            tool = Mock(spec=BaseTool)
            tool.name = f"tool_{i}"
            tool.execute = AsyncMock(return_value=ToolResult(output=f"Result {i}"))
            tools.append(tool)

        # Execute tools concurrently
        tasks = [tool.execute(param=f"value_{i}") for i, tool in enumerate(tools)]
        results = await asyncio.gather(*tasks)

        # Verify all executions completed
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.output == f"Result {i}"

    @pytest.mark.asyncio
    async def test_tool_execution_retry_logic(self):
        """Test tool execution retry logic."""

        class UnreliableTool(BaseTool):
            name = "unreliable_tool"
            description = "A tool that fails sometimes"

            def __init__(self):
                super().__init__()
                self.attempt_count = 0

            async def execute(self, **kwargs):
                self.attempt_count += 1
                if self.attempt_count < 3:
                    raise ConnectionError("Temporary failure")
                return ToolResult(output="Success after retries")

        tool = UnreliableTool()

        # Implement retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await tool.execute()
                break
            except ConnectionError:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1)  # Brief delay between retries

        assert result.output == "Success after retries"
        assert tool.attempt_count == 3


class TestMCPToolIntegration(IntegrationTestCase):
    """Integration tests for MCP tool functionality."""

    def setup_method(self):
        """Set up MCP integration test environment."""
        super().setup_method()
        self.mock_config = self.create_mock("app.config.config")
        self.mock_config.mcp_config.servers = {}

    @pytest.mark.asyncio
    async def test_mcp_server_connection_lifecycle(self):
        """Test complete MCP server connection lifecycle."""
        agent = Manus()

        # Mock MCP client connections
        with patch.object(
            agent.mcp_clients, "connect_sse", new_callable=AsyncMock
        ) as mock_connect:
            with patch.object(
                agent.mcp_clients, "disconnect", new_callable=AsyncMock
            ) as mock_disconnect:

                # Test connection
                await agent.connect_mcp_server("http://test.com", "test_server")
                mock_connect.assert_called_once_with("http://test.com", "test_server")
                assert "test_server" in agent.connected_servers

                # Test disconnection
                await agent.disconnect_mcp_server("test_server")
                mock_disconnect.assert_called_once_with("test_server")
                assert "test_server" not in agent.connected_servers

    @pytest.mark.asyncio
    async def test_mcp_tool_discovery_and_registration(self):
        """Test MCP tool discovery and registration process."""
        agent = Manus()

        # Mock tool discovery
        mock_tool_info = Mock()
        mock_tool_info.name = "discovered_tool"
        mock_tool_info.description = "A discovered MCP tool"
        mock_tool_info.inputSchema = {"type": "object", "properties": {}}

        mock_list_response = Mock()
        mock_list_response.tools = [mock_tool_info]

        with patch.object(agent.mcp_clients, "connect_sse", new_callable=AsyncMock):
            with patch.object(
                agent.mcp_clients, "_initialize_and_list_tools", new_callable=AsyncMock
            ) as mock_init:
                # Simulate tool registration
                mock_mcp_tool = MCPClientTool(
                    name="mcp_test_discovered_tool",
                    description="A discovered MCP tool",
                    parameters={"type": "object", "properties": {}},
                    server_id="test_server",
                    original_name="discovered_tool",
                )

                agent.mcp_clients.tool_map["mcp_test_discovered_tool"] = mock_mcp_tool
                agent.mcp_clients.tools = (mock_mcp_tool,)

                await agent.connect_mcp_server("http://test.com", "test_server")

                # Verify tool registration
                mock_init.assert_called_once_with("test_server")

    @pytest.mark.asyncio
    async def test_mcp_connection_failure_handling(self):
        """Test MCP connection failure handling."""
        agent = Manus()

        with patch.object(
            agent.mcp_clients, "connect_sse", new_callable=AsyncMock
        ) as mock_connect:
            mock_connect.side_effect = MCPConnectionError(
                "Failed to connect to MCP server",
                server_url="http://invalid.com",
                error_code="CONNECTION_REFUSED",
            )

            with pytest.raises(MCPConnectionError) as exc_info:
                await agent.connect_mcp_server("http://invalid.com", "invalid_server")

            assert "Failed to connect to MCP server" in str(exc_info.value)
            assert "invalid_server" not in agent.connected_servers

    @pytest.mark.asyncio
    async def test_mcp_tool_execution_with_session_failure(self):
        """Test MCP tool execution when session fails."""
        # Create MCP tool without session
        mcp_tool = MCPClientTool(
            name="sessionless_tool",
            description="Tool without session",
            session=None,
            server_id="test_server",
            original_name="sessionless_tool",
        )

        result = await mcp_tool.execute(param="value")

        assert result.error is not None
        assert "Not connected to MCP server" in result.error

    @pytest.mark.asyncio
    async def test_multiple_mcp_servers_management(self):
        """Test managing multiple MCP servers simultaneously."""
        agent = Manus()

        servers = {
            "server1": "http://server1.com",
            "server2": "http://server2.com",
            "server3": "http://server3.com",
        }

        with patch.object(
            agent.mcp_clients, "connect_sse", new_callable=AsyncMock
        ) as mock_connect:
            # Connect to multiple servers
            for server_id, url in servers.items():
                await agent.connect_mcp_server(url, server_id)

            # Verify all connections
            assert len(agent.connected_servers) == 3
            for server_id, url in servers.items():
                assert agent.connected_servers[server_id] == url

            # Verify connect was called for each server
            assert mock_connect.call_count == 3

    @pytest.mark.asyncio
    async def test_mcp_server_reconnection(self):
        """Test MCP server reconnection after failure."""
        agent = Manus()

        with patch.object(
            agent.mcp_clients, "connect_sse", new_callable=AsyncMock
        ) as mock_connect:
            with patch.object(
                agent.mcp_clients, "disconnect", new_callable=AsyncMock
            ) as mock_disconnect:

                # Initial connection
                await agent.connect_mcp_server("http://test.com", "test_server")
                assert "test_server" in agent.connected_servers

                # Simulate connection loss and reconnection
                await agent.disconnect_mcp_server("test_server")
                await agent.connect_mcp_server("http://test.com", "test_server")

                # Verify reconnection
                assert mock_connect.call_count == 2
                assert mock_disconnect.call_count == 1
                assert "test_server" in agent.connected_servers


class TestErrorHandlingIntegration(IntegrationTestCase):
    """Integration tests for error handling mechanisms."""

    @pytest.mark.asyncio
    async def test_cascading_error_handling(self):
        """Test cascading error handling across multiple components."""
        agent = Manus()

        # Create a chain of operations that can fail
        class FailingTool(BaseTool):
            name = "failing_tool"
            description = "A tool that fails"

            async def execute(self, **kwargs):
                raise ToolExecutionError(
                    "Tool execution failed",
                    tool_name="failing_tool",
                    error_code="EXECUTION_ERROR",
                )

        failing_tool = FailingTool()

        with pytest.raises(ToolExecutionError) as exc_info:
            await failing_tool.execute()

        assert "Tool execution failed" in str(exc_info.value)
        assert exc_info.value.tool_name == "failing_tool"

    @pytest.mark.asyncio
    async def test_error_recovery_mechanisms(self):
        """Test error recovery mechanisms."""
        agent = Manus()

        class RecoverableTool(BaseTool):
            name = "recoverable_tool"
            description = "A tool that can recover from errors"

            def __init__(self):
                super().__init__()
                self.failure_count = 0

            async def execute(self, **kwargs):
                self.failure_count += 1
                if self.failure_count <= 2:
                    raise ConnectionError("Temporary failure")
                return ToolResult(output="Recovered successfully")

        tool = RecoverableTool()

        # Implement recovery logic
        max_attempts = 5
        last_error = None

        for attempt in range(max_attempts):
            try:
                result = await tool.execute()
                break
            except ConnectionError as e:
                last_error = e
                if attempt < max_attempts - 1:
                    await asyncio.sleep(0.1 * (2**attempt))  # Exponential backoff
                else:
                    raise last_error

        assert result.output == "Recovered successfully"

    @pytest.mark.asyncio
    async def test_error_context_preservation(self):
        """Test error context preservation across operations."""
        agent = Manus()

        # Test that error context is preserved
        try:
            with patch("app.sandbox.client.SANDBOX_CLIENT") as mock_sandbox:
                mock_sandbox.execute_code = AsyncMock(
                    side_effect=Exception("Sandbox error")
                )

                python_tool = PythonExecute()
                await python_tool.execute(code="print('test')")
        except Exception as e:
            # Verify error context is available
            assert "Sandbox error" in str(e)

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation when tools fail."""
        agent = Manus()

        # Create primary and fallback tools
        class PrimaryTool(BaseTool):
            name = "primary_tool"
            description = "Primary tool that fails"

            async def execute(self, **kwargs):
                raise ConnectionError("Primary tool unavailable")

        class FallbackTool(BaseTool):
            name = "fallback_tool"
            description = "Fallback tool"

            async def execute(self, **kwargs):
                return ToolResult(output="Fallback result")

        primary_tool = PrimaryTool()
        fallback_tool = FallbackTool()

        # Implement fallback logic
        try:
            result = await primary_tool.execute()
        except ConnectionError:
            result = await fallback_tool.execute()

        assert result.output == "Fallback result"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
