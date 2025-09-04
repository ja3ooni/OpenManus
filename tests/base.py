"""
Base test classes for OpenManus testing framework.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import psutil
import pytest
from faker import Faker

from app.agent.base import BaseAgent
from app.config import Config
from app.exceptions import OpenManusError
from app.schema import AgentResponse, ToolResult


class BaseTestCase(ABC):
    """Base test case with common utilities."""

    def setup_method(self):
        """Set up method called before each test."""
        self.fake = Faker()
        self.start_time = time.time()

    def teardown_method(self):
        """Tear down method called after each test."""
        pass

    def assert_response_valid(self, response: AgentResponse):
        """Assert that an agent response is valid."""
        assert response is not None
        assert hasattr(response, "content")
        assert hasattr(response, "success")
        if not response.success:
            assert hasattr(response, "error")

    def assert_tool_result_valid(self, result: ToolResult):
        """Assert that a tool result is valid."""
        assert result is not None
        assert hasattr(result, "success")
        assert hasattr(result, "result")
        if not result.success:
            assert hasattr(result, "error")

    def generate_test_data(self, size: str = "small") -> Dict[str, Any]:
        """Generate test data of specified size."""
        sizes = {
            "small": 10,
            "medium": 100,
            "large": 1000,
        }

        count = sizes.get(size, 10)
        return {
            "items": [
                {
                    "id": self.fake.uuid4(),
                    "name": self.fake.name(),
                    "email": self.fake.email(),
                    "text": self.fake.text(),
                }
                for _ in range(count)
            ]
        }


class UnitTestCase(BaseTestCase):
    """Base class for unit tests."""

    def setup_method(self):
        """Set up unit test environment."""
        super().setup_method()
        self.mocks = {}

    def teardown_method(self):
        """Clean up unit test environment."""
        # Stop all mocks
        for mock in self.mocks.values():
            if hasattr(mock, "stop"):
                mock.stop()
        super().teardown_method()

    def create_mock(self, target: str, **kwargs) -> Mock:
        """Create and register a mock."""
        mock = patch(target, **kwargs)
        self.mocks[target] = mock
        return mock.start()


class IntegrationTestCase(BaseTestCase):
    """Base class for integration tests."""

    @pytest.fixture(autouse=True)
    def setup_integration(self, mock_config, temp_workspace):
        """Set up integration test environment."""
        self.config = mock_config
        self.workspace = temp_workspace

    def setup_method(self):
        """Set up integration test environment."""
        super().setup_method()
        self.external_services = {}

    async def wait_for_condition(
        self, condition_func, timeout: float = 10.0, interval: float = 0.1
    ) -> bool:
        """Wait for a condition to become true."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await condition_func():
                return True
            await asyncio.sleep(interval)
        return False

    def assert_external_service_called(
        self, service_name: str, expected_calls: int = 1
    ):
        """Assert that an external service was called the expected number of times."""
        service = self.external_services.get(service_name)
        assert service is not None, f"External service {service_name} not found"
        assert service.call_count == expected_calls


class PerformanceTestCase(BaseTestCase):
    """Base class for performance tests."""

    def setup_method(self):
        """Set up performance test environment."""
        super().setup_method()
        self.performance_metrics = {}
        self.memory_start = psutil.virtual_memory().used
        self.cpu_start = psutil.cpu_percent()

    def teardown_method(self):
        """Clean up and record performance metrics."""
        self.performance_metrics.update(
            {
                "execution_time": time.time() - self.start_time,
                "memory_delta": psutil.virtual_memory().used - self.memory_start,
                "cpu_usage": psutil.cpu_percent(),
            }
        )
        super().teardown_method()

    def assert_performance_within_limits(
        self,
        max_time: float = 30.0,
        max_memory_mb: float = 100.0,
        max_cpu_percent: float = 80.0,
    ):
        """Assert that performance is within acceptable limits."""
        execution_time = self.performance_metrics.get("execution_time", 0)
        memory_delta_mb = self.performance_metrics.get("memory_delta", 0) / (
            1024 * 1024
        )
        cpu_usage = self.performance_metrics.get("cpu_usage", 0)

        assert (
            execution_time <= max_time
        ), f"Execution time {execution_time}s exceeds limit {max_time}s"
        assert (
            memory_delta_mb <= max_memory_mb
        ), f"Memory usage {memory_delta_mb}MB exceeds limit {max_memory_mb}MB"
        assert (
            cpu_usage <= max_cpu_percent
        ), f"CPU usage {cpu_usage}% exceeds limit {max_cpu_percent}%"

    async def measure_concurrent_performance(
        self, async_func, concurrent_count: int = 5, **kwargs
    ) -> Dict[str, Any]:
        """Measure performance of concurrent operations."""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used

        # Run concurrent operations
        tasks = [async_func(**kwargs) for _ in range(concurrent_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        end_memory = psutil.virtual_memory().used

        # Count successful operations
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful

        return {
            "total_time": end_time - start_time,
            "avg_time_per_operation": (end_time - start_time) / concurrent_count,
            "memory_delta": end_memory - start_memory,
            "successful_operations": successful,
            "failed_operations": failed,
            "success_rate": successful / len(results),
            "operations_per_second": concurrent_count / (end_time - start_time),
        }


class SecurityTestCase(BaseTestCase):
    """Base class for security tests."""

    def setup_method(self):
        """Set up security test environment."""
        super().setup_method()
        self.security_violations = []

    def assert_input_sanitized(self, input_value: str, output_value: str):
        """Assert that input has been properly sanitized."""
        dangerous_patterns = [
            "<script",
            "javascript:",
            "'; DROP",
            "../",
            "rm -rf",
            "eval(",
        ]

        for pattern in dangerous_patterns:
            if pattern.lower() in input_value.lower():
                assert (
                    pattern.lower() not in output_value.lower()
                ), f"Dangerous pattern '{pattern}' not sanitized in output"

    def assert_no_sensitive_data_in_logs(self, log_content: str):
        """Assert that logs don't contain sensitive data."""
        sensitive_patterns = [
            r"password\s*[:=]\s*\S+",
            r"api[_-]?key\s*[:=]\s*\S+",
            r"secret\s*[:=]\s*\S+",
            r"token\s*[:=]\s*\S+",
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card
        ]

        import re

        for pattern in sensitive_patterns:
            matches = re.findall(pattern, log_content, re.IGNORECASE)
            assert not matches, f"Sensitive data found in logs: {matches}"

    def test_injection_attacks(self, test_function, payloads: List[str]):
        """Test function against injection attack payloads."""
        for payload in payloads:
            try:
                result = test_function(payload)
                # Check if the payload was executed or caused unexpected behavior
                self.assert_input_sanitized(payload, str(result))
            except Exception as e:
                # Ensure the exception is a proper validation error, not a system error
                assert isinstance(
                    e, (ValueError, OpenManusError)
                ), f"Unexpected exception type for payload '{payload}': {type(e)}"


class AgentTestCase(IntegrationTestCase):
    """Base class for agent tests."""

    def setup_method(self):
        """Set up agent test environment."""
        super().setup_method()
        self.agent = None
        self.conversation_history = []

    async def send_message_to_agent(
        self, message: str, agent: Optional[BaseAgent] = None
    ) -> AgentResponse:
        """Send a message to the agent and return the response."""
        if agent is None:
            agent = self.agent

        assert agent is not None, "No agent available for testing"

        response = await agent.process_message(message)
        self.conversation_history.append(
            {
                "message": message,
                "response": response,
                "timestamp": time.time(),
            }
        )

        return response

    def assert_agent_response_contains(
        self, response: AgentResponse, expected_content: str
    ):
        """Assert that agent response contains expected content."""
        self.assert_response_valid(response)
        assert (
            expected_content.lower() in response.content.lower()
        ), f"Expected content '{expected_content}' not found in response: {response.content}"

    def assert_tool_was_called(self, tool_name: str, expected_calls: int = 1):
        """Assert that a specific tool was called."""
        tool_calls = []
        for entry in self.conversation_history:
            if hasattr(entry["response"], "tool_calls"):
                tool_calls.extend(entry["response"].tool_calls or [])

        actual_calls = sum(1 for call in tool_calls if call.name == tool_name)
        assert (
            actual_calls == expected_calls
        ), f"Expected {expected_calls} calls to {tool_name}, but got {actual_calls}"


class ToolTestCase(UnitTestCase):
    """Base class for tool tests."""

    def setup_method(self):
        """Set up tool test environment."""
        super().setup_method()
        self.tool = None

    async def execute_tool(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        assert self.tool is not None, "No tool available for testing"
        return await self.tool.execute(**kwargs)

    def assert_tool_execution_successful(self, result: ToolResult):
        """Assert that tool execution was successful."""
        self.assert_tool_result_valid(result)
        assert result.success, f"Tool execution failed: {result.error}"

    def assert_tool_execution_failed(
        self, result: ToolResult, expected_error_type: type = None
    ):
        """Assert that tool execution failed as expected."""
        self.assert_tool_result_valid(result)
        assert not result.success, "Expected tool execution to fail"
        if expected_error_type:
            assert isinstance(
                result.error, expected_error_type
            ), f"Expected error type {expected_error_type}, got {type(result.error)}"


class MCPTestCase(IntegrationTestCase):
    """Base class for MCP (Model Context Protocol) tests."""

    def setup_method(self):
        """Set up MCP test environment."""
        super().setup_method()
        self.mcp_servers = {}
        self.mcp_tools = {}

    async def setup_mock_mcp_server(self, server_name: str, tools: List[str]):
        """Set up a mock MCP server with specified tools."""
        mock_server = Mock()
        mock_server.name = server_name
        mock_server.tools = tools
        mock_server.is_connected = True

        self.mcp_servers[server_name] = mock_server
        return mock_server

    def assert_mcp_server_connected(self, server_name: str):
        """Assert that MCP server is connected."""
        server = self.mcp_servers.get(server_name)
        assert server is not None, f"MCP server {server_name} not found"
        assert server.is_connected, f"MCP server {server_name} is not connected"

    def assert_mcp_tool_available(self, tool_name: str):
        """Assert that MCP tool is available."""
        available_tools = []
        for server in self.mcp_servers.values():
            available_tools.extend(server.tools)

        assert (
            tool_name in available_tools
        ), f"MCP tool {tool_name} not available. Available tools: {available_tools}"
