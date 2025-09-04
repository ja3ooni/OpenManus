"""
Comprehensive test configuration and fixtures for OpenManus testing framework.
"""

import asyncio
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import pytest_asyncio
from faker import Faker
from httpx import AsyncClient
from pytest_httpserver import HTTPServer

from app.agent.base import BaseAgent
from app.agent.manus import Manus
from app.config import Config
from app.exceptions import OpenManusError
from app.llm import LLMClient
from app.schema import (
    AgentResponse,
    ConversationMessage,
    LLMConfig,
    ToolCall,
    ToolResult,
)
from app.tool.base import BaseTool

# Test configuration
pytest_plugins = ["pytest_asyncio"]

# Global test settings
TEST_TIMEOUT = 30
INTEGRATION_TEST_TIMEOUT = 120
PERFORMANCE_TEST_TIMEOUT = 300

# Faker instance for generating test data
fake = Faker()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Test configuration dictionary."""
    return {
        "environment": "test",
        "debug": True,
        "log_level": "DEBUG",
        "workspace_root": "/tmp/test_workspace",
        "max_iterations": 10,
        "timeout": TEST_TIMEOUT,
        "llm": {
            "provider": "mock",
            "model": "test-model",
            "temperature": 0.0,
            "max_tokens": 1000,
        },
        "sandbox": {
            "enabled": True,
            "timeout": 60,
            "memory_limit": "512MB",
            "cpu_limit": "1.0",
        },
        "security": {
            "enable_input_validation": True,
            "max_file_size": "10MB",
            "allowed_domains": ["example.com", "test.com"],
        },
        "monitoring": {
            "enabled": True,
            "metrics_port": 9090,
            "health_check_interval": 30,
        },
    }


@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Create a temporary workspace directory for testing."""
    with tempfile.TemporaryDirectory(prefix="openmanus_test_") as temp_dir:
        workspace_path = Path(temp_dir)
        # Create common subdirectories
        (workspace_path / "files").mkdir()
        (workspace_path / "logs").mkdir()
        (workspace_path / "cache").mkdir()
        yield workspace_path


@pytest.fixture
def mock_config(test_config: Dict[str, Any], temp_workspace: Path) -> Config:
    """Mock configuration for testing."""
    config_dict = test_config.copy()
    config_dict["workspace_root"] = str(temp_workspace)

    with patch("app.config.Config._instance", None):
        config = Config()
        config._config = config_dict
        yield config


@pytest.fixture
def correlation_id() -> str:
    """Generate a unique correlation ID for test tracing."""
    return f"test-{uuid.uuid4().hex[:8]}"


# LLM and API Mocking Fixtures


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self):
        self.call_count = 0
        self.responses = []
        self.errors = []
        self.delay = 0

    def set_responses(self, responses: List[str]):
        """Set predefined responses."""
        self.responses = responses
        self.call_count = 0

    def set_errors(self, errors: List[Exception]):
        """Set predefined errors to raise."""
        self.errors = errors
        self.call_count = 0

    def set_delay(self, delay: float):
        """Set artificial delay for responses."""
        self.delay = delay

    async def create_completion(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> Dict[str, Any]:
        """Mock completion creation."""
        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.errors and self.call_count < len(self.errors):
            error = self.errors[self.call_count]
            self.call_count += 1
            raise error

        response_index = min(self.call_count, len(self.responses) - 1)
        response = self.responses[response_index] if self.responses else "Mock response"

        self.call_count += 1

        return {
            "id": f"mock-completion-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """Mock LLM client fixture."""
    return MockLLMClient()


@pytest.fixture
def mock_llm_config() -> LLMConfig:
    """Mock LLM configuration."""
    return LLMConfig(
        provider="mock",
        model="test-model",
        temperature=0.0,
        max_tokens=1000,
        api_key="mock-api-key",
    )


# Tool Mocking Fixtures


class MockTool(BaseTool):
    """Mock tool for testing."""

    name = "mock_tool"
    description = "A mock tool for testing"

    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.responses = []
        self.errors = []
        self.delay = 0

    def set_responses(self, responses: List[Any]):
        """Set predefined responses."""
        self.responses = responses
        self.call_count = 0

    def set_errors(self, errors: List[Exception]):
        """Set predefined errors to raise."""
        self.errors = errors
        self.call_count = 0

    def set_delay(self, delay: float):
        """Set artificial delay for responses."""
        self.delay = delay

    async def execute(self, **kwargs) -> ToolResult:
        """Mock tool execution."""
        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.errors and self.call_count < len(self.errors):
            error = self.errors[self.call_count]
            self.call_count += 1
            raise error

        response_index = min(self.call_count, len(self.responses) - 1)
        response = (
            self.responses[response_index] if self.responses else "Mock tool result"
        )

        self.call_count += 1

        return ToolResult(
            success=True,
            result=response,
            metadata={"call_count": self.call_count},
        )


@pytest.fixture
def mock_tool() -> MockTool:
    """Mock tool fixture."""
    return MockTool()


@pytest.fixture
def mock_tool_collection(mock_tool: MockTool) -> Dict[str, BaseTool]:
    """Mock tool collection fixture."""
    return {"mock_tool": mock_tool}


# Agent Mocking Fixtures


@pytest.fixture
def mock_agent(
    mock_llm_client: MockLLMClient, mock_tool_collection: Dict[str, BaseTool]
) -> Manus:
    """Mock Manus agent for testing."""
    with patch("app.agent.manus.LLMClient") as mock_llm_class:
        mock_llm_class.return_value = mock_llm_client

        agent = Manus()
        agent.tools = mock_tool_collection
        return agent


# External Service Mocking Fixtures


@pytest.fixture
def http_server() -> Generator[HTTPServer, None, None]:
    """HTTP server for mocking external APIs."""
    server = HTTPServer(host="127.0.0.1", port=0)
    server.start()
    yield server
    server.stop()


@pytest.fixture
def mock_external_api(http_server: HTTPServer) -> str:
    """Mock external API endpoint."""
    base_url = f"http://{http_server.host}:{http_server.port}"

    # Set up common endpoints
    http_server.expect_request("/health").respond_with_json({"status": "ok"})
    http_server.expect_request("/api/search").respond_with_json(
        {
            "results": [
                {"title": "Test Result 1", "url": "https://example.com/1"},
                {"title": "Test Result 2", "url": "https://example.com/2"},
            ]
        }
    )

    return base_url


# Database and Storage Mocking Fixtures


@pytest.fixture
def mock_database() -> Dict[str, Any]:
    """Mock database for testing."""
    return {
        "conversations": {},
        "agents": {},
        "tools": {},
        "metrics": {},
    }


@pytest.fixture
def mock_cache() -> Dict[str, Any]:
    """Mock cache for testing."""
    return {}


# Sandbox Environment Fixtures


@pytest.fixture
def mock_sandbox_client():
    """Mock sandbox client for testing."""
    mock_client = Mock()
    mock_client.execute_code = AsyncMock(
        return_value={
            "success": True,
            "output": "Mock execution output",
            "error": None,
            "execution_time": 0.1,
        }
    )
    mock_client.create_file = AsyncMock(return_value=True)
    mock_client.read_file = AsyncMock(return_value="Mock file content")
    mock_client.list_files = AsyncMock(return_value=["file1.py", "file2.txt"])
    return mock_client


# Performance Testing Fixtures


@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture."""
    import time

    import psutil

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.start_cpu = None

        def start(self):
            """Start monitoring."""
            self.start_time = time.time()
            self.start_memory = psutil.virtual_memory().used
            self.start_cpu = psutil.cpu_percent()

        def stop(self) -> Dict[str, float]:
            """Stop monitoring and return metrics."""
            end_time = time.time()
            end_memory = psutil.virtual_memory().used
            end_cpu = psutil.cpu_percent()

            return {
                "execution_time": end_time - self.start_time,
                "memory_delta": end_memory - self.start_memory,
                "cpu_usage": (self.start_cpu + end_cpu) / 2,
            }

    return PerformanceMonitor()


# Security Testing Fixtures


@pytest.fixture
def security_test_payloads() -> Dict[str, List[str]]:
    """Common security test payloads."""
    return {
        "sql_injection": [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users --",
        ],
        "xss": [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert('xss');//",
        ],
        "command_injection": [
            "; ls -la",
            "| cat /etc/passwd",
            "&& rm -rf /",
            "`whoami`",
        ],
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        ],
    }


# Test Data Generation Fixtures


@pytest.fixture
def test_conversation_messages() -> List[ConversationMessage]:
    """Generate test conversation messages."""
    return [
        ConversationMessage(
            role="user",
            content="Hello, can you help me with a task?",
            timestamp=fake.date_time(),
        ),
        ConversationMessage(
            role="assistant",
            content="Of course! I'd be happy to help. What do you need assistance with?",
            timestamp=fake.date_time(),
        ),
        ConversationMessage(
            role="user",
            content="I need to analyze some data and create a report.",
            timestamp=fake.date_time(),
        ),
    ]


@pytest.fixture
def test_tool_calls() -> List[ToolCall]:
    """Generate test tool calls."""
    return [
        ToolCall(
            name="file_read",
            arguments={"path": "data.csv"},
            call_id=fake.uuid4(),
        ),
        ToolCall(
            name="python_execute",
            arguments={"code": "import pandas as pd\ndf = pd.read_csv('data.csv')"},
            call_id=fake.uuid4(),
        ),
        ToolCall(
            name="web_search",
            arguments={"query": "data analysis best practices"},
            call_id=fake.uuid4(),
        ),
    ]


# Async Test Utilities


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client for testing."""
    async with AsyncClient() as client:
        yield client


# Cleanup Fixtures


@pytest.fixture(autouse=True)
def cleanup_environment():
    """Automatically clean up test environment after each test."""
    yield
    # Clean up any global state, temporary files, etc.
    # This runs after each test
    pass


# Parametrized Fixtures for Different Test Scenarios


@pytest.fixture(params=["openai", "anthropic", "azure", "ollama"])
def llm_provider(request) -> str:
    """Parametrized fixture for different LLM providers."""
    return request.param


@pytest.fixture(params=["small", "medium", "large"])
def test_data_size(request) -> str:
    """Parametrized fixture for different test data sizes."""
    return request.param


@pytest.fixture(params=[1, 5, 10])
def concurrent_requests(request) -> int:
    """Parametrized fixture for concurrent request testing."""
    return request.param


# Error Simulation Fixtures


@pytest.fixture
def network_error():
    """Network error for testing."""
    return ConnectionError("Network connection failed")


@pytest.fixture
def timeout_error():
    """Timeout error for testing."""
    return TimeoutError("Operation timed out")


@pytest.fixture
def api_error():
    """API error for testing."""
    return OpenManusError("API request failed", recoverable=True)


# Integration Test Fixtures


@pytest.fixture(scope="session")
def integration_test_config():
    """Configuration for integration tests."""
    return {
        "skip_external": os.getenv("SKIP_EXTERNAL_TESTS", "false").lower() == "true",
        "api_timeout": int(os.getenv("API_TIMEOUT", "30")),
        "max_retries": int(os.getenv("MAX_RETRIES", "3")),
    }


# Markers for Test Organization


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "security: mark test as a security test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line(
        "markers", "external: mark test as requiring external services"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)

        # Add slow marker for tests that take longer than expected
        if hasattr(item, "get_closest_marker"):
            if item.get_closest_marker("slow"):
                item.add_marker(pytest.mark.slow)
