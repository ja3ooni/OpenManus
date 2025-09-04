"""
Test database setup and utilities for integration testing.
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest


class TestDatabase:
    """In-memory test database for integration testing."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            # Use in-memory database for tests
            self.db_path = ":memory:"
        else:
            self.db_path = str(db_path)

        self.connection = None
        self.setup_database()

    def setup_database(self):
        """Set up the test database schema."""
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row

        cursor = self.connection.cursor()

        # Conversations table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                user_id TEXT,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """
        )

        # Messages table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        """
        )

        # Tool calls table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_calls (
                id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                arguments TEXT NOT NULL,
                result TEXT,
                success BOOLEAN,
                execution_time REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (message_id) REFERENCES messages (id)
            )
        """
        )

        # Agents table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                config TEXT,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Performance metrics table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id TEXT PRIMARY KEY,
                operation_type TEXT NOT NULL,
                execution_time REAL NOT NULL,
                memory_usage INTEGER,
                cpu_usage REAL,
                success BOOLEAN NOT NULL,
                error_message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """
        )

        # Security events table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS security_events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                source_ip TEXT,
                user_agent TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """
        )

        self.connection.commit()

    def insert_conversation(
        self,
        conversation_id: str,
        agent_id: str,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Insert a conversation record."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO conversations (id, agent_id, user_id, title, metadata)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                conversation_id,
                agent_id,
                user_id,
                title,
                json.dumps(metadata) if metadata else None,
            ),
        )
        self.connection.commit()

    def insert_message(
        self,
        message_id: str,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Insert a message record."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, metadata)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                message_id,
                conversation_id,
                role,
                content,
                json.dumps(metadata) if metadata else None,
            ),
        )
        self.connection.commit()

    def insert_tool_call(
        self,
        call_id: str,
        message_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Optional[Any] = None,
        success: bool = True,
        execution_time: Optional[float] = None,
    ):
        """Insert a tool call record."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO tool_calls
            (id, message_id, tool_name, arguments, result, success, execution_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                call_id,
                message_id,
                tool_name,
                json.dumps(arguments),
                json.dumps(result) if result else None,
                success,
                execution_time,
            ),
        )
        self.connection.commit()

    def insert_agent(
        self,
        agent_id: str,
        name: str,
        agent_type: str,
        config: Optional[Dict[str, Any]] = None,
        status: str = "active",
    ):
        """Insert an agent record."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO agents (id, name, type, config, status)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                agent_id,
                name,
                agent_type,
                json.dumps(config) if config else None,
                status,
            ),
        )
        self.connection.commit()

    def insert_performance_metric(
        self,
        metric_id: str,
        operation_type: str,
        execution_time: float,
        memory_usage: Optional[int] = None,
        cpu_usage: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Insert a performance metric record."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO performance_metrics
            (id, operation_type, execution_time, memory_usage, cpu_usage,
             success, error_message, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metric_id,
                operation_type,
                execution_time,
                memory_usage,
                cpu_usage,
                success,
                error_message,
                json.dumps(metadata) if metadata else None,
            ),
        )
        self.connection.commit()

    def insert_security_event(
        self,
        event_id: str,
        event_type: str,
        severity: str,
        description: str,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Insert a security event record."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO security_events
            (id, event_type, severity, description, source_ip, user_agent, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                event_id,
                event_type,
                severity,
                description,
                source_ip,
                user_agent,
                json.dumps(metadata) if metadata else None,
            ),
        )
        self.connection.commit()

    def get_conversations(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get conversations, optionally filtered by agent ID."""
        cursor = self.connection.cursor()
        if agent_id:
            cursor.execute(
                "SELECT * FROM conversations WHERE agent_id = ?", (agent_id,)
            )
        else:
            cursor.execute("SELECT * FROM conversations")

        return [dict(row) for row in cursor.fetchall()]

    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get messages for a conversation."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT * FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
        """,
            (conversation_id,),
        )

        return [dict(row) for row in cursor.fetchall()]

    def get_tool_calls(self, message_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get tool calls, optionally filtered by message ID."""
        cursor = self.connection.cursor()
        if message_id:
            cursor.execute(
                "SELECT * FROM tool_calls WHERE message_id = ?", (message_id,)
            )
        else:
            cursor.execute("SELECT * FROM tool_calls ORDER BY timestamp DESC")

        return [dict(row) for row in cursor.fetchall()]

    def get_performance_metrics(
        self, operation_type: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get performance metrics."""
        cursor = self.connection.cursor()
        if operation_type:
            cursor.execute(
                """
                SELECT * FROM performance_metrics
                WHERE operation_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (operation_type, limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM performance_metrics
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (limit,),
            )

        return [dict(row) for row in cursor.fetchall()]

    def get_security_events(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get security events."""
        cursor = self.connection.cursor()

        query = "SELECT * FROM security_events WHERE 1=1"
        params = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if severity:
            query += " AND severity = ?"
            params.append(severity)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def clear_all_data(self):
        """Clear all data from the database."""
        cursor = self.connection.cursor()
        tables = [
            "tool_calls",
            "messages",
            "conversations",
            "agents",
            "performance_metrics",
            "security_events",
        ]

        for table in tables:
            cursor.execute(f"DELETE FROM {table}")

        self.connection.commit()

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()


class MockSandboxEnvironment:
    """Mock sandbox environment for testing."""

    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="sandbox_test_"))
        self.files = {}
        self.execution_history = []
        self.is_running = True

    async def execute_code(
        self, code: str, language: str = "python", timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Mock code execution."""
        execution_record = {
            "code": code,
            "language": language,
            "timeout": timeout,
            "timestamp": "2024-01-01T00:00:00Z",
            "success": True,
            "output": f"Mock execution output for: {code[:50]}...",
            "error": None,
            "execution_time": 0.1,
        }

        # Simulate some failures for testing
        if "raise Exception" in code or "error" in code.lower():
            execution_record.update(
                {
                    "success": False,
                    "output": "",
                    "error": "Mock execution error",
                }
            )

        self.execution_history.append(execution_record)
        return execution_record

    async def create_file(self, path: str, content: str) -> bool:
        """Mock file creation."""
        self.files[path] = content
        return True

    async def read_file(self, path: str) -> str:
        """Mock file reading."""
        if path not in self.files:
            raise FileNotFoundError(f"File not found: {path}")
        return self.files[path]

    async def list_files(self, directory: str = ".") -> List[str]:
        """Mock file listing."""
        return list(self.files.keys())

    async def delete_file(self, path: str) -> bool:
        """Mock file deletion."""
        if path in self.files:
            del self.files[path]
            return True
        return False

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.execution_history.copy()

    def clear_execution_history(self):
        """Clear execution history."""
        self.execution_history.clear()

    def cleanup(self):
        """Clean up the mock sandbox."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


# Pytest fixtures


@pytest.fixture
def test_database():
    """Test database fixture."""
    db = TestDatabase()
    yield db
    db.close()


@pytest.fixture
def persistent_test_database(temp_workspace):
    """Persistent test database fixture."""
    db_path = temp_workspace / "test.db"
    db = TestDatabase(db_path)
    yield db
    db.close()


@pytest.fixture
def mock_sandbox():
    """Mock sandbox environment fixture."""
    sandbox = MockSandboxEnvironment()
    yield sandbox
    sandbox.cleanup()


@pytest.fixture
def populated_test_database(test_database, test_data_generator):
    """Test database populated with sample data."""
    db = test_database
    gen = test_data_generator

    # Add sample agents
    db.insert_agent("agent-1", "Test Agent 1", "manus")
    db.insert_agent("agent-2", "Test Agent 2", "browser")

    # Add sample conversations
    db.insert_conversation("conv-1", "agent-1", "user-1", "Test Conversation 1")
    db.insert_conversation("conv-2", "agent-1", "user-2", "Test Conversation 2")

    # Add sample messages
    db.insert_message("msg-1", "conv-1", "user", "Hello, can you help me?")
    db.insert_message(
        "msg-2", "conv-1", "assistant", "Of course! How can I assist you?"
    )
    db.insert_message("msg-3", "conv-1", "user", "I need to analyze some data.")

    # Add sample tool calls
    db.insert_tool_call(
        "call-1",
        "msg-2",
        "web_search",
        {"query": "data analysis techniques"},
        {"results": ["Result 1", "Result 2"]},
        True,
        1.5,
    )

    # Add sample performance metrics
    db.insert_performance_metric(
        "metric-1", "agent_response", 2.3, 1024 * 1024, 15.5, True
    )

    # Add sample security events
    db.insert_security_event(
        "event-1", "input_validation", "low", "Suspicious input detected"
    )

    yield db
