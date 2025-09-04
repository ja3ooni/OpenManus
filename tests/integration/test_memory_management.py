"""
Integration tests for memory management and conversation persistence.

This module contains comprehensive tests for agent memory management,
conversation history persistence, context optimization, and memory limits.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.agent.manus import Manus
from app.exceptions import MemoryError, ResourceError
from app.schema import Memory, Message, Role, ToolCall, ToolResult
from tests.base import AgentTestCase, IntegrationTestCase


class TestMemoryManagement(IntegrationTestCase):
    """Tests for agent memory management functionality."""

    def setup_method(self):
        """Set up memory management test environment."""
        super().setup_method()
        self.agent = Manus()

    @pytest.mark.asyncio
    async def test_basic_memory_operations(self):
        """Test basic memory operations."""
        # Test adding messages
        self.agent.update_memory("user", "Hello, agent!")
        self.agent.update_memory("assistant", "Hello! How can I help you?")
        self.agent.update_memory("user", "What's the weather like?")

        # Verify messages are stored
        assert len(self.agent.memory.messages) == 3
        assert self.agent.memory.messages[0].role == Role.USER
        assert self.agent.memory.messages[1].role == Role.ASSISTANT
        assert self.agent.memory.messages[2].role == Role.USER

        # Test message content
        assert "Hello, agent!" in self.agent.memory.messages[0].content
        assert "Hello! How can I help you?" in self.agent.memory.messages[1].content
        assert "What's the weather like?" in self.agent.memory.messages[2].content

    @pytest.mark.asyncio
    async def test_memory_limit_enforcement(self):
        """Test memory limit enforcement."""
        # Set a small memory limit
        self.agent.memory.max_messages = 5

        # Add more messages than the limit
        for i in range(10):
            self.agent.update_memory("user", f"Message {i}")

        # Verify only the most recent messages are kept
        assert len(self.agent.memory.messages) == 5

        # Verify the messages are the most recent ones
        message_contents = [msg.content for msg in self.agent.memory.messages]
        assert "Message 5" in message_contents[0]
        assert "Message 9" in message_contents[-1]

    @pytest.mark.asyncio
    async def test_memory_with_tool_calls(self):
        """Test memory management with tool calls."""
        # Create a tool call
        tool_call = ToolCall(
            name="test_tool", arguments={"param": "value"}, call_id="call_123"
        )

        # Add assistant message with tool call
        assistant_msg = Message.assistant_message(
            content="I'll use a tool to help you.", tool_calls=[tool_call]
        )
        self.agent.memory.add_message(assistant_msg)

        # Add tool result message
        self.agent.update_memory(
            "tool", "Tool execution successful", tool_call_id="call_123"
        )

        # Verify messages are stored correctly
        assert len(self.agent.memory.messages) == 2
        assert self.agent.memory.messages[0].tool_calls is not None
        assert len(self.agent.memory.messages[0].tool_calls) == 1
        assert self.agent.memory.messages[1].role == Role.TOOL

    @pytest.mark.asyncio
    async def test_memory_with_images(self):
        """Test memory management with base64 images."""
        # Mock base64 image data
        mock_image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

        # Add message with image
        self.agent.update_memory(
            "user", "Here's an image for you to analyze", base64_image=mock_image_data
        )

        # Verify image is stored
        assert len(self.agent.memory.messages) == 1
        assert self.agent.memory.messages[0].base64_image == mock_image_data

    @pytest.mark.asyncio
    async def test_memory_clear_operation(self):
        """Test memory clear operation."""
        # Add some messages
        for i in range(5):
            self.agent.update_memory("user", f"Message {i}")

        assert len(self.agent.memory.messages) == 5

        # Clear memory
        self.agent.memory.clear()

        # Verify memory is empty
        assert len(self.agent.memory.messages) == 0

    @pytest.mark.asyncio
    async def test_recent_messages_retrieval(self):
        """Test retrieval of recent messages."""
        # Add multiple messages
        for i in range(10):
            self.agent.update_memory("user", f"Message {i}")

        # Get recent messages
        recent_3 = self.agent.memory.get_recent_messages(3)
        recent_5 = self.agent.memory.get_recent_messages(5)

        # Verify correct number of messages
        assert len(recent_3) == 3
        assert len(recent_5) == 5

        # Verify they are the most recent
        assert "Message 7" in recent_3[0].content
        assert "Message 9" in recent_3[-1].content
        assert "Message 5" in recent_5[0].content
        assert "Message 9" in recent_5[-1].content

    @pytest.mark.asyncio
    async def test_memory_serialization(self):
        """Test memory serialization to dict format."""
        # Add various types of messages
        self.agent.update_memory("user", "Hello")
        self.agent.update_memory("assistant", "Hi there!")

        tool_call = ToolCall(name="test_tool", arguments={}, call_id="call_1")
        assistant_msg = Message.assistant_message("Using tool", tool_calls=[tool_call])
        self.agent.memory.add_message(assistant_msg)

        # Serialize to dict
        dict_messages = self.agent.memory.to_dict_list()

        # Verify serialization
        assert len(dict_messages) == 3
        assert all(isinstance(msg, dict) for msg in dict_messages)
        assert dict_messages[0]["role"] == "user"
        assert dict_messages[1]["role"] == "assistant"
        assert dict_messages[2]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_memory_performance_with_large_history(self):
        """Test memory performance with large conversation history."""
        # Add a large number of messages
        message_count = 1000

        start_time = asyncio.get_event_loop().time()

        for i in range(message_count):
            self.agent.update_memory(
                "user", f"Message {i} with some content to make it realistic"
            )
            if i % 2 == 0:
                self.agent.update_memory(
                    "assistant", f"Response {i} with detailed information"
                )

        end_time = asyncio.get_event_loop().time()

        # Verify performance is reasonable (should complete in under 1 second)
        execution_time = end_time - start_time
        assert (
            execution_time < 1.0
        ), f"Memory operations took too long: {execution_time}s"

        # Verify memory limit is enforced
        assert len(self.agent.memory.messages) <= self.agent.memory.max_messages


class TestConversationPersistence(IntegrationTestCase):
    """Tests for conversation persistence functionality."""

    def setup_method(self):
        """Set up conversation persistence test environment."""
        super().setup_method()
        self.agent = Manus()
        self.persistence_file = self.workspace / "conversation.json"

    async def save_conversation_to_file(self, filename: Path):
        """Helper method to save conversation to file."""
        conversation_data = {
            "messages": self.agent.memory.to_dict_list(),
            "metadata": {
                "agent_name": self.agent.name,
                "max_messages": self.agent.memory.max_messages,
                "message_count": len(self.agent.memory.messages),
            },
        }

        with open(filename, "w") as f:
            json.dump(conversation_data, f, indent=2, default=str)

    async def load_conversation_from_file(self, filename: Path):
        """Helper method to load conversation from file."""
        with open(filename, "r") as f:
            conversation_data = json.load(f)

        # Reconstruct messages
        self.agent.memory.clear()
        for msg_data in conversation_data["messages"]:
            if msg_data["role"] == "user":
                self.agent.update_memory("user", msg_data["content"])
            elif msg_data["role"] == "assistant":
                self.agent.update_memory("assistant", msg_data["content"])
            elif msg_data["role"] == "tool":
                self.agent.update_memory("tool", msg_data["content"])

    @pytest.mark.asyncio
    async def test_conversation_save_and_load(self):
        """Test saving and loading conversation history."""
        # Create a conversation
        conversation = [
            ("user", "Hello, I need help with a task"),
            ("assistant", "I'd be happy to help! What do you need assistance with?"),
            ("user", "I want to analyze some data"),
            (
                "assistant",
                "Great! I can help you analyze data. What type of data do you have?",
            ),
        ]

        for role, content in conversation:
            self.agent.update_memory(role, content)

        original_message_count = len(self.agent.memory.messages)
        original_messages = [msg.content for msg in self.agent.memory.messages]

        # Save conversation
        await self.save_conversation_to_file(self.persistence_file)

        # Verify file was created
        assert self.persistence_file.exists()

        # Clear memory and reload
        self.agent.memory.clear()
        assert len(self.agent.memory.messages) == 0

        await self.load_conversation_from_file(self.persistence_file)

        # Verify conversation was restored
        assert len(self.agent.memory.messages) == original_message_count
        restored_messages = [msg.content for msg in self.agent.memory.messages]
        assert restored_messages == original_messages

    @pytest.mark.asyncio
    async def test_conversation_persistence_with_tool_calls(self):
        """Test persistence of conversations with tool calls."""
        # Add messages with tool calls
        self.agent.update_memory("user", "Search for information about Python")

        tool_call = ToolCall(
            name="web_search",
            arguments={"query": "Python programming language"},
            call_id="search_123",
        )

        assistant_msg = Message.assistant_message(
            "I'll search for information about Python for you.", tool_calls=[tool_call]
        )
        self.agent.memory.add_message(assistant_msg)

        self.agent.update_memory(
            "tool",
            "Found comprehensive information about Python programming language.",
            tool_call_id="search_123",
        )

        # Save and reload
        await self.save_conversation_to_file(self.persistence_file)
        original_count = len(self.agent.memory.messages)

        self.agent.memory.clear()
        await self.load_conversation_from_file(self.persistence_file)

        # Verify tool call information is preserved
        assert len(self.agent.memory.messages) == original_count

        # Find the assistant message with tool call
        assistant_messages = [
            msg for msg in self.agent.memory.messages if msg.role == Role.ASSISTANT
        ]
        tool_messages = [
            msg for msg in self.agent.memory.messages if msg.role == Role.TOOL
        ]

        assert len(assistant_messages) >= 1
        assert len(tool_messages) >= 1

    @pytest.mark.asyncio
    async def test_conversation_versioning(self):
        """Test conversation versioning and branching."""
        # Create base conversation
        base_conversation = [
            ("user", "Hello"),
            ("assistant", "Hi! How can I help you?"),
            ("user", "I have a question about programming"),
        ]

        for role, content in base_conversation:
            self.agent.update_memory(role, content)

        # Save version 1
        version1_file = self.workspace / "conversation_v1.json"
        await self.save_conversation_to_file(version1_file)

        # Continue conversation (branch A)
        self.agent.update_memory(
            "assistant", "I'd be happy to help with programming questions!"
        )
        self.agent.update_memory(
            "user", "What's the difference between Python and JavaScript?"
        )

        # Save version 2 (branch A)
        version2a_file = self.workspace / "conversation_v2a.json"
        await self.save_conversation_to_file(version2a_file)

        # Load version 1 and create branch B
        await self.load_conversation_from_file(version1_file)
        self.agent.update_memory(
            "assistant", "What specific programming topic interests you?"
        )
        self.agent.update_memory("user", "I want to learn about data structures")

        # Save version 2 (branch B)
        version2b_file = self.workspace / "conversation_v2b.json"
        await self.save_conversation_to_file(version2b_file)

        # Verify both branches exist and are different
        assert version2a_file.exists()
        assert version2b_file.exists()

        # Load and compare branches
        await self.load_conversation_from_file(version2a_file)
        branch_a_messages = [msg.content for msg in self.agent.memory.messages]

        await self.load_conversation_from_file(version2b_file)
        branch_b_messages = [msg.content for msg in self.agent.memory.messages]

        # Verify branches are different
        assert branch_a_messages != branch_b_messages
        assert "JavaScript" in str(branch_a_messages)
        assert "data structures" in str(branch_b_messages)

    @pytest.mark.asyncio
    async def test_conversation_metadata_persistence(self):
        """Test persistence of conversation metadata."""
        # Set up conversation with metadata
        self.agent.memory.max_messages = 50

        for i in range(10):
            self.agent.update_memory("user", f"Message {i}")
            self.agent.update_memory("assistant", f"Response {i}")

        # Save with metadata
        await self.save_conversation_to_file(self.persistence_file)

        # Verify metadata in file
        with open(self.persistence_file, "r") as f:
            data = json.load(f)

        assert "metadata" in data
        assert data["metadata"]["agent_name"] == "Manus"
        assert data["metadata"]["max_messages"] == 50
        assert data["metadata"]["message_count"] == 20

    @pytest.mark.asyncio
    async def test_large_conversation_persistence(self):
        """Test persistence of large conversations."""
        # Create a large conversation
        message_count = 500

        for i in range(message_count):
            self.agent.update_memory(
                "user", f"User message {i} with detailed content about various topics"
            )
            self.agent.update_memory(
                "assistant",
                f"Assistant response {i} with comprehensive information and analysis",
            )

        # Save large conversation
        start_time = asyncio.get_event_loop().time()
        await self.save_conversation_to_file(self.persistence_file)
        save_time = asyncio.get_event_loop().time() - start_time

        # Verify file size is reasonable
        file_size = self.persistence_file.stat().st_size
        assert file_size > 0

        # Load large conversation
        self.agent.memory.clear()
        start_time = asyncio.get_event_loop().time()
        await self.load_conversation_from_file(self.persistence_file)
        load_time = asyncio.get_event_loop().time() - start_time

        # Verify performance is acceptable
        assert save_time < 5.0, f"Save took too long: {save_time}s"
        assert load_time < 5.0, f"Load took too long: {load_time}s"

        # Verify data integrity
        assert len(self.agent.memory.messages) <= self.agent.memory.max_messages


class TestMemoryOptimization(IntegrationTestCase):
    """Tests for memory optimization and context management."""

    def setup_method(self):
        """Set up memory optimization test environment."""
        super().setup_method()
        self.agent = Manus()

    @pytest.mark.asyncio
    async def test_context_window_optimization(self):
        """Test context window optimization for LLM calls."""
        # Create a conversation that exceeds typical context windows
        for i in range(200):
            long_content = (
                f"This is a very long message {i} " * 50
            )  # ~2500 chars per message
            self.agent.update_memory("user", long_content)
            self.agent.update_memory("assistant", f"Response to message {i}")

        # Test getting optimized context
        recent_messages = self.agent.memory.get_recent_messages(20)

        # Verify we get a reasonable number of recent messages
        assert len(recent_messages) == 20

        # Verify they are the most recent
        assert "message 199" in recent_messages[-1].content.lower()

    @pytest.mark.asyncio
    async def test_memory_compression_simulation(self):
        """Test memory compression simulation."""
        # Add many messages
        for i in range(100):
            self.agent.update_memory(
                "user", f"Question {i}: What is the capital of country {i}?"
            )
            self.agent.update_memory(
                "assistant", f"The capital of country {i} is City {i}."
            )

        original_count = len(self.agent.memory.messages)

        # Simulate compression by keeping only every 5th message pair
        compressed_messages = []
        for i in range(0, len(self.agent.memory.messages), 10):  # Every 5th pair
            if i < len(self.agent.memory.messages):
                compressed_messages.append(self.agent.memory.messages[i])
            if i + 1 < len(self.agent.memory.messages):
                compressed_messages.append(self.agent.memory.messages[i + 1])

        # Apply compression
        self.agent.memory.messages = compressed_messages

        # Verify compression worked
        assert len(self.agent.memory.messages) < original_count
        assert len(self.agent.memory.messages) > 0

    @pytest.mark.asyncio
    async def test_memory_priority_based_retention(self):
        """Test priority-based message retention."""
        # Add messages with different priorities (simulated)
        important_messages = []

        for i in range(50):
            if i % 10 == 0:  # Every 10th message is "important"
                content = f"IMPORTANT: Critical information {i}"
                important_messages.append(content)
            else:
                content = f"Regular message {i}"

            self.agent.update_memory("user", content)
            self.agent.update_memory("assistant", f"Response {i}")

        # Simulate priority-based retention
        # Keep all messages marked as important and recent messages
        filtered_messages = []

        for msg in self.agent.memory.messages:
            if "IMPORTANT" in msg.content or len(filtered_messages) < 20:
                filtered_messages.append(msg)

        # Verify important messages are retained
        important_count = sum(
            1 for msg in filtered_messages if "IMPORTANT" in msg.content
        )
        assert important_count > 0

    @pytest.mark.asyncio
    async def test_memory_deduplication(self):
        """Test memory deduplication for repeated content."""
        # Add some duplicate messages
        self.agent.update_memory("user", "Hello")
        self.agent.update_memory("assistant", "Hi there!")
        self.agent.update_memory("user", "Hello")  # Duplicate
        self.agent.update_memory("assistant", "Hi there!")  # Duplicate
        self.agent.update_memory("user", "How are you?")
        self.agent.update_memory("assistant", "I'm doing well!")

        original_count = len(self.agent.memory.messages)

        # Simulate deduplication
        seen_content = set()
        deduplicated_messages = []

        for msg in self.agent.memory.messages:
            content_key = f"{msg.role.value}:{msg.content}"
            if content_key not in seen_content:
                seen_content.add(content_key)
                deduplicated_messages.append(msg)

        # Verify deduplication worked
        assert len(deduplicated_messages) < original_count
        assert len(deduplicated_messages) == 4  # Should have 4 unique messages


class TestMemoryErrorHandling(IntegrationTestCase):
    """Tests for memory error handling and recovery."""

    def setup_method(self):
        """Set up memory error handling test environment."""
        super().setup_method()
        self.agent = Manus()

    @pytest.mark.asyncio
    async def test_memory_limit_exceeded_handling(self):
        """Test handling when memory limits are exceeded."""
        # Set very small memory limit
        self.agent.memory.max_messages = 2

        # Add messages beyond limit
        messages_to_add = [
            ("user", "Message 1"),
            ("assistant", "Response 1"),
            ("user", "Message 2"),
            ("assistant", "Response 2"),
            ("user", "Message 3"),  # This should trigger limit
        ]

        for role, content in messages_to_add:
            self.agent.update_memory(role, content)

        # Verify limit is enforced
        assert len(self.agent.memory.messages) == 2

        # Verify most recent messages are kept
        assert "Message 3" in self.agent.memory.messages[-1].content

    @pytest.mark.asyncio
    async def test_invalid_message_handling(self):
        """Test handling of invalid messages."""
        # Test invalid role
        with pytest.raises(ValueError):
            self.agent.update_memory("invalid_role", "Some content")

        # Test empty content (should be allowed)
        self.agent.update_memory("user", "")
        assert len(self.agent.memory.messages) == 1

    @pytest.mark.asyncio
    async def test_memory_corruption_recovery(self):
        """Test recovery from memory corruption."""
        # Add valid messages
        for i in range(5):
            self.agent.update_memory("user", f"Message {i}")

        # Simulate corruption by directly modifying memory
        self.agent.memory.messages[2] = None  # Corrupt one message

        # Test recovery by filtering out corrupted messages
        valid_messages = [msg for msg in self.agent.memory.messages if msg is not None]
        self.agent.memory.messages = valid_messages

        # Verify recovery
        assert len(self.agent.memory.messages) == 4
        assert all(msg is not None for msg in self.agent.memory.messages)

    @pytest.mark.asyncio
    async def test_concurrent_memory_access(self):
        """Test concurrent access to memory."""

        async def add_messages(start_idx: int, count: int):
            for i in range(count):
                self.agent.update_memory("user", f"Concurrent message {start_idx + i}")

        # Run concurrent memory operations
        tasks = [
            add_messages(0, 10),
            add_messages(10, 10),
            add_messages(20, 10),
        ]

        await asyncio.gather(*tasks)

        # Verify all messages were added (subject to memory limits)
        message_count = len(self.agent.memory.messages)
        assert message_count > 0
        assert message_count <= self.agent.memory.max_messages


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
