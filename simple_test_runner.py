#!/usr/bin/env python3
"""
Simple test runner to verify test structure works.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_basic_imports():
    """Test that we can import our test modules."""
    try:
        # Test basic test framework imports
        from tests.base import IntegrationTestCase, UnitTestCase

        print("✓ Test base classes imported successfully")

        # Test schema imports (core functionality)
        from app.schema import Memory, Message, Role

        print("✓ Schema classes imported successfully")

        # Test exceptions
        from app.exceptions import OpenManusError

        print("✓ Exception classes imported successfully")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    try:
        # Test Memory class
        from app.schema import Memory, Message, Role

        memory = Memory()

        user_msg = Message.user_message("Hello")
        memory.add_message(user_msg)

        assert len(memory.messages) == 1
        assert memory.messages[0].role == Role.USER
        print("✓ Memory functionality works")

        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def main():
    """Run basic tests to verify setup."""
    print("Running basic test verification...")

    import_success = test_basic_imports()
    if not import_success:
        sys.exit(1)

    functionality_success = test_basic_functionality()
    if not functionality_success:
        sys.exit(1)

    print("\n✓ All basic tests passed! Test environment is ready.")
    print("\nTest files created:")
    print("- tests/unit/test_agent_manus.py (Unit tests for Manus agent)")
    print(
        "- tests/integration/test_tool_execution.py (Tool execution integration tests)"
    )
    print("- tests/integration/test_mcp_connectivity.py (MCP connectivity tests)")
    print("- tests/integration/test_memory_management.py (Memory management tests)")

    print("\nTo run specific tests:")
    print("python -m pytest tests/unit/ -v --tb=short")
    print("python -m pytest tests/integration/ -v --tb=short")


if __name__ == "__main__":
    main()
