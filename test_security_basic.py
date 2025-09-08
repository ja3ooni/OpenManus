#!/usr/bin/env python3
"""Basic test for security functionality."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.security.manager import SecurityManager
from app.security.models import SecurityContext, SecurityPolicy
from app.security.validators import InputValidator


async def test_security_manager():
    """Test SecurityManager functionality."""
    print("Testing SecurityManager...")

    manager = SecurityManager()
    context = SecurityContext(user_id="test_user", ip_address="127.0.0.1")

    # Test 1: Clean input validation
    result = await manager.validate_input("Hello world", context)
    print(f"✓ Clean input validation: {result.status.value}")

    # Test 2: SQL injection detection
    result = await manager.validate_input("'; DROP TABLE users; --", context)
    print(
        f"✓ SQL injection detection: {result.status.value}, threats: {len(result.threats_detected)}"
    )

    # Test 3: XSS detection
    result = await manager.validate_input("<script>alert('xss')</script>", context)
    print(
        f"✓ XSS detection: {result.status.value}, threats: {len(result.threats_detected)}"
    )

    # Test 4: Rate limiting
    allowed = await manager.check_rate_limit("test_client", "test_operation", context)
    print(f"✓ Rate limit check: {allowed}")

    # Test 5: Token generation
    token = await manager.generate_secure_token()
    print(f"✓ Token generation: {len(token)} chars")

    # Test 6: Password hashing
    password_hash, salt = await manager.hash_password("test_password")
    verified = await manager.verify_password("test_password", password_hash, salt)
    print(f"✓ Password hashing and verification: {verified}")

    print("SecurityManager tests completed successfully!")


def test_input_validator():
    """Test InputValidator functionality."""
    print("\nTesting InputValidator...")

    validator = InputValidator()

    # Test 1: String validation
    result = validator.validate_string("Clean string")
    print(f"✓ Clean string validation: {result.status.value}")

    # Test 2: SQL injection detection
    result = validator.validate_string("'; DROP TABLE users; --")
    print(f"✓ SQL injection detection: {result.status.value}")

    # Test 3: Email validation
    result = validator.validate_email("user@example.com")
    print(f"✓ Valid email: {result.status.value}")

    result = validator.validate_email("invalid-email")
    print(f"✓ Invalid email: {result.status.value}")

    # Test 4: URL validation
    result = validator.validate_url("https://example.com")
    print(f"✓ Valid URL: {result.status.value}")

    # Test 5: Filename validation
    result = validator.validate_filename("document.txt")
    print(f"✓ Valid filename: {result.status.value}")

    result = validator.validate_filename("../../../etc/passwd")
    print(f"✓ Path traversal detection: {result.status.value}")

    # Test 6: HTML sanitization
    sanitized = validator.sanitize_html("<p>Clean</p><script>alert('xss')</script>")
    print(f"✓ HTML sanitization: {'<script>' not in sanitized}")

    print("InputValidator tests completed successfully!")


async def main():
    """Run all tests."""
    print("Starting security module tests...\n")

    try:
        await test_security_manager()
        test_input_validator()
        print("\n🎉 All security tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
