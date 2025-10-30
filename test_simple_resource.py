#!/usr/bin/env python3
"""
Simple test to check resource manager imports.
"""

try:
    import app.performance.resource_manager as rm

    print("Available classes and functions:")
    for name in dir(rm):
        if not name.startswith("_"):
            print(f"  - {name}")

    # Test basic functionality
    if hasattr(rm, "ResourceUsage"):
        usage = rm.ResourceUsage()
        print(f"\n✓ ResourceUsage created: {usage}")

    if hasattr(rm, "ResourceLimits"):
        limits = rm.ResourceLimits()
        print(f"✓ ResourceLimits created: {limits}")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
