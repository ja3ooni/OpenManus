#!/usr/bin/env python3
"""Debug import issues step by step."""

import sys

sys.path.append(".")

try:
    print("Step 1: Importing required modules...")
    from datetime import datetime, timedelta
    from pathlib import Path
    from typing import Any, Dict, List, Optional, Union

    print("✓ Basic imports successful")

    print("Step 2: Importing cryptography...")
    from cryptography.fernet import Fernet

    print("✓ Cryptography imports successful")

    print("Step 3: Importing app modules...")
    from app.logger import define_log_level
    from app.security.models import SecurityContext, SecurityEvent, SecurityLevel

    print("✓ App imports successful")

    print("Step 4: Importing credentials module...")
    import app.security.credentials

    print("✓ Credentials module imported")

    print("Step 5: Checking available classes...")
    print("Available classes:")
    for name in dir(app.security.credentials):
        if not name.startswith("_") and name[0].isupper():
            print(f"  - {name}")

    print("Step 6: Testing direct class access...")
    try:
        from app.security.credentials import CredentialManager

        print("✓ CredentialManager imported successfully")
    except ImportError as e:
        print(f"✗ CredentialManager import failed: {e}")

    try:
        from app.security.credentials import EnvironmentCredentialProvider

        print("✓ EnvironmentCredentialProvider imported successfully")
    except ImportError as e:
        print(f"✗ EnvironmentCredentialProvider import failed: {e}")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
