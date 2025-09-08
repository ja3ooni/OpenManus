#!/usr/bin/env python3
"""Debug import issues."""

import sys

sys.path.append(".")

try:
    print("Importing module...")
    import app.security.credentials as creds_module

    print(f"Module imported successfully: {creds_module}")

    print("Available attributes:")
    for attr in dir(creds_module):
        if not attr.startswith("_"):
            print(f"  - {attr}")

    print("\nTrying to access EnvironmentCredentialProvider...")
    if hasattr(creds_module, "EnvironmentCredentialProvider"):
        print("EnvironmentCredentialProvider found!")
        provider_class = getattr(creds_module, "EnvironmentCredentialProvider")
        print(f"Class: {provider_class}")
    else:
        print("EnvironmentCredentialProvider NOT found!")

    print("\nTrying to access CredentialManager...")
    if hasattr(creds_module, "CredentialManager"):
        print("CredentialManager found!")
        manager_class = getattr(creds_module, "CredentialManager")
        print(f"Class: {manager_class}")
    else:
        print("CredentialManager NOT found!")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
