#!/usr/bin/env python
"""Test system credentials functionality."""

import asyncio
import os

from backend.executor.utils import get_system_credentials


async def main():
    print("Testing system credentials...")

    # Set test environment variables
    os.environ["OPENAI_API_KEY"] = "test-openai-key"
    os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"

    # Get system credentials
    system_creds = get_system_credentials()

    print(f"\nFound {len(system_creds)} system credentials:")
    for cred_id, cred in system_creds.items():
        print(f"  - {cred_id}: {cred.provider} ({cred.type})")

    # Check if system-openai is available
    if "system-openai" in system_creds:
        print("\n✅ system-openai credential found")
        openai_cred = system_creds["system-openai"]
        print(f"   Provider: {openai_cred.provider}")
        print(f"   Type: {openai_cred.type}")
        print(f"   ID: {openai_cred.id}")
    else:
        print("\n❌ system-openai credential NOT found")

    # Check if system-anthropic is available
    if "system-anthropic" in system_creds:
        print("\n✅ system-anthropic credential found")
        anthropic_cred = system_creds["system-anthropic"]
        print(f"   Provider: {anthropic_cred.provider}")
        print(f"   Type: {anthropic_cred.type}")
        print(f"   ID: {anthropic_cred.id}")
    else:
        print("\n❌ system-anthropic credential NOT found")


if __name__ == "__main__":
    asyncio.run(main())
