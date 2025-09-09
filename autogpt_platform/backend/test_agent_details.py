#!/usr/bin/env python3
"""
Test script to verify get_agent_details function works correctly
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.server.v2.chat.tools.get_agent_details import GetAgentDetailsTool
from backend.util.service import get_service_client


async def test_get_agent_details():
    print("Testing get_agent_details function...")

    # Initialize the tool
    tool = GetAgentDetailsTool()

    # Test with a marketplace agent
    agent_id = "autogpt-store/slug-h"
    print(f"\n🔍 Testing with agent ID: {agent_id}")

    try:
        # Execute the tool
        result = await tool.execute(
            agent_id=agent_id,
            user_id="anon_test_user_123",  # Anonymous user for testing
            user_email=None,
        )

        print("\n✅ Successfully got agent details!")
        print(f"Response type: {result.type}")

        if hasattr(result, "details"):
            print("\nAgent Details:")
            print(f"  Name: {result.details.name}")
            print(f"  Description: {result.details.description}")
            print(f"  Version: {result.details.version}")
            print(f"  Credentials: {result.details.credentials}")
            print(f"    - Type: {type(result.details.credentials)}")
            print(
                f"    - Count: {len(result.details.credentials) if result.details.credentials else 0}"
            )
            if result.details.credentials:
                for i, cred in enumerate(result.details.credentials):
                    print(f"    - Credential {i+1}: {cred}")
        else:
            print(f"Result: {result}")
            if hasattr(result, "message"):
                print(f"Message: {result.message}")
            if hasattr(result, "required_credentials"):
                print(f"Required credentials: {result.required_credentials}")

        return True

    except Exception as e:
        print(f"\n❌ Error getting agent details: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    # Connect to database
    prisma_client = get_service_client()
    await prisma_client.connect()

    try:
        success = await test_get_agent_details()
        if success:
            print("\n✅ Test passed!")
            sys.exit(0)
        else:
            print("\n❌ Test failed!")
            sys.exit(1)
    finally:
        await prisma_client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
