#!/usr/bin/env python
"""
Test feature flag evaluation with Supabase role.

Usage:
    poetry run python test_feature_flag.py <user_id> <flag_key>
"""

import asyncio
import sys

from autogpt_libs.feature_flag.client import is_feature_enabled


async def test_feature_flag(user_id: str, flag_key: str):
    """Test feature flag evaluation with user role from Supabase."""

    print(f"\n🚩 Testing feature flag: {flag_key}")
    print(f"👤 User ID: {user_id}\n")

    # Test with full context (fetches from Supabase)
    print("Testing with full context (Supabase + DB)...")
    result = await is_feature_enabled(
        flag_key=flag_key,
        user_id=user_id,
        default=False,
        use_user_id_only=False,  # This will fetch full context
    )
    print(f"  Result: {'✅ ENABLED' if result else '❌ DISABLED'}")

    # Test with user ID only (no Supabase fetch)
    print("\nTesting with user ID only (no context fetch)...")
    result_simple = await is_feature_enabled(
        flag_key=flag_key,
        user_id=user_id,
        default=False,
        use_user_id_only=True,  # Simple context
    )
    print(f"  Result: {'✅ ENABLED' if result_simple else '❌ DISABLED'}")

    # Compare results
    if result != result_simple:
        print("\n⚠️  Different results with/without full context!")
        print("   This suggests the role from Supabase affects the flag evaluation.")


async def main():
    if len(sys.argv) < 3:
        print("Usage: poetry run python test_feature_flag.py <user_id> <flag_key>")
        print("\nExample:")
        print("  poetry run python test_feature_flag.py YOUR_USER_ID api-keys-enabled")
        sys.exit(1)

    user_id = sys.argv[1]
    flag_key = sys.argv[2]

    await test_feature_flag(user_id, flag_key)


if __name__ == "__main__":
    asyncio.run(main())
