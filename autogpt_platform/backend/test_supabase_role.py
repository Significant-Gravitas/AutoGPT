#!/usr/bin/env python
"""
Manual test script to verify Supabase role fetching works correctly.

Usage:
    poetry run python test_supabase_role.py <user_id>
"""

import asyncio
import json
import sys

from autogpt_libs.feature_flag.client import _fetch_user_context_data

from backend.data.supabase_auth import get_user_auth_data_from_supabase
from backend.server.integrations.utils import get_supabase


async def test_supabase_role(user_id: str):
    """Test fetching role from Supabase for a real user."""

    print(f"\n🔍 Testing Supabase role fetch for user: {user_id}\n")

    # 1. Test direct Supabase auth fetch
    print("1️⃣  Fetching auth data directly from Supabase...")
    try:
        auth_data = await get_user_auth_data_from_supabase(user_id)
        if auth_data:
            print(f"   ✅ Auth data: {json.dumps(auth_data, indent=2)}")
        else:
            print("   ⚠️  No auth data returned (user might not exist in Supabase)")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # 2. Test the full LaunchDarkly context build
    print("\n2️⃣  Building full LaunchDarkly context...")
    try:
        context = await _fetch_user_context_data(user_id)
        print(f"   ✅ LaunchDarkly context: {json.dumps(context, indent=2)}")

        # Verify structure
        assert "kind" in context, "Missing 'kind' field"
        assert "key" in context, "Missing 'key' field"
        assert "email" in context, "Missing 'email' field"
        assert "custom" in context, "Missing 'custom' field"
        assert "role" in context["custom"], "Missing 'role' in custom field"

        print("\n   📋 Summary:")
        print(f"      - User ID: {context['key']}")
        print(f"      - Email: {context['email']}")
        print(f"      - Role: {context['custom']['role']}")
        if "age" in context["custom"]:
            print(f"      - Account age: {context['custom']['age']} days")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    # 3. Direct Supabase API check (for comparison)
    print("\n3️⃣  Direct Supabase API check...")
    try:
        supabase = get_supabase()
        response = supabase.auth.admin.get_user_by_id(user_id)
        if response and response.user:
            user = response.user
            print("   ✅ Supabase user found:")
            print(f"      - Email: {user.email if hasattr(user, 'email') else 'N/A'}")
            print(f"      - Role: {user.role if hasattr(user, 'role') else 'N/A'}")
            if hasattr(user, "app_metadata") and user.app_metadata:
                print(
                    f"      - App metadata: {json.dumps(user.app_metadata, indent=8)}"
                )
            if hasattr(user, "user_metadata") and user.user_metadata:
                print(
                    f"      - User metadata: {json.dumps(user.user_metadata, indent=8)}"
                )
        else:
            print("   ⚠️  User not found in Supabase")
    except Exception as e:
        print(f"   ❌ Error accessing Supabase: {e}")


async def main():
    if len(sys.argv) < 2:
        print("Usage: poetry run python test_supabase_role.py <user_id>")
        print("\nExample user IDs you can try:")
        print("  - Your own user ID from Supabase")
        print("  - 3e53486c-cf57-477e-ba2a-cb02dc828e1a (DEFAULT_USER_ID)")
        sys.exit(1)

    user_id = sys.argv[1]
    await test_supabase_role(user_id)


if __name__ == "__main__":
    asyncio.run(main())
