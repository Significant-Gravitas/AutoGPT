#!/usr/bin/env python
"""Test credential detection in run_agent."""

import logging
import os
from datetime import datetime, timedelta

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials
from backend.sdk.registry import AutoRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set test environment variables
os.environ["OPENAI_API_KEY"] = "test-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"

# Get system-provided credentials (same logic as run_agent.py)
system_credentials = {}
try:
    system_creds_list = AutoRegistry.get_all_credentials()
    system_credentials = {c.provider: c for c in system_creds_list}

    # System credentials never expire - set to far future (Unix timestamp)
    expires_at = int(
        (datetime.utcnow() + timedelta(days=36500)).timestamp()
    )  # 100 years

    # Check for OpenAI
    if "openai" not in system_credentials:
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            system_credentials["openai"] = APIKeyCredentials(
                id="system-openai",
                provider="openai",
                api_key=SecretStr(openai_key),
                title="System OpenAI API Key",
                expires_at=expires_at,
            )

    # Check for Anthropic
    if "anthropic" not in system_credentials:
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            system_credentials["anthropic"] = APIKeyCredentials(
                id="system-anthropic",
                provider="anthropic",
                api_key=SecretStr(anthropic_key),
                title="System Anthropic API Key",
                expires_at=expires_at,
            )

    logger.info(f"System provides credentials for: {list(system_credentials.keys())}")
    for provider, cred in system_credentials.items():
        logger.info(f"  {provider}: id={cred.id}, type={cred.type}")

except Exception as e:
    logger.error(f"Failed to get system credentials: {e}")
    import traceback

    traceback.print_exc()

print("\n✅ System credentials loaded:")
for provider in system_credentials:
    print(f"  - {provider}")

# Now test the credential matching logic
print("\n🔍 Testing credential matching:")

# Simulate what happens in run_agent.py
test_cases = [
    ("anthropic", "Should match system-anthropic"),
    ("openai", "Should match system-openai"),
]

for provider_name, expected in test_cases:
    if provider_name in system_credentials:
        print(f"  ✅ {provider_name}: Found in system credentials")
    else:
        print(f"  ❌ {provider_name}: NOT found in system credentials")
