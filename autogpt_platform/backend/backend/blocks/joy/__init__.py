"""
Joy Trust Network blocks for AutoGPT.

Provides trust verification for AI agent delegation. Before delegating a task
to another agent, verify their trust score meets your threshold.

Blocks:
- JoyVerifyTrustBlock: Check if agent meets minimum trust threshold
- JoyGetTrustScoreBlock: Get detailed trust profile for an agent
- JoyDiscoverAgentsBlock: Find trusted agents by capability

Learn more: https://choosejoy.com.au
"""

from .discovery import JoyDiscoverAgentsBlock
from .trust import JoyGetTrustScoreBlock, JoyVerifyTrustBlock

__all__ = [
    "JoyVerifyTrustBlock",
    "JoyGetTrustScoreBlock",
    "JoyDiscoverAgentsBlock",
]
