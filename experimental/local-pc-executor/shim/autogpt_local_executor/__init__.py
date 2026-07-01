"""
autogpt_local_executor — Local PC Shim for AutoGPT

⚠️  EXPERIMENTAL / DANGEROUS / UNTESTED ⚠️

This package implements a daemon that connects your local machine to the AutoGPT
hosted platform, allowing AutoGPT to execute commands and access files on your PC.

DO NOT USE IN PRODUCTION. DO NOT RUN AS ROOT. READ THE SECURITY DOCS FIRST.

Architecture overview:
    The shim maintains a persistent outbound WebSocket connection to the AutoGPT
    platform. The platform sends JSON messages (see docs/PROTOCOL.md) and the shim
    executes them locally, returning results over the same connection.

    The shim is intentionally thin — it executes what the platform tells it to,
    within the bounds of user-configured permissions. All intelligence stays in
    the cloud; the shim is just an execution bridge.
"""

__version__ = "0.0.1-experimental"
__all__ = ["ShimDaemon", "ShimConfig"]
