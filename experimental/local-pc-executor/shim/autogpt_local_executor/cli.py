"""
CLI entry point — `autogpt-shim <command>`

Commands:
    auth    Run the OAuth flow to authenticate with the AutoGPT platform
    start   Start the shim daemon (foreground)
    stop    Stop a running daemon
    status  Show connection status
    revoke  Revoke OAuth tokens and disconnect
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autogpt-shim",
        description="⚠️  EXPERIMENTAL: AutoGPT Local PC Executor shim",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("auth", help="Authenticate with the AutoGPT platform (OAuth)")
    sub.add_parser("start", help="Start the shim daemon")
    sub.add_parser("stop", help="Stop a running daemon")
    sub.add_parser("status", help="Show connection status")
    sub.add_parser("revoke", help="Revoke tokens and disconnect")

    args = parser.parse_args()

    print("⚠️  WARNING: This is experimental, untested software.")
    print("   It gives the AutoGPT platform code execution access to this machine.")
    print("   Read docs/SECURITY.md before continuing.\n")

    if args.command == "auth":
        _cmd_auth()
    elif args.command == "start":
        asyncio.run(_cmd_start())
    elif args.command == "status":
        _cmd_status()
    elif args.command == "revoke":
        _cmd_revoke()
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_auth() -> None:
    from .config import ShimConfig
    from .auth import OAuthFlow, KeychainTokenStore
    config = ShimConfig()
    flow = OAuthFlow(config=config, token_store=KeychainTokenStore())
    flow.run()


async def _cmd_start() -> None:
    from .config import ShimConfig
    from .auth import KeychainTokenStore
    from .daemon import ShimDaemon
    config = ShimConfig()
    daemon = ShimDaemon(config=config, token_store=KeychainTokenStore())
    print(f"Starting shim daemon (machine_id={config.machine_id})")
    print(f"Allowed root: {config.allowed_root}")
    print("Press Ctrl+C to stop.\n")
    try:
        await daemon.run()
    except KeyboardInterrupt:
        await daemon.stop()
        print("\nShim stopped.")


def _cmd_status() -> None:
    # TODO: check pidfile / unix socket for running daemon status
    print("Status check not yet implemented.")


def _cmd_revoke() -> None:
    from .auth import KeychainTokenStore
    store = KeychainTokenStore()
    store.clear_tokens()
    print("Tokens revoked. Run `autogpt-shim auth` to re-authenticate.")
