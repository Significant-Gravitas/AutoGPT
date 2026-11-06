"""`python -m autogpt_desktop serve` starts the stack and blocks until asked to stop."""

from __future__ import annotations

import argparse

from autogpt_desktop.supervisor import serve


def main() -> None:
    parser = argparse.ArgumentParser(prog="autogpt_desktop")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="start AutoGPT and run until stdin closes")
    parser.parse_args()
    serve()


if __name__ == "__main__":
    main()
