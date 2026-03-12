#!/usr/bin/env python3
"""
Change Navigator CLI entry point.

Usage:
    python run_change_navigator.py
    python run_change_navigator.py --config custom_settings.yaml
    python run_change_navigator.py --coachee "Jane Doe" --week 5
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a Change Navigator weekly check-in session."
    )
    parser.add_argument(
        "--config",
        default="change_navigator_settings.yaml",
        help="Path to the YAML settings file (default: change_navigator_settings.yaml)",
    )
    parser.add_argument("--coachee", help="Override coachee_name from settings")
    parser.add_argument("--week", type=int, help="Override week_number from settings")
    return parser.parse_args()


def main():
    args = parse_args()

    from autogpt.change_navigator.agent import ChangeNavigatorAgent, ChangeNavigatorConfig

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[Error] Settings file not found: {config_path}")
        sys.exit(1)

    config = ChangeNavigatorConfig.from_yaml(config_path)

    # CLI overrides
    if args.coachee:
        config.coachee_name = args.coachee
    if args.week is not None:
        config.week_number = args.week

    agent = ChangeNavigatorAgent(config)
    entry = agent.run_checkin()

    if entry and entry.status.value == "approved":
        print("\n[Check-in complete. Journal approved and saved.]")
        sys.exit(0)
    else:
        print("\n[Session ended without approval.]")
        sys.exit(1)


if __name__ == "__main__":
    main()
