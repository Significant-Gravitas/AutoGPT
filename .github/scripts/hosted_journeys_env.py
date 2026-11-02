"""Print the backend .env lines that put a compose stack in hosted configuration.

The journeys leg of platform-fullstack-ci appends these to backend/.env. Flag
values come from the same file the Playwright journeys hand the frontend, so
both sides always agree.
"""

import argparse
import json
from pathlib import Path

DEFAULT_FLAGS_FILE = (
    Path(__file__).resolve().parents[2]
    / "autogpt_platform/frontend/src/playwright/journeys/hosted-flags.json"
)

# Settings the hosted product runs with that the compose defaults leave off.
# EXECUTION_COST_COUNT_THRESHOLD is NOT a hosted value (hosted bills one credit
# per 100 node executions): billing every node makes a run's cost visible, so
# journeys can assert it exactly.
HOSTED_SETTINGS = {
    "ENABLE_CREDIT": "true",
    "EXECUTION_COST_COUNT_THRESHOLD": "1",
}


def flag_env_name(flag_key: str) -> str:
    return "FORCE_FLAG_" + flag_key.upper().replace("-", "_")


def hosted_env_lines(flags: dict[str, object]) -> list[str]:
    lines = [f"{name}={value}" for name, value in HOSTED_SETTINGS.items()]
    for key, value in flags.items():
        if not isinstance(value, bool):
            raise ValueError(f"hosted flag {key!r} must be true or false")
        lines.append(f"{flag_env_name(key)}={'true' if value else 'false'}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flags-file", type=Path, default=DEFAULT_FLAGS_FILE)
    args = parser.parse_args()
    flags = json.loads(args.flags_file.read_text(encoding="utf-8"))
    print("\n".join(hosted_env_lines(flags)))


if __name__ == "__main__":
    main()
