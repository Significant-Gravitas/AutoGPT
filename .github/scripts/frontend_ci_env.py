import argparse
import re
from pathlib import Path

import yaml

RUNTIME_SETTINGS = ("BETTER_AUTH_SECRET", "DATABASE_URL")


def prepare_build_environment(compose_path: Path, env_path: Path) -> None:
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    runtime = compose["services"]["frontend"]["environment"]
    for name in RUNTIME_SETTINGS:
        if not isinstance(runtime.get(name), str) or not runtime[name].strip():
            raise ValueError(f"Resolved frontend runtime is missing {name}")
    original = env_path.read_text(encoding="utf-8")
    pattern = r"^(" + "|".join(RUNTIME_SETTINGS) + r")=.*$"
    names = re.findall(pattern, original, re.MULTILINE)
    if sorted(names) != sorted(RUNTIME_SETTINGS):
        raise ValueError(
            "Generated frontend environment has missing or repeated settings"
        )
    sanitized = re.sub(pattern, r"\1=", original, flags=re.MULTILINE)
    env_path.write_text(sanitized, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose", required=True, type=Path)
    parser.add_argument("--env", required=True, type=Path)
    args = parser.parse_args()
    prepare_build_environment(args.compose, args.env)


if __name__ == "__main__":
    main()
