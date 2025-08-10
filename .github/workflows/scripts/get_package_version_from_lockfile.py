#!/usr/bin/env python3
import sys

if sys.version_info < (3, 11):
    print("Python version 3.11 or higher required")
    sys.exit(1)

import tomllib


def get_package_version(package_name: str, lockfile_path: str) -> str | None:
    """Extract package version from poetry.lock file."""
    try:
        if lockfile_path == "-":
            data = tomllib.load(sys.stdin.buffer)
        else:
            with open(lockfile_path, "rb") as f:
                data = tomllib.load(f)
    except FileNotFoundError:
        print(f"Error: File '{lockfile_path}' not found", file=sys.stderr)
        sys.exit(1)
    except tomllib.TOMLDecodeError as e:
        print(f"Error parsing TOML file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    # Look for the package in the packages list
    packages = data.get("package", [])
    for package in packages:
        if package.get("name", "").lower() == package_name.lower():
            return package.get("version")

    return None


def main():
    if len(sys.argv) not in (2, 3):
        print(
            "Usages: python get_package_version_from_lockfile.py <package name> [poetry.lock path]\n"
            "        cat poetry.lock | python get_package_version_from_lockfile.py <package name> -",
            file=sys.stderr,
        )
        sys.exit(1)

    package_name = sys.argv[1]
    lockfile_path = sys.argv[2] if len(sys.argv) == 3 else "poetry.lock"

    version = get_package_version(package_name, lockfile_path)

    if version:
        print(version)
    else:
        print(f"Package '{package_name}' not found in {lockfile_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
