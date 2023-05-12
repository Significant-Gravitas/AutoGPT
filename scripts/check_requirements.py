import re
import sys

import pkg_resources


def main():
    requirements_file = sys.argv[1]
    with open(requirements_file, "r") as f:
        required_packages = [
            line.strip().split("#")[0].strip() for line in f.readlines()
        ]

    installed_packages = dict(
        [(package.key, package.version) for package in pkg_resources.working_set]
    )

    pattern = (
        r"^(?P<name>[^\s<>=]+)(?:\s*[@<>=]+\s*(?P<min_version>[^\s<>=]+))?(?:\s*,\s*<(?P<max_version>["
        r"^\s<>=]+))?$"
    )

    url_pattern = r"(https?://[^\s/$.?#].[^\s]*)"

    missing_packages = []
    for package in required_packages:
        if not package:  # Skip empty lines
            continue
        package_data = re.match(pattern, package.strip())
        if not package_data:
            continue
        package_name = package_data[1]
        min_version = package_data[2] if package_data[2] else None
        max_version = package_data[3] if package_data[3] else None
        if min_version is not None and re.match(url_pattern, min_version):
            min_version = None
            max_version = None
        if (
            package_name.lower() not in installed_packages
            or (
                max_version is not None
                and pkg_resources.parse_version(
                    installed_packages[package_name.lower()]
                )
                < pkg_resources.parse_version(min_version)
            )
            or (
                max_version is not None
                and pkg_resources.parse_version(
                    installed_packages[package_name.lower()]
                )
                > pkg_resources.parse_version(max_version)
            )
        ):
            missing_packages.append(package)

    if missing_packages:
        print("Missing packages:")
        print(", ".join(missing_packages))
        sys.exit(1)
    else:
        print("All packages are installed.")


if __name__ == "__main__":
    main()
