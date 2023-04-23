import re
import sys

import pkg_resources


def main():
    requirements_file = sys.argv[1]
    with open(requirements_file, "r") as f:
        required_packages = [str(req) for req in pkg_resources.parse_requirements(f)]

    installed_packages = {package.key.lower() for package in pkg_resources.working_set}

    missing_packages = [package for package in required_packages if package.lower().split()[0] not in installed_packages]

    if missing_packages:
        print("Missing packages:")
        print(", ".join(missing_packages))
        sys.exit(1)
    else:
        print("All packages are installed.")


if __name__ == "__main__":
    main()
