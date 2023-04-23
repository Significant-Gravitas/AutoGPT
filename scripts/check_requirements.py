import re
import sys

import pkg_resources

from autogpt.logs import logger


def main():
    requirements_file = sys.argv[1]
    with open(requirements_file, "r") as f:
        required_packages = [
            line.strip().split("#")[0].strip() for line in f.readlines()
        ]

    installed_packages = [package.key for package in pkg_resources.working_set]

    missing_packages = []
    for package in required_packages:
        if not package:  # Skip empty lines
            continue
        package_name = re.split("[<>=@ ]+", package.strip())[0]
        if package_name.lower() not in installed_packages:
            missing_packages.append(package_name)

    if missing_packages:
        logger.warn("Missing packages:")
        logger.info(", ".join(missing_packages))
        sys.exit(1)
    else:
        logger.info("All packages are installed.")


if __name__ == "__main__":
    main()
