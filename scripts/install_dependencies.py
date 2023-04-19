"""Based on config options, install optional dependencies."""
import subprocess
import sys
from pathlib import Path

import pkg_resources

from autogpt.config import Config
from autogpt.logs import logger

CFG = Config()


def install_package(package: str) -> None:
    """Install a package using pip."""
    if pkg_resources.get_distribution(package).version:
        logger.info(f"{package} is already installed.")
        return

    logger.info(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def parse_optional_requirements(optional_requirements_path: Path) -> dict:
    """Parse optional-requirements.txt file."""
    optional_requirements = {}

    # Open file and go through each line
    with open(optional_requirements_path) as f:
        for line in f:
            # Skip comments and empty lines
            if line.startswith("#") or not line.strip():
                continue
            # line format: [config_option=value] package (optionally with version: package==version)

            # Split line into config_option and package
            config_option, package = line.split(" ", 1)

            # Remove trailing comments, whitespace and newlines
            package = package.split("#", 1)[0].strip()

            # Split config_option into option name and value
            config_option = config_option[1:-1].split("=")
            if len(config_option) == 1:
                config_option.append(None)

            # Add to optional_requirements
            optional_requirements.setdefault(config_option[0], []).append(
                (config_option[1], package)
            )

    return optional_requirements


def fetch_requirements(config_option: str, optional_requirements: dict) -> list:
    """Fetch requirements for a given config option."""
    requirements = []

    # Check if config option is in optional_requirements
    if config_option not in optional_requirements:
        return requirements

    # Go through each requirement
    for value, package in optional_requirements[config_option]:
        # Check if config option exists
        if not hasattr(CFG, config_option):
            logger.warning(
                f"Config option {config_option} does not exist, skipping {package}."
            )
            continue
        # Check if config option value matches
        if value is None or getattr(CFG, config_option) == value:
            requirements.append(package)

    return requirements


def install_dependencies(opts: dict[str, tuple]) -> None:
    """Install optional dependencies based on config options."""
    # Go through each config option
    for config_option, requirements in opts.items():
        # Fetch requirements for config option
        requirements = fetch_requirements(config_option, opts)
        # Install requirements
        for requirement in requirements:
            install_package(requirement)


if __name__ == "__main__":
    opts = parse_optional_requirements("requirements-optional.txt")
    install_dependencies(opts)
