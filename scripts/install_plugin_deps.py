import os
import subprocess
import sys
import zipfile
from glob import glob
from pathlib import Path

from autogpt.logs import logger


def install_plugin_dependencies():
    """
    Installs dependencies for all plugins in the plugins dir.

    Args:
        None

    Returns:
        None
    """
    plugins_dir = Path(os.getenv("PLUGINS_DIR", "plugins"))

    logger.debug(f"Checking for dependencies in zipped plugins...")

    # Install zip-based plugins
    for plugin_archive in plugins_dir.glob("*.zip"):
        logger.debug(f"Checking for requirements in '{plugin_archive}'...")
        with zipfile.ZipFile(str(plugin_archive), "r") as zfile:
            if not zfile.namelist():
                continue

            # Assume the first entry in the list will be (in) the lowest common dir
            first_entry = zfile.namelist()[0]
            basedir = first_entry.rsplit("/", 1)[0] if "/" in first_entry else ""
            logger.debug(f"Looking for requirements.txt in '{basedir}'")

            basereqs = os.path.join(basedir, "requirements.txt")
            try:
                extracted = zfile.extract(basereqs, path=plugins_dir)
            except KeyError as e:
                logger.debug(e.args[0])
                continue

            logger.debug(f"Installing dependencies from '{basereqs}'...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", extracted]
            )
            os.remove(extracted)
            os.rmdir(os.path.join(plugins_dir, basedir))

    logger.debug(f"Checking for dependencies in other plugin folders...")

    # Install directory-based plugins
    for requirements_file in glob(f"{plugins_dir}/*/requirements.txt"):
        logger.debug(f"Installing dependencies from '{requirements_file}'...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", requirements_file],
            stdout=subprocess.DEVNULL,
        )

    logger.debug("Finished installing plugin dependencies")


if __name__ == "__main__":
    install_plugin_dependencies()
