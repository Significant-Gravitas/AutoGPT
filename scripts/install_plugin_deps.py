import os
import subprocess
import sys
import zipfile
from pathlib import Path


def install_plugin_dependencies():
    """
    Installs dependencies for all plugins in the plugins dir.

    Args:
        None

    Returns:
        None
    """
    plugins_dir = Path(os.getenv("PLUGINS_DIR", "plugins"))
    for plugin in plugins_dir.glob("*.zip"):
        with zipfile.ZipFile(str(plugin), "r") as zfile:
            try:
                basedir = zfile.namelist()[0]
                basereqs = os.path.join(basedir, "requirements.txt")
                extracted = zfile.extract(basereqs, path=plugins_dir)
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-r", extracted]
                )
                os.remove(extracted)
                os.rmdir(os.path.join(plugins_dir, basedir))
            except KeyError:
                continue


if __name__ == "__main__":
    install_plugin_dependencies()
