import os
import subprocess
import sys
import zipfile
from pathlib import Path


def main():
    # Plugin packages
    plugins_dir = Path(os.getenv("PLUGINS_DIR", "plugins"))
    for plugin in plugins_dir.glob("*.zip"):
        with zipfile.ZipFile(str(plugin), "r") as zfile:
            try:
                basedir = zfile.namelist()[0]
                basereqs = os.path.join(basedir, 'requirements.txt')
                extracted = zfile.extract(basereqs)
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", extracted])
            except KeyError:
                continue


if __name__ == "__main__":
    main()
