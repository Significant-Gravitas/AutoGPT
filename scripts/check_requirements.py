import re
import sys
import os
import zipfile
from pathlib import Path
import subprocess
import sys

import pkg_resources


def _install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    requirements_file = sys.argv[1]
    with open(requirements_file, "r") as f:
        required_packages = [
            line.strip().split("#")[0].strip() for line in f.readlines()
        ]

    installed_packages = [package.key for package in pkg_resources.working_set]

    # Plugin packages
    plugins_dir = Path(os.getenv("PLUGINS_DIR", "plugins"))
    for plugin in plugins_dir.glob("*.zip"):
        with zipfile.ZipFile(str(plugin), "r") as zfile:
            basedir = zfile.namelist()[0]
            basereqs = os.path.join(basedir, 'requirements.txt')
            required_packages += [
                line.decode('utf-8').strip().split("#")[0].strip()
                for line in zfile.open(basereqs).readlines()
            ]

    missing_packages = []
    for package in required_packages:
        if not package:  # Skip empty lines
            continue
        package_name = re.split("[<>=@ ]+", package.strip())[0]
        if package_name.lower() not in installed_packages:
            missing_packages.append(package_name)

    if missing_packages:
        print("Missing packages:")
        print(", ".join(missing_packages))
        for package in missing_packages:
            _install(package)
        sys.exit(1)
    else:
        print("All packages are installed.")


if __name__ == "__main__":
    main()
