import contextlib
import os
import sys
from importlib.metadata import version

try:
    import poetry.factory  # noqa
except ModuleNotFoundError:
    os.system(f"{sys.executable} -m pip install 'poetry>=1.6.1,<2.0.0'")

from poetry.core.constraints.version.version import Version
from poetry.factory import Factory


def main():
    poetry_project = Factory().create_poetry()
    dependency_group = poetry_project.package.dependency_group("main")

    missing_packages = []
    for dep in dependency_group.dependencies:
        if dep.is_optional():
            continue
        # Try to verify that the installed version is suitable
        with contextlib.suppress(ModuleNotFoundError):
            installed_version = version(dep.name)  # if this fails -> not installed
            if dep.constraint.allows(Version.parse(installed_version)):
                continue
        # If the above verification fails, mark the package as missing
        missing_packages.append(str(dep))

    if missing_packages:
        print("Missing packages:")
        print(", ".join(missing_packages))
        sys.exit(1)


if __name__ == "__main__":
    main()
