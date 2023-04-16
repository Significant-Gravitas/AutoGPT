"""Check if all packages are installed"""
import pkg_resources
import sys
import os


def main() -> None:
    """Check if all packages are installed

    Returns:
        None
    """
    cwd = os.getcwd()
    requirements_file = sys.argv[1]
    requirements_path = os.path.join(cwd, requirements_file)
    with open(requirements_path, "r") as f:
        required_packages = [
            line.strip().split("#")[0].strip() for line in f.readlines()
        ]

    installed_packages = [package.key for package in pkg_resources.working_set]

    missing_packages = []
    for package in required_packages:
        if not package:  # Skip empty lines
            continue
        package_name = package.strip().split("==")[0]
        if package_name.lower() not in installed_packages:
            missing_packages.append(package_name)

    if missing_packages:
        print("Missing packages:")
        print(", ".join(missing_packages))
        sys.exit(1)
    else:
        print("All packages are installed.")


if __name__ == "__main__":
    main()
