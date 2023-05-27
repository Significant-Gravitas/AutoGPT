"""
This code checks if all the packages listed in a requirements file are installed on the system.
It takes the path to the requirements file as an argument.
The code reads the file and extracts package names from each line, ignoring comments.
Then it compares these package names with the list of installed packages on the system.
If required package is missing or is incompatible, it adds it to a list of missing packages.
Prints either "Missing packages:" "All packages are installed."
"""  # noqa: E501
import sys

import pkg_resources


def main():
    # Get the path to the requirements file from command line arguments
    requirements_file = sys.argv[1]

    # Read the requirements file and extract package names from each line, ignoring comments  # noqa: E501
    with open(requirements_file, "r", encoding="utf-8") as f:
        required_packages = []
        for line in f:
            # Ignore comments
            if line.startswith("#"):
                continue
            # Extract package name
            if package_name := line.split("#")[0].strip():
                required_packages.append(package_name)

    # Get a dictionary of installed packages on the system
    installed_packages = {pkg.key: str(pkg.version) for pkg in pkg_resources.working_set}  # noqa: E501

    # Check if each required package is installed and has a compatible version
    missing_packages = []
    for required_package in required_packages:  # Parse requirement string into a Requirement object  # noqa: E501
        pkg = pkg_resources.Requirement.parse(required_package)
        # Check if this requirement is satisfied by any installed package
        if pkg.key not in installed_packages or not pkg.specifier.contains(  # Package not found
            installed_packages[pkg.key]
        ):  # Incompatible version  # noqa: E501
            missing_packages.append(pkg)

    # Print results
    if missing_packages:
        print("Missing packages:")
        print(", ".join(str(pkg) for pkg in missing_packages))
        sys.exit(1)
    else:
        print("All packages are installed.")

    if __name__ == "__main__":
        main()
