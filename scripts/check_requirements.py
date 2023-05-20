import re
import sys
import hashlib
import os
import pkg_resources


def main():
    requirements_file = sys.argv[1]
    # check hash
    fileHash:str
    with open(requirements_file, 'rb') as f:
        sha1obj = hashlib.sha1()
        sha1obj.update(f.read())
        hash = sha1obj.hexdigest()
        if os.path.exists('hash.calc'):
            lastHash = open('hash.calc').read()
            if lastHash == hash:
                print("No requirements changed")
                sys.exit(0)
            else:
                os.remove('hash.calc')
        fileHash = hash
    with open(requirements_file, "r") as f:
        required_packages = [
            line.strip().split("#")[0].strip() for line in f.readlines()
        ]

    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

    missing_packages = []
    for required_package in required_packages:
        if not required_package:  # Skip empty lines
            continue
        pkg = pkg_resources.Requirement.parse(required_package)
        if (
            pkg.key not in installed_packages
            or pkg_resources.parse_version(installed_packages[pkg.key])
            not in pkg.specifier
        ):
            missing_packages.append(str(pkg))

    if missing_packages:
        print("Missing packages:")
        print(", ".join(missing_packages))
        sys.exit(1)
    else:
        with open('hash.calc','a') as f:
            f.write(fileHash)
        print("All packages are installed.")


if __name__ == "__main__":
    main()
