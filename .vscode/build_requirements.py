import os
import pkg_resources
import re

def get_imports_from_file(file_path):
    with open(file_path, 'r') as file:
        contents = file.read()
    imports = re.findall(r'^import (\S+)', contents, re.MULTILINE)
    imports += re.findall(r'^from (\S+) import', contents, re.MULTILINE)
    return set(imports)

def get_installed_packages():
    return {pkg.key for pkg in pkg_resources.working_set}

def main(project_directory):
    all_imports = set()
    for root, dirs, files in os.walk(project_directory):
        py_files = [f for f in files if f.endswith('.py')]
        for file in py_files:
            all_imports.update(get_imports_from_file(os.path.join(root, file)))

    installed_packages = get_installed_packages()
    with open('requirements.txt', 'w') as req_file:
        for imp in all_imports:
            if imp in installed_packages:
                version = pkg_resources.get_distribution(imp).version
                req_file.write(f"{imp}=={version}\n")

if __name__ == "__main__":
    main('.')
