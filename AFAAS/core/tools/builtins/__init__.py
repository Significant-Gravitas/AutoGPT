import glob
import os


def load_builtin_modules():
    # Get the directory of the current file (__init__.py)
    builtins_dir = os.path.dirname(__file__)

    # Use glob to find all .py files in the directory, excluding __init__.py
    module_files = glob.glob(os.path.join(builtins_dir, "*.py"))
    module_files = [f for f in module_files if not f.endswith("__init__.py")]

    # Convert file paths to module names by extracting the base name and removing the .py extension
    builtin_modules = [
        "AFAAS.core.tools.builtins." + os.path.basename(f)[:-3] for f in module_files
    ]

    return builtin_modules


# Call the function to generate the list of builtin modules
BUILTIN_MODULES = load_builtin_modules()
