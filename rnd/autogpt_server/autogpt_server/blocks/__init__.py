import os
import glob
import importlib
from autogpt_server.data.block import Block

# Dynamically load all modules under autogpt_server.blocks
AVAILABLE_MODULES = []
current_dir = os.path.dirname(__file__)
modules = glob.glob(os.path.join(current_dir, "*.py"))
modules = [
    os.path.basename(f)[:-3]
    for f in modules
    if os.path.isfile(f) and f.endswith(".py") and not f.endswith("__init__.py")
]
for module in modules:
    importlib.import_module(f".{module}", package=__name__)
    AVAILABLE_MODULES.append(module)

# Load all Block instances from the available modules
AVAILABLE_BLOCKS = {
    block.id: block
    for block in [v() for v in Block.__subclasses__()]
}

__all__ = ["AVAILABLE_MODULES", "AVAILABLE_BLOCKS"]
