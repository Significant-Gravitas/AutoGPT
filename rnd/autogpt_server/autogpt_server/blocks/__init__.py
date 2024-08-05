import glob
import importlib
import os
import re
from pathlib import Path

from autogpt_server.data.block import Block

# Dynamically load all modules under autogpt_server.blocks
AVAILABLE_MODULES = []
current_dir = os.path.dirname(__file__)
modules = glob.glob(os.path.join(current_dir, "*.py"))
modules = [
    Path(f).stem
    for f in modules
    if os.path.isfile(f) and f.endswith(".py") and not f.endswith("__init__.py")
]
for module in modules:
    if not re.match("^[a-z_]+$", module):
        raise ValueError(
            f"Block module {module} error: module name must be lowercase, separated by underscores, and contain only alphabet characters"
        )

    importlib.import_module(f".{module}", package=__name__)
    AVAILABLE_MODULES.append(module)

# Load all Block instances from the available modules
AVAILABLE_BLOCKS = {}


def all_subclasses(clz):
    subclasses = clz.__subclasses__()
    for subclass in subclasses:
        subclasses += all_subclasses(subclass)
    return subclasses


for cls in all_subclasses(Block):
    name = cls.__name__

    if cls.__name__.endswith("Base"):
        continue

    if not cls.__name__.endswith("Block"):
        raise ValueError(
            f"Block class {cls.__name__} does not end with 'Block', If you are creating an abstract class, please name the class with 'Base' at the end"
        )

    block = cls()

    if not isinstance(block.id, str) or len(block.id) != 36:
        raise ValueError(f"Block ID {block.name} error: {block.id} is not a valid UUID")

    if block.id in AVAILABLE_BLOCKS:
        raise ValueError(f"Block ID {block.name} error: {block.id} is already in use")

    if block.disabled:
        continue

    AVAILABLE_BLOCKS[block.id] = block

__all__ = ["AVAILABLE_MODULES", "AVAILABLE_BLOCKS"]
