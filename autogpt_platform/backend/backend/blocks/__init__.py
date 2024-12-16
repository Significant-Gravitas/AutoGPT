import importlib
import os
import re
from pathlib import Path
from typing import Type, TypeVar

from backend.data.block import Block

# Dynamically load all modules under backend.blocks
AVAILABLE_MODULES = []
current_dir = Path(__file__).parent
modules = [
    str(f.relative_to(current_dir))[:-3].replace(os.path.sep, ".")
    for f in current_dir.rglob("*.py")
    if f.is_file() and f.name != "__init__.py"
]
for module in modules:
    if not re.match("^[a-z0-9_.]+$", module):
        raise ValueError(
            f"Block module {module} error: module name must be lowercase, "
            "and contain only alphanumeric characters and underscores."
        )

    importlib.import_module(f".{module}", package=__name__)
    AVAILABLE_MODULES.append(module)

# Load all Block instances from the available modules
AVAILABLE_BLOCKS: dict[str, Type[Block]] = {}


T = TypeVar("T")


def all_subclasses(cls: Type[T]) -> list[Type[T]]:
    subclasses = cls.__subclasses__()
    for subclass in subclasses:
        subclasses += all_subclasses(subclass)
    return subclasses


for block_cls in all_subclasses(Block):
    name = block_cls.__name__

    if block_cls.__name__.endswith("Base"):
        continue

    if not block_cls.__name__.endswith("Block"):
        raise ValueError(
            f"Block class {block_cls.__name__} does not end with 'Block', If you are creating an abstract class, please name the class with 'Base' at the end"
        )

    block = block_cls.create()

    if not isinstance(block.id, str) or len(block.id) != 36:
        raise ValueError(f"Block ID {block.name} error: {block.id} is not a valid UUID")

    if block.id in AVAILABLE_BLOCKS:
        raise ValueError(f"Block ID {block.name} error: {block.id} is already in use")

    input_schema = block.input_schema.model_fields
    output_schema = block.output_schema.model_fields

    # Make sure `error` field is a string in the output schema
    if "error" in output_schema and output_schema["error"].annotation is not str:
        raise ValueError(
            f"{block.name} `error` field in output_schema must be a string"
        )

    # Make sure all fields in input_schema and output_schema are annotated and has a value
    for field_name, field in [*input_schema.items(), *output_schema.items()]:
        if field.annotation is None:
            raise ValueError(
                f"{block.name} has a field {field_name} that is not annotated"
            )
        if field.json_schema_extra is None:
            raise ValueError(
                f"{block.name} has a field {field_name} not defined as SchemaField"
            )

    for field in block.input_schema.model_fields.values():
        if field.annotation is bool and field.default not in (True, False):
            raise ValueError(f"{block.name} has a boolean field with no default value")

    if block.disabled:
        continue

    AVAILABLE_BLOCKS[block.id] = block_cls

__all__ = ["AVAILABLE_MODULES", "AVAILABLE_BLOCKS"]
