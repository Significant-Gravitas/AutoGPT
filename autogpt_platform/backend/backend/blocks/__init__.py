import functools
import importlib
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from backend.data.block import Block

T = TypeVar("T")


@functools.cache
def load_all_blocks() -> dict[str, type["Block"]]:
    from backend.data.block import Block

    # Dynamically load all modules under backend.blocks
    current_dir = Path(__file__).parent
    modules = [
        str(f.relative_to(current_dir))[:-3].replace(os.path.sep, ".")
        for f in current_dir.rglob("*.py")
        if f.is_file() and f.name != "__init__.py" and not f.name.startswith("test_")
    ]
    for module in modules:
        if not re.match("^[a-z0-9_.]+$", module):
            raise ValueError(
                f"Block module {module} error: module name must be lowercase, "
                "and contain only alphanumeric characters and underscores."
            )

        importlib.import_module(f".{module}", package=__name__)

    # Load all Block instances from the available modules
    available_blocks: dict[str, type["Block"]] = {}
    for block_cls in all_subclasses(Block):
        class_name = block_cls.__name__

        if class_name.endswith("Base"):
            continue

        if not class_name.endswith("Block"):
            raise ValueError(
                f"Block class {class_name} does not end with 'Block'. "
                "If you are creating an abstract class, "
                "please name the class with 'Base' at the end"
            )

        block = block_cls.create()

        if not isinstance(block.id, str) or len(block.id) != 36:
            raise ValueError(
                f"Block ID {block.name} error: {block.id} is not a valid UUID"
            )

        if block.id in available_blocks:
            raise ValueError(
                f"Block ID {block.name} error: {block.id} is already in use"
            )

        input_schema = block.input_schema.model_fields
        output_schema = block.output_schema.model_fields

        # Make sure `error` field is a string in the output schema
        if "error" in output_schema and output_schema["error"].annotation is not str:
            raise ValueError(
                f"{block.name} `error` field in output_schema must be a string"
            )

        # Ensure all fields in input_schema and output_schema are annotated SchemaFields
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
                raise ValueError(
                    f"{block.name} has a boolean field with no default value"
                )

        available_blocks[block.id] = block_cls

    return available_blocks


__all__ = ["load_all_blocks"]


def all_subclasses(cls: type[T]) -> list[type[T]]:
    subclasses = cls.__subclasses__()
    for subclass in subclasses:
        subclasses += all_subclasses(subclass)
    return subclasses
