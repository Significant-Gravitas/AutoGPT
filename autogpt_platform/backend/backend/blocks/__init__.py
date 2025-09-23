import importlib
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from autogpt_libs.utils.cache import cached

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from backend.data.block import Block

T = TypeVar("T")


@cached()
def load_all_blocks() -> dict[str, type["Block"]]:
    from backend.data.block import Block
    from backend.util.settings import Config

    # Check if example blocks should be loaded from settings
    config = Config()
    load_examples = config.enable_example_blocks

    # Dynamically load all modules under backend.blocks
    current_dir = Path(__file__).parent
    modules = []
    for f in current_dir.rglob("*.py"):
        if not f.is_file() or f.name == "__init__.py" or f.name.startswith("test_"):
            continue

        # Skip examples directory if not enabled
        relative_path = f.relative_to(current_dir)
        if not load_examples and relative_path.parts[0] == "examples":
            continue

        module_path = str(relative_path)[:-3].replace(os.path.sep, ".")
        modules.append(module_path)

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

    # Filter out blocks with incomplete auth configs, e.g. missing OAuth server secrets
    from backend.data.block import is_block_auth_configured

    filtered_blocks = {}
    for block_id, block_cls in available_blocks.items():
        if is_block_auth_configured(block_cls):
            filtered_blocks[block_id] = block_cls

    return filtered_blocks


__all__ = ["load_all_blocks"]


def all_subclasses(cls: type[T]) -> list[type[T]]:
    subclasses = cls.__subclasses__()
    for subclass in subclasses:
        subclasses += all_subclasses(subclass)
    return subclasses
