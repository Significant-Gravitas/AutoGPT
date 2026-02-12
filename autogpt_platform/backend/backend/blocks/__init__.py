import importlib
import logging
import os
import re
from pathlib import Path
from typing import Sequence, Type, TypeVar

from backend.blocks._base import AnyBlockSchema, BlockType
from backend.util.cache import cached

logger = logging.getLogger(__name__)

T = TypeVar("T")


@cached(ttl_seconds=3600)
def load_all_blocks() -> dict[str, type["AnyBlockSchema"]]:
    from backend.blocks._base import Block
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
    available_blocks: dict[str, type["AnyBlockSchema"]] = {}
    for block_cls in _all_subclasses(Block):
        class_name = block_cls.__name__

        if class_name.endswith("Base"):
            continue

        if not class_name.endswith("Block"):
            raise ValueError(
                f"Block class {class_name} does not end with 'Block'. "
                "If you are creating an abstract class, "
                "please name the class with 'Base' at the end"
            )

        block = block_cls()  # pyright: ignore[reportAbstractUsage]

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
    from ._utils import is_block_auth_configured

    filtered_blocks = {}
    for block_id, block_cls in available_blocks.items():
        if is_block_auth_configured(block_cls):
            filtered_blocks[block_id] = block_cls

    return filtered_blocks


def _all_subclasses(cls: type[T]) -> list[type[T]]:
    subclasses = cls.__subclasses__()
    for subclass in subclasses:
        subclasses += _all_subclasses(subclass)
    return subclasses


# ============== Block access helper functions ============== #


def get_blocks() -> dict[str, Type["AnyBlockSchema"]]:
    return load_all_blocks()


# Note on the return type annotation: https://github.com/microsoft/pyright/issues/10281
def get_block(block_id: str) -> "AnyBlockSchema | None":
    cls = get_blocks().get(block_id)
    return cls() if cls else None


@cached(ttl_seconds=3600)
def get_webhook_block_ids() -> Sequence[str]:
    return [
        id
        for id, B in get_blocks().items()
        if B().block_type in (BlockType.WEBHOOK, BlockType.WEBHOOK_MANUAL)
    ]


@cached(ttl_seconds=3600)
def get_io_block_ids() -> Sequence[str]:
    return [
        id
        for id, B in get_blocks().items()
        if B().block_type in (BlockType.INPUT, BlockType.OUTPUT)
    ]


@cached(ttl_seconds=3600)
def get_human_in_the_loop_block_ids() -> Sequence[str]:
    return [
        id
        for id, B in get_blocks().items()
        if B().block_type == BlockType.HUMAN_IN_THE_LOOP
    ]
