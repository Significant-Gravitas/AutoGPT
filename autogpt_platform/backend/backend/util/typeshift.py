from __future__ import annotations

import inspect
from types import UnionType
from typing import Any, Union, get_args, get_origin

from prisma import Json as PrismaJson
from pydantic import BaseModel, Json, create_model

###############################################################################
# 1) A GLOBAL REPLACEMENTS DICTIONARY
###############################################################################
# By default, we replace pydantic.Json & prisma.Json with Any.
# You can add or remove entries if needed.
GLOBAL_REPLACEMENTS: dict[Any, Any] = {
    Json: Any,
    PrismaJson: Any,
}


###############################################################################
# 2) A GLOBAL CACHE that maps `type` -> transformed type
###############################################################################
_GLOBAL_TYPE_TRANSFORM_CACHE: dict[Any, Any] = {}


def transform_type(tp: Any) -> Any:
    """
    Public entry point to transform `tp` by:
      - Replacing any types found in GLOBAL_REPLACEMENTS
      - Recursively handling container types (list[X], set[X], etc.)
      - Recursively handling union types (X | Y or Union[X, Y])
      - Transforming Pydantic models by creating a new dynamic model
        if any fields are replaced
      - Caching globally so repeated calls for the same type won't rebuild
    """
    # If we've already transformed this type, just return it
    if tp in _GLOBAL_TYPE_TRANSFORM_CACHE:
        return _GLOBAL_TYPE_TRANSFORM_CACHE[tp]

    # Local recursion cache to avoid infinite loops in cyclical references
    local_cache: dict[Any, Any] = {}
    transformed = _transform_type_recursive(tp, local_cache)

    _GLOBAL_TYPE_TRANSFORM_CACHE[tp] = transformed
    return transformed


def _transform_type_recursive(tp: Any, local_cache: dict[Any, Any]) -> Any:
    """
    Recursively apply transformations:
      1) If `tp` is in GLOBAL_REPLACEMENTS, return the replacement.
      2) If container type (list, tuple, set, frozenset), transform sub-args.
      3) If union type (typing.Union or types.UnionType), transform sub-args.
      4) If a Pydantic model, create a new model with replaced fields.
      5) Otherwise, return `tp` as-is.
    """
    if tp in local_cache:
        return local_cache[tp] or tp

    # 1) Direct replacement if tp is in the replacements dict
    if tp in GLOBAL_REPLACEMENTS:
        replaced = GLOBAL_REPLACEMENTS[tp]
        local_cache[tp] = replaced
        return replaced

    origin = get_origin(tp)

    # 2) Check if it's a built-in container type (list, tuple, set, frozenset)
    if origin in (list, tuple, set, frozenset):
        args = get_args(tp)
        if len(args) == 1:
            # e.g. list[X]
            transformed_arg = _transform_type_recursive(args[0], local_cache)
            new_tp = origin[transformed_arg]  # type: ignore
        else:
            # e.g. tuple[X, Y]
            transformed_args = tuple(
                _transform_type_recursive(a, local_cache) for a in args
            )
            new_tp = origin[transformed_args]  # type: ignore
        local_cache[tp] = new_tp
        return new_tp

    # 3) Check if it's a union (covers old-style typing.Union and Python 3.10+ pipe union)
    if origin in (Union, UnionType):
        args = get_args(tp)
        transformed_args = tuple(
            _transform_type_recursive(a, local_cache) for a in args
        )
        new_union = Union[transformed_args]
        local_cache[tp] = new_union
        return new_union

    # 4) If it's a Pydantic model class
    if inspect.isclass(tp) and issubclass(tp, BaseModel):
        # Ensure forward references in the original are resolved
        tp.model_rebuild(force=True)
        new_model = _transform_model(tp, local_cache)
        local_cache[tp] = new_model
        return new_model

    # 5) Otherwise, no transformation
    local_cache[tp] = tp
    return tp


def _transform_model(
    model_cls: type[BaseModel], local_cache: dict[Any, Any]
) -> type[BaseModel]:
    """
    Create a brand-new Pydantic model if needed (i.e. if any field changes),
    or reuse the original model_cls if nothing changes.
    """
    # Check if we have a placeholder from cyclical references
    if model_cls in local_cache and local_cache[model_cls] is None:
        return model_cls

    # Mark this model as "in progress"
    local_cache[model_cls] = None

    field_defs = {}
    any_field_changed = False

    for field_name, field_info in model_cls.model_fields.items():
        old_anno = field_info.annotation
        transformed_anno = _transform_type_recursive(old_anno, local_cache)

        if transformed_anno != old_anno:
            any_field_changed = True

        # Use "..." if field is required, else its default
        default_val = ... if field_info.is_required() else field_info.default
        field_defs[field_name] = (transformed_anno, default_val)

    if not any_field_changed:
        # Reuse the original model if nothing changed
        local_cache[model_cls] = model_cls
        return model_cls

    # Build a new model class
    new_name = f"{model_cls.__name__}Transformed"
    NewModel = create_model(new_name, __base__=model_cls, **field_defs)
    NewModel.model_rebuild(force=True)

    local_cache[model_cls] = NewModel
    return NewModel
