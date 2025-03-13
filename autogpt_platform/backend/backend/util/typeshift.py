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
# 2) A GLOBAL CACHE that maps (type, id-of-replacements) -> transformed type
###############################################################################
# We must also incorporate the ID of the replacements dict
# if we want the cache to be invalidated when the global dictionary changes.
# But if you intend for this dictionary never to change at runtime,
# you can simply use a single cache keyed by the 'type' alone.
_GLOBAL_TYPE_TRANSFORM_CACHE: dict[Any, Any] = {}


def transform_type(tp: Any) -> Any:
    """
    Public entry point:
      - Recursively traverse a type annotation (could be a Pydantic model, list[model], Union, etc.).
      - Return a new type where each "original_type" in `GLOBAL_REPLACEMENTS` is replaced.
      - Uses a global cache to avoid rebuilding the same transformations.
      - Also avoids infinite recursion for self-referencing or mutually-referencing models.
    """
    # If we've already transformed this `tp` given the current replacements, return it
    # (assuming the dictionary does not change at runtime).
    if tp in _GLOBAL_TYPE_TRANSFORM_CACHE:
        return _GLOBAL_TYPE_TRANSFORM_CACHE[tp]

    # We keep a *local* recursion cache to avoid infinite loops in this single call chain
    local_cache: dict[Any, Any] = {}
    transformed = _transform_type_recursive(tp, local_cache)

    # Store in the global cache
    _GLOBAL_TYPE_TRANSFORM_CACHE[tp] = transformed
    return transformed


def _transform_type_recursive(tp: Any, local_cache: dict[Any, Any]) -> Any:
    """
    Does the actual recursion:
      - If `tp` is in `GLOBAL_REPLACEMENTS`, use that replacement.
      - If it's a Pydantic model, build a new model with replaced field types.
      - If it's a container (list, tuple, Union, etc.), transform each argument.
      - Otherwise, return as-is.
    """
    if tp in local_cache:
        # Already transformed (or in process); return it
        return local_cache[tp]

    # If exactly in the global replacements, do a direct swap
    if tp in GLOBAL_REPLACEMENTS:
        replaced = GLOBAL_REPLACEMENTS[tp]
        local_cache[tp] = replaced
        return replaced

    origin = get_origin(tp)

    # Handle container types: list[X], Union[X, Y], tuple[X, Y], etc.
    if origin in (list, tuple, set, frozenset):
        args = get_args(tp)
        if len(args) == 1:
            # e.g. list[X]
            new_arg = _transform_type_recursive(args[0], local_cache)
            new_container = origin[new_arg]  # type: ignore
        else:
            # e.g. tuple[X, Y]
            new_args = tuple(_transform_type_recursive(a, local_cache) for a in args)
            new_container = origin[new_args]  # type: ignore
        local_cache[tp] = new_container
        return new_container

    if origin is Union:
        args = get_args(tp)
        new_args = tuple(_transform_type_recursive(a, local_cache) for a in args)
        new_union = Union[new_args]
        local_cache[tp] = new_union
        return new_union

    # Handle Pydantic model classes
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        new_model_cls = _transform_model(tp, local_cache)
        local_cache[tp] = new_model_cls
        return new_model_cls

    # Otherwise, no transformation
    local_cache[tp] = tp
    return tp


def _transform_model(
    model_cls: type[BaseModel], local_cache: dict[Any, Any]
) -> type[BaseModel]:
    """
    Build a new Pydantic model that mirrors `model_cls` but with replaced field types.
    Prevent infinite recursion with local_cache placeholders.
    """
    # If we've started building this model in the recursion, return immediately
    if model_cls in local_cache and local_cache[model_cls] is None:
        return model_cls

    # Mark as 'in progress' with None
    local_cache[model_cls] = None

    # Build new fields
    field_defs = {}
    for field_name, field_info in model_cls.model_fields.items():
        old_annotation = field_info.annotation
        new_annotation = _transform_type_recursive(old_annotation, local_cache)

        if field_info.is_required():
            default_val = ...
        else:
            default_val = field_info.default

        field_defs[field_name] = (new_annotation, default_val)

    new_name = f"{model_cls.__name__}Transformed"
    NewModel = create_model(
        new_name,
        __base__=BaseModel,
        **field_defs,
    )

    local_cache[model_cls] = NewModel
    return NewModel
