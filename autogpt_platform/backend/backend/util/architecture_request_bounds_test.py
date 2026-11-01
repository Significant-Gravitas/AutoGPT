"""Every string a request stores carries a length bound.

A text column written from a request body is read back whole by a listing, so
its bound is what keeps that listing's size predictable. The bound belongs on
the request model, which refuses the value before anything is parsed into the
database layer and tells the generated client the limit.

Where a response model inherits the request model, the bound goes on the write
instead: a bound on the shared model would make a stored row already longer
than it unreadable. Such a model is listed in ``BOUNDED_AT_THE_WRITE`` with the
response model that reads through it, and this file checks that relationship
still holds — so the exemption disappears by itself once it stops being true.
"""

import importlib
import inspect
from collections.abc import Iterator
from typing import Any

import pydantic
import pytest

# Modules whose request models are bounded. A module joins this list with the
# change that bounds its models; it is never removed to make a new field pass.
BOUNDED_MODULES = [
    "backend.api.features.chat.routes",
    "backend.api.features.library.model",
    "backend.api.features.orgs.model",
    "backend.api.features.orgs.team_model",
    "backend.api.features.store.model",
    "backend.api.features.store.skill_model",
    "backend.api.features.transfers.model",
    "backend.api.features.workspace.folder_routes",
]

# Request model -> the response model that reads stored rows through it.
BOUNDED_AT_THE_WRITE = {
    "backend.api.features.library.model.LibraryAgentPresetCreatable": (
        "backend.api.features.library.model.LibraryAgentPreset"
    ),
}

_REQUEST_MODEL_SUFFIXES = ("Request", "Creatable", "Updatable", "Create", "Update")

# A JSON-schema string position is bounded by a length, a closed value set, or
# a format whose values are fixed-width.
_BOUNDED_FORMATS = frozenset({"date", "date-time", "time", "duration", "uuid"})


def test_every_request_string_field_is_bounded():
    unbounded = [
        f"{model.__module__}.{model.__qualname__}.{path}"
        for model in _request_models()
        if _qualified(model) not in BOUNDED_AT_THE_WRITE
        for path in _unbounded_string_paths(model)
    ]
    assert not unbounded, (
        "These request fields store text with no length bound. Give each one a "
        "`max_length`, or bound it at the write and list its model in "
        "BOUNDED_AT_THE_WRITE:\n  " + "\n  ".join(sorted(unbounded))
    )


def test_write_bounded_models_are_still_read_through_by_a_response():
    for request_path, response_path in BOUNDED_AT_THE_WRITE.items():
        request_model = _import_attr(request_path)
        response_model = _import_attr(response_path)
        assert issubclass(response_model, request_model), (
            f"{response_path} no longer reads stored rows through "
            f"{request_path}, so that model can take field bounds now; "
            "bound its fields and drop the BOUNDED_AT_THE_WRITE entry"
        )


@pytest.mark.parametrize(
    "model_path, field_name, over_the_bound",
    [
        ("backend.api.features.orgs.model.CreateOrgRequest", "description", "x" * 2001),
        (
            "backend.api.features.orgs.model.CreateInvitationRequest",
            "team_ids",
            ["i"] * 101,
        ),
        (
            "backend.api.features.chat.routes.UpdateSessionTitleRequest",
            "title",
            "x" * 201,
        ),
        (
            "backend.api.features.store.model.ProfileUpdateRequest",
            "links",
            ["u" * 2049],
        ),
        (
            "backend.api.features.transfers.model.CreateTransferRequest",
            "reason",
            "x" * 2001,
        ),
    ],
)
def test_the_declared_bounds_refuse_a_value_over_them(
    model_path: str, field_name: str, over_the_bound: Any
):
    """The bounds validate, rather than only appearing in the schema."""
    model = _import_attr(model_path)
    fields = {
        name: _placeholder(field)
        for name, field in model.model_fields.items()
        if field.is_required()
    }
    fields[field_name] = over_the_bound

    with pytest.raises(pydantic.ValidationError):
        model(**fields)


def _placeholder(field) -> Any:
    annotation = field.annotation
    if annotation is bool:
        return False
    if annotation is int:
        return 1
    if getattr(annotation, "__origin__", None) is list:
        return []
    return "x"


def test_the_check_fails_on_an_unbounded_string():
    class WithBound(pydantic.BaseModel):
        text: str = pydantic.Field(max_length=10)

    class WithoutBound(pydantic.BaseModel):
        text: str

    class WithoutBoundInAList(pydantic.BaseModel):
        texts: list[str]

    assert list(_unbounded_string_paths(WithBound)) == []
    assert list(_unbounded_string_paths(WithoutBound)) == ["text"]
    assert list(_unbounded_string_paths(WithoutBoundInAList)) == ["texts[]"]


def _request_models() -> Iterator[type[pydantic.BaseModel]]:
    for module_path in BOUNDED_MODULES:
        module = importlib.import_module(module_path)
        for name, obj in vars(module).items():
            if not (inspect.isclass(obj) and issubclass(obj, pydantic.BaseModel)):
                continue
            if obj.__module__ != module_path:
                continue
            if name.endswith(_REQUEST_MODEL_SUFFIXES):
                yield obj


def _unbounded_string_paths(model: type[pydantic.BaseModel]) -> Iterator[str]:
    """Names of this model's own fields that hold unbounded string values."""
    try:
        schema = model.model_json_schema(ref_template="{model}")
    except pydantic.PydanticInvalidForJsonSchema:  # pragma: no cover - not hit today
        pytest.skip(f"{model.__qualname__} has no JSON schema")
    for field_name in model.model_fields:
        prop = (schema.get("properties") or {}).get(field_name)
        if isinstance(prop, dict):
            yield from (f"{field_name}{suffix}" for suffix in _unbounded(prop))


def _unbounded(node: dict[str, Any], suffix: str = "") -> Iterator[str]:
    """Suffixes of the unbounded string positions inside one schema node."""
    if node.get("enum") or "const" in node:
        return
    if node.get("type") == "string":
        if "maxLength" not in node and node.get("format") not in _BOUNDED_FORMATS:
            yield suffix
        return
    for key in ("anyOf", "oneOf", "allOf"):
        for child in node.get(key) or []:
            if isinstance(child, dict):
                yield from _unbounded(child, suffix)
    for key in ("items", "additionalProperties"):
        child = node.get(key)
        if isinstance(child, dict):
            yield from _unbounded(child, f"{suffix}[]")
    for child in node.get("prefixItems") or []:
        if isinstance(child, dict):
            yield from _unbounded(child, f"{suffix}[]")


def _qualified(model: type[pydantic.BaseModel]) -> str:
    return f"{model.__module__}.{model.__qualname__}"


def _import_attr(path: str) -> Any:
    module_path, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(module_path), attr)
