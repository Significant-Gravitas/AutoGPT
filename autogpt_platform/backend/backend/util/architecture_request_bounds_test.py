"""Every string a request stores carries a length bound.

The bound belongs on the request model, which refuses the value before
anything is parsed into the database layer and tells the generated client the
limit. Where a response model inherits the request model, it goes on the write
instead, because a bound on the shared model would make a stored row already
over it unreadable; such a model is listed in ``BOUNDED_AT_THE_WRITE`` with the
response model that reads through it, and this file checks that relationship
still holds.

This walks every module under ``backend.api``, so a request model in a new
module is covered by default. It reads string lengths; a bound on how many
entries a list or dict field takes is checked by that field's own test.
"""

import importlib
import inspect
import pkgutil
from typing import Any, Iterator

import pydantic
import pytest

import backend.api

# Request models that still take unbounded text. The list only shrinks:
# bounding one of these means deleting its line, which
# ``test_the_allowlist_only_shrinks`` enforces.
UNBOUNDED_REQUEST_MODELS = frozenset(
    {
        "backend.api.external.v1.integrations.CreateAPIKeyCredentialRequest",
        "backend.api.external.v1.integrations.CreateHostScopedCredentialRequest",
        "backend.api.external.v1.integrations.CreateUserPasswordCredentialRequest",
        "backend.api.external.v1.integrations.OAuthCompleteRequest",
        "backend.api.external.v1.integrations.OAuthInitiateRequest",
        "backend.api.external.v1.tools.FindAgentRequest",
        "backend.api.external.v1.tools.RunAgentRequest",
        "backend.api.external.v2.pagination.PageRequest",
        "backend.api.features.admin.diagnostics_admin_routes.ScheduleCleanupRequest",
        "backend.api.features.admin.diagnostics_admin_routes.StopExecutionRequest",
        "backend.api.features.admin.diagnostics_admin_routes.StopExecutionsRequest",
        "backend.api.features.admin.execution_analytics_routes.AccuracyTrendsRequest",
        "backend.api.features.admin.execution_analytics_routes.ExecutionAnalyticsRequest",
        "backend.api.features.admin.impersonation_admin_routes.ImpersonationNotifyRequest",
        "backend.api.features.admin.rate_limit_admin_routes.SetUserTierRequest",
        "backend.api.features.api_keys.model.CreateAPIKeyRequest",
        "backend.api.features.auth_email.routes.AuthEmailRequest",
        "backend.api.features.auth_email.routes.AuthEmailSuppressionRequest",
        "backend.api.features.billing.subscriptions.routes.SubscriptionTierRequest",
        "backend.api.features.chat.speech.SpeechRequest",
        "backend.api.features.experts.models.ExpertSkillsUpdate",
        "backend.api.features.experts.routes.AssignPodRequest",
        "backend.api.features.experts.routes.GrantCredentialsRequest",
        "backend.api.features.experts.routes.HireRequest",
        "backend.api.features.experts.routes.InstallWorkflowRequest",
        "backend.api.features.integrations.router.DeviceAuthPollRequest",
        "backend.api.features.mcp.routes.DiscoverToolsRequest",
        "backend.api.features.mcp.routes.MCPOAuthCallbackRequest",
        "backend.api.features.mcp.routes.MCPOAuthLoginRequest",
        "backend.api.features.mcp.routes.MCPStoreTokenRequest",
        "backend.api.features.oauth.AuthorizeRequest",
        "backend.api.features.oauth.UpdateAppLogoRequest",
        "backend.api.features.onboarding_dump.models.FinalizeRequest",
        "backend.api.features.onboarding_dump.models.SuggestedTasksUpdate",
        "backend.api.features.otto.models.ChatRequest",
        "backend.api.features.platform_linking.routes.ConfirmLinkRequest",
        "backend.api.features.schedules.model.ScheduleCreationRequest",
        "backend.api.features.schedules.model.ScheduleUpdateRequest",
        "backend.api.features.schedules.model.SetScheduledChatCadenceRequest",
        "backend.api.features.skill_learning.models.EditSkillRequest",
        "backend.api.features.skills.model.UploadCopilotSkillRequest",
        "backend.api.features.subscription_trial_models.TrialCheckoutRequest",
    }
)

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
        f"{_qualified(model)}.{path}"
        for model in _request_models()
        if _qualified(model) not in BOUNDED_AT_THE_WRITE
        and _qualified(model) not in UNBOUNDED_REQUEST_MODELS
        for path in _unbounded_string_paths(model)
    ]
    assert not unbounded, (
        "These request fields store text with no length bound. Give each one a "
        "`max_length`, or bound it at the write and list its model in "
        "BOUNDED_AT_THE_WRITE:\n  " + "\n  ".join(sorted(unbounded))
    )


def test_the_allowlist_only_shrinks():
    models = {_qualified(model): model for model in _request_models()}
    gone = sorted(UNBOUNDED_REQUEST_MODELS - models.keys())
    assert not gone, (
        "These models are in UNBOUNDED_REQUEST_MODELS but no longer exist; "
        "drop their lines:\n  " + "\n  ".join(gone)
    )
    bounded_now = sorted(
        name
        for name in UNBOUNDED_REQUEST_MODELS
        if not list(_unbounded_string_paths(models[name]))
    )
    assert not bounded_now, (
        "These models are bounded now, so they must come off "
        "UNBOUNDED_REQUEST_MODELS:\n  " + "\n  ".join(bounded_now)
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
    for module in pkgutil.walk_packages(backend.api.__path__, "backend.api."):
        if module.name.endswith("_test") or "_test." in module.name:
            continue
        imported = importlib.import_module(module.name)
        for name, obj in vars(imported).items():
            if not (inspect.isclass(obj) and issubclass(obj, pydantic.BaseModel)):
                continue
            if obj.__module__ != module.name:
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
