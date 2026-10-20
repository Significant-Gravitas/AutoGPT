"""Persisted bounds and identity for one background subtask and its follow-up."""

from datetime import datetime
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, Json
from pydantic.alias_generators import to_camel

from backend.copilot.config import CopilotLlmAuthProvider
from backend.copilot.credential_selection import CredentialPins
from backend.copilot.permissions import CopilotPermissions
from backend.copilot.tree import TurnEnvelope
from backend.copilot.trial_cost_context import TrialCostContext

TASK_SECONDS = 600
FOLLOWUP_SECONDS = 120
DELIVERY_SECONDS = 600
BACKGROUND_TASK_KEY = "background_task_id"
BACKGROUND_STAGE_KEY = "background_task_stage"

BackgroundOutcome = Literal["completed", "failed", "cancelled", "timed_out"]
BackgroundStage = Literal["child", "followup"]


class BackgroundContext(BaseModel):
    envelope: TurnEnvelope
    child_envelope: TurnEnvelope
    permissions: CopilotPermissions | None = None
    cost_context: TrialCostContext
    organization_id: str | None = None
    team_id: str | None = None
    expert_id: str | None = None
    llm_auth_provider: CopilotLlmAuthProvider
    llm_credential_id: str | None = None
    credential_pins: CredentialPins = Field(default_factory=dict)
    impersonated: bool = False


class BackgroundTask(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: str
    user_id: str
    parent_session_id: str
    child_session_id: str
    launch_turn_id: str
    child_turn_id: str
    context: BackgroundContext | Json[BackgroundContext]
    deadline: datetime
    outcome: BackgroundOutcome | None = None
    result: str | None = None
    started_at: datetime | None = None
    followup_started_at: datetime | None = None
    followup_state: Literal["waiting", "queued", "finished", "suppressed"] = "waiting"
    wake_suppressed: bool = False

    @property
    def wake_id(self) -> str:
        return operation_id(self.id, "followup")


def operation_id(parent: str, operation: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"background-subtask:{parent}:{operation}"))


def task_metadata(
    task: BackgroundTask, stage: BackgroundStage, envelope: TurnEnvelope
) -> dict[str, Any]:
    return {
        BACKGROUND_TASK_KEY: task.id,
        BACKGROUND_STAGE_KEY: stage,
        "turn_id": task.child_turn_id if stage == "child" else task.wake_id,
        "envelope": envelope.model_dump(mode="json"),
        "envelope_tree_id": envelope.tree_id,
        "permissions": (
            permissions.model_dump(mode="json")
            if (permissions := task.context.permissions)
            else None
        ),
        "cost_context": task.context.cost_context.model_dump(mode="json"),
        "credential_pins": {
            key: pin.model_dump(mode="json")
            for key, pin in task.context.credential_pins.items()
        },
        "llm_auth_provider": task.context.llm_auth_provider,
        "llm_credential_id": task.context.llm_credential_id,
        "impersonated": task.context.impersonated,
        "tool_name": (
            "run_sub_session" if stage == "child" else "background_subtask_followup"
        ),
        "tool_call_id": task.id,
    }
