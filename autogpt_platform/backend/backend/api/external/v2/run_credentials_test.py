"""A run's `credentials_inputs` reach the executor as credential models.

`add_graph_execution` maps each credential onto the graph's nodes with
`.model_dump()`. Typed as a plain dict, every v2 run of an agent that needs a
credential reached it as a dict and failed with a 500.
"""

from unittest.mock import AsyncMock, Mock

import pytest
import pytest_mock
from prisma.enums import APIKeyPermission
from pydantic import ValidationError

from backend.data.execution import ExecutionStatus, GraphExecutionMeta
from backend.data.model import CredentialsMetaInput

from .models import AgentRunRequest, AgentRunScheduleCreateRequest
from .tenancy import TenantContext

FIELD = "openai_api_key_credentials"
CREDENTIAL = {"id": "cred-1", "title": "Mine", "provider": "openai", "type": "api_key"}


def test_a_run_request_parses_credentials_into_models() -> None:
    request = AgentRunRequest.model_validate(
        {"credentials_inputs": {FIELD: CREDENTIAL}}
    )

    parsed = request.credentials_inputs[FIELD]
    assert isinstance(parsed, CredentialsMetaInput)
    assert parsed.model_dump(exclude_none=True) == CREDENTIAL


def test_a_schedule_request_parses_credentials_into_models() -> None:
    request = AgentRunScheduleCreateRequest.model_validate(
        {
            "graph_id": "graph-1",
            "name": "Daily",
            "cron": "0 9 * * *",
            "credentials_inputs": {FIELD: CREDENTIAL},
        }
    )

    assert isinstance(request.credentials_inputs[FIELD], CredentialsMetaInput)


def test_a_credential_without_an_id_is_a_validation_error() -> None:
    with pytest.raises(ValidationError):
        AgentRunRequest.model_validate(
            {"credentials_inputs": {FIELD: {"provider": "openai", "type": "api_key"}}}
        )


@pytest.mark.asyncio
async def test_a_run_hands_the_executor_credential_models(
    mocker: pytest_mock.MockFixture,
) -> None:
    from .library.agents import execute_agent

    mocker.patch(
        "backend.api.external.v2.library.agents.graph_exec_limiter.check",
        new_callable=AsyncMock,
    )
    mocker.patch(
        "backend.api.external.v2.library.helpers.get_credit_model",
        new_callable=AsyncMock,
        return_value=Mock(get_credits=AsyncMock(return_value=100)),
    )
    mocker.patch(
        "backend.api.features.library.db.get_library_agent",
        new_callable=AsyncMock,
        return_value=Mock(organization_id="org-1", graph_id="graph-1", graph_version=1),
    )
    add_execution = mocker.patch(
        "backend.executor.utils.add_graph_execution",
        new_callable=AsyncMock,
        return_value=GraphExecutionMeta.model_construct(
            id="run-1",
            user_id="user-1",
            graph_id="graph-1",
            graph_version=1,
            preset_id=None,
            status=ExecutionStatus.QUEUED,
            started_at=None,
            ended_at=None,
            inputs={},
            credential_inputs=None,
            is_shared=False,
            share_token=None,
            stats=None,
            organization_id="org-1",
            team_id=None,
        ),
    )

    await execute_agent(
        request=AgentRunRequest.model_validate(
            {"credentials_inputs": {FIELD: CREDENTIAL}}
        ),
        agent_id="agent-1",
        auth=TenantContext(
            user_id="user-1",
            scopes=[APIKeyPermission.RUN_AGENT],
            type="api_key",
            organization_id="org-1",
        ),
    )

    passed = add_execution.await_args.kwargs["graph_credentials_inputs"]
    assert passed[FIELD].model_dump(exclude_none=True) == CREDENTIAL
