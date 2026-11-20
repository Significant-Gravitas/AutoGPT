from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
import pytest_mock
from cryptography.fernet import Fernet

from backend.api.features.experts import credentials as grants
from backend.copilot.credential_selection import CredentialPin, set_turn_credential_pins
from backend.copilot.tools import expert_scope, run_agent
from backend.copilot.tools.models import SetupRequirementsResponse
from backend.data import user as user_data
from backend.data.model import CredentialsFieldInfo
from backend.executor import utils as executor
from backend.integrations.credentials_store import IntegrationCredentialsStore
from backend.util import encryption
from backend.util.exceptions import ExpertNotFoundError

SECRET = "credential-snapshot-test-secret"


@pytest.fixture
def storage(mocker: pytest_mock.MockFixture):
    mocker.patch.object(encryption, "ENCRYPTION_KEY", Fernet.generate_key().decode())
    rows = _credential_rows()
    credential_rows = AsyncMock(return_value=rows)
    mocker.patch.object(
        user_data,
        "prisma",
        SimpleNamespace(
            integrationcredential=SimpleNamespace(find_many=credential_rows)
        ),
    )
    mocker.patch.object(
        IntegrationCredentialsStore,
        "db_manager",
        new_callable=PropertyMock,
        return_value=user_data,
    )
    mocker.patch.object(expert_scope, "experts_db", return_value=grants)
    mocker.patch("backend.data.db_accessors.experts_db", return_value=grants)
    mocker.patch.object(executor, "get_experts_db", return_value=grants)
    expert = SimpleNamespace(credentialsSeededAt=datetime.now(UTC))
    owned_expert = AsyncMock(return_value=expert)
    grant_rows = AsyncMock(return_value=[])
    mocker.patch.object(
        grants.prisma.models.Expert,
        "prisma",
        return_value=SimpleNamespace(find_first=owned_expert),
    )
    mocker.patch.object(
        grants.prisma.models.ExpertCredential,
        "prisma",
        return_value=SimpleNamespace(find_many=grant_rows),
    )
    mocker.patch(
        "backend.copilot.credential_selection.get_redis_async",
        AsyncMock(return_value=SimpleNamespace(hgetall=AsyncMock(return_value={}))),
    )
    set_turn_credential_pins(None)
    yield SimpleNamespace(
        rows=rows, credentials=credential_rows, expert=owned_expert, grants=grant_rows
    )
    set_turn_credential_pins(None)


@pytest.mark.parametrize("unattended", [False, True])
async def test_card_and_hint_use_one_owner_scoped_storage_snapshot(
    storage, mocker: pytest_mock.MockFixture, caplog, unattended: bool
):
    storage.credentials.side_effect = [storage.rows, storage.rows, []]
    mocker.patch.object(run_agent, "is_unattended_turn", return_value=unattended)

    matched, response = await run_agent.RunAgentTool()._check_prerequisites(
        _graph(), "owner", run_agent.RunAgentInput(), "session", "expert"
    )

    assert matched == {}
    assert response is not None
    if not unattended:
        assert isinstance(response, SetupRequirementsResponse)
        candidates = response.setup_info.user_readiness.missing_credentials[
            "credentials"
        ]["expert_grant"]["credentials"]
        assert [candidate["id"] for candidate in candidates] == ["work", "personal"]
    assert "credential_id=work" in response.message
    assert "credential_id=personal" in response.message
    assert "other-host" not in response.message
    assert SECRET not in response.model_dump_json()
    assert SECRET not in caplog.text
    storage.credentials.assert_awaited_once_with(
        where={"ownerType": "USER", "ownerId": "owner", "status": "active"},
        order={"createdAt": "asc"},
    )
    assert storage.expert.await_count == 1
    storage.grants.assert_awaited_once_with(
        where={"expertId": "expert"}, order={"createdAt": "asc"}
    )


async def test_foreign_expert_cannot_turn_snapshot_into_personal_access(storage):
    storage.expert.return_value = None

    with pytest.raises(ExpertNotFoundError):
        await run_agent.RunAgentTool()._check_prerequisites(
            _graph(), "owner", run_agent.RunAgentInput(), "session", "foreign-expert"
        )

    assert storage.expert.await_args.kwargs["where"] == {
        "id": "foreign-expert",
        "ownerUserId": "owner",
        "isTemplate": False,
        "isArchived": False,
    }
    storage.grants.assert_not_awaited()


async def test_grant_id_cannot_import_another_owners_credential(storage):
    storage.grants.return_value = [SimpleNamespace(credentialId="foreign-key")]
    matched, response = await run_agent.RunAgentTool()._check_prerequisites(
        _graph(), "owner", run_agent.RunAgentInput(), "session", "expert"
    )
    assert matched == {}
    assert response is not None
    assert "foreign-key" not in response.model_dump_json()


async def test_execution_rechecks_grants_after_successful_snapshot_match(storage):
    storage.grants.side_effect = [[SimpleNamespace(credentialId="work")], []]
    matched, response = await run_agent.RunAgentTool()._check_prerequisites(
        _graph(),
        "owner",
        run_agent.RunAgentInput(use_defaults=True),
        "session",
        "expert",
    )
    assert response is None
    assert matched["credentials"].id == "work"

    with pytest.raises(ValueError, match="has not been given access"):
        await executor._enforce_expert_credential_scope("owner", "expert", matched)

    assert storage.credentials.await_count == 1
    assert storage.expert.await_count == 2
    assert storage.grants.await_count == 2


@pytest.mark.parametrize("pin", ["deleted", "ungranted", "granted"])
async def test_snapshot_keeps_host_pins_and_other_host_credentials(storage, pin: str):
    storage.grants.return_value = [
        SimpleNamespace(credentialId="personal"),
        SimpleNamespace(credentialId="other-host"),
    ]
    if pin == "deleted":
        storage.credentials.return_value = storage.rows[1:]
    if pin == "granted":
        storage.grants.return_value.append(SimpleNamespace(credentialId="work"))
    set_turn_credential_pins({"http:api.example.test": CredentialPin(id="work")})

    matched, response = await run_agent.RunAgentTool()._check_prerequisites(
        _graph(),
        "owner",
        run_agent.RunAgentInput(use_defaults=True),
        "session",
        "expert",
    )
    if pin == "granted":
        assert response is None
        assert matched["credentials"].id == "work"
    else:
        assert matched == {}
        assert response is not None

    other, response = await run_agent.RunAgentTool()._check_prerequisites(
        _graph("other.example.test"),
        "owner",
        run_agent.RunAgentInput(use_defaults=True),
        "session",
        "expert",
    )
    assert response is None
    assert other["credentials"].id == "other-host"
    assert storage.credentials.await_count == 2
    assert storage.grants.await_count == 2


async def test_next_operation_refreshes_revoked_credentials(storage):
    storage.grants.side_effect = [[SimpleNamespace(credentialId="work")], []]
    for permitted in [True, False]:
        matched, response = await run_agent.RunAgentTool()._check_prerequisites(
            _graph(),
            "owner",
            run_agent.RunAgentInput(use_defaults=True),
            "session",
            "expert",
        )
        assert bool(matched) is permitted
        assert (response is None) is permitted
    assert storage.credentials.await_count == 2
    assert storage.grants.await_count == 2


async def test_unattended_pin_diagnostic_retains_its_fresh_scope_read(
    storage, mocker: pytest_mock.MockFixture
):
    storage.grants.return_value = [SimpleNamespace(credentialId="personal")]
    set_turn_credential_pins(
        {"http:api.example.test": CredentialPin(id="work", title="Work account")}
    )
    mocker.patch.object(run_agent, "is_unattended_turn", return_value=True)

    matched, response = await run_agent.RunAgentTool()._check_prerequisites(
        _graph(), "owner", run_agent.RunAgentInput(), "session", "expert"
    )

    assert matched == {}
    assert response is not None
    assert response.type == "error"
    assert "Work account" in response.message
    assert "personal" not in response.message
    assert SECRET not in response.model_dump_json()
    assert storage.credentials.await_count == 2
    assert storage.grants.await_count == 2


def _credential_rows():
    cryptor = encryption.JSONCryptor()
    return [
        SimpleNamespace(
            id=credential_id,
            encryptedPayload=cryptor.encrypt(
                {
                    "id": credential_id,
                    "provider": "http",
                    "type": "host_scoped",
                    "host": host,
                    "title": credential_id,
                    "headers": {"Authorization": SECRET},
                }
            ),
        )
        for credential_id, host in [
            ("work", "api.example.test"),
            ("personal", "api.example.test"),
            ("other-host", "other.example.test"),
        ]
    ]


def _graph(host: str = "api.example.test") -> MagicMock:
    field = CredentialsFieldInfo.model_validate(
        {
            "credentials_provider": ["http"],
            "credentials_types": ["host_scoped"],
            "discriminator": "url",
            "discriminator_values": [f"https://{host}/v1"],
        },
        by_alias=True,
    )
    graph = MagicMock(id="graph", version=1)
    graph.name = "Test workflow"
    graph.input_schema = {}
    graph.trigger_setup_info = None
    graph.regular_credentials_inputs = {
        "credentials": (field, {("node", "credentials")}, True)
    }
    return graph
