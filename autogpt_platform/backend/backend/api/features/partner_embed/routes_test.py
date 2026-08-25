from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from backend.api.features.partner_embed import routes as embed_routes
from backend.api.features.partner_embed.auth import EmbedPrincipal
from backend.api.features.partner_embed.models import ProvisionPartnerIdentityResponse

app = FastAPI()
app.include_router(embed_routes.router)
client = TestClient(app, raise_server_exceptions=False)

PRINCIPAL = EmbedPrincipal(
    user_id="0234dc86-e049-5c61-8b7e-826f7a7c225f",
    partner_id="forwarding-digital",
    organization_id="70d89c3b-2af3-5f56-8a21-2951b469ba95",
    team_id="600e3708-3a7a-54c7-b527-53d2c62b8d5b",
    external_account_id="forwarder-42",
    capabilities=["jobs.read", "reports.read"],
    scopes=["embed:chat"],
)

PROVISION_BODY = {
    "partner_id": "forwarding-digital",
    "external_subject": "user-123",
    "external_account_id": "forwarder-42",
    "display_name": "Jon Heavyside",
    "account_name": "Acme Forwarding",
    "is_admin": True,
}


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    """Route unit tests mock backend services and do not need SpinTestServer."""
    yield


@pytest.fixture(autouse=True)
def dependency_overrides():
    app.dependency_overrides[embed_routes.requires_partner_provisioning_service] = (
        lambda: None
    )
    app.dependency_overrides[embed_routes.requires_embed_chat] = lambda: PRINCIPAL
    yield
    app.dependency_overrides.clear()


def test_provision_route_returns_only_internal_mapping(mocker):
    provision = mocker.patch(
        "backend.api.features.partner_embed.routes.provision_partner_identity",
        return_value=ProvisionPartnerIdentityResponse(
            user_id=PRINCIPAL.user_id,
            organization_id=PRINCIPAL.organization_id,
            team_id=PRINCIPAL.team_id or "",
        ),
    )

    response = client.post("/api/embed/v1/provision", json=PROVISION_BODY)

    assert response.status_code == 200
    assert response.json() == {
        "user_id": PRINCIPAL.user_id,
        "organization_id": PRINCIPAL.organization_id,
        "team_id": PRINCIPAL.team_id,
    }
    provision.assert_awaited_once()


def test_create_session_uses_token_locked_tenancy(mocker):
    resolve_route = mocker.patch(
        "backend.api.features.partner_embed.routes.resolve_embed_llm_route",
        return_value=("codex", "credential-1"),
    )
    create = mocker.patch(
        "backend.api.features.partner_embed.routes.create_chat_session",
        return_value=SimpleNamespace(
            session_id="session-1",
            started_at=SimpleNamespace(isoformat=lambda: "2026-08-24T12:00:00+00:00"),
        ),
    )

    response = client.post("/api/embed/v1/sessions")

    assert response.status_code == 201
    assert response.json() == {
        "id": "session-1",
        "created_at": "2026-08-24T12:00:00+00:00",
    }
    assert create.await_args.kwargs == {
        "dry_run": False,
        "organization_id": PRINCIPAL.organization_id,
        "team_id": PRINCIPAL.team_id,
        "source_platform": PRINCIPAL.partner_id,
        "external_account_id": PRINCIPAL.external_account_id,
        "external_capabilities": PRINCIPAL.capabilities,
        "llm_auth_provider": "codex",
        "llm_credential_id": "credential-1",
    }
    assert create.await_args.args == (PRINCIPAL.user_id,)
    resolve_route.assert_awaited_once_with(PRINCIPAL.user_id)


def test_stream_rejects_a_session_from_another_customer_account(mocker):
    mocker.patch(
        "backend.api.features.partner_embed.routes.get_chat_session_metadata",
        return_value=SimpleNamespace(
            organization_id="another-org",
            team_id=PRINCIPAL.team_id,
            metadata=SimpleNamespace(
                source_platform=PRINCIPAL.partner_id,
                external_account_id=PRINCIPAL.external_account_id,
            ),
        ),
    )
    stream = mocker.patch("backend.api.features.partner_embed.routes.stream_chat_post")

    response = client.post(
        "/api/embed/v1/sessions/session-1/stream",
        json={"message": "Summarize today's shipments"},
    )

    assert response.status_code == 404
    stream.assert_not_awaited()


def test_stream_rejects_a_session_from_another_external_account(mocker):
    mocker.patch(
        "backend.api.features.partner_embed.routes.get_chat_session_metadata",
        return_value=SimpleNamespace(
            organization_id=PRINCIPAL.organization_id,
            team_id=PRINCIPAL.team_id,
            metadata=SimpleNamespace(
                source_platform=PRINCIPAL.partner_id,
                external_account_id="another-account",
            ),
        ),
    )
    stream = mocker.patch("backend.api.features.partner_embed.routes.stream_chat_post")

    response = client.post(
        "/api/embed/v1/sessions/session-1/stream",
        json={"message": "Summarize today's shipments"},
    )

    assert response.status_code == 404
    stream.assert_not_awaited()


def test_stream_rejects_a_session_from_another_team(mocker):
    mocker.patch(
        "backend.api.features.partner_embed.routes.get_chat_session_metadata",
        return_value=SimpleNamespace(
            organization_id=PRINCIPAL.organization_id,
            team_id="another-team",
            metadata=SimpleNamespace(
                source_platform=PRINCIPAL.partner_id,
                external_account_id=PRINCIPAL.external_account_id,
                external_capabilities=PRINCIPAL.capabilities,
            ),
        ),
    )
    stream = mocker.patch("backend.api.features.partner_embed.routes.stream_chat_post")

    response = client.post(
        "/api/embed/v1/sessions/session-1/stream",
        json={"message": "Summarize today's shipments"},
    )

    assert response.status_code == 404
    stream.assert_not_awaited()


def test_stream_forwards_locked_identity_to_autopilot(mocker):
    mocker.patch(
        "backend.api.features.partner_embed.routes.get_chat_session_metadata",
        return_value=SimpleNamespace(
            organization_id=PRINCIPAL.organization_id,
            team_id=PRINCIPAL.team_id,
            metadata=SimpleNamespace(
                source_platform=PRINCIPAL.partner_id,
                external_account_id=PRINCIPAL.external_account_id,
                external_capabilities=PRINCIPAL.capabilities,
            ),
        ),
    )

    async def body():
        yield b'data: {"type":"finish"}\n\n'

    stream = mocker.patch(
        "backend.api.features.partner_embed.routes.stream_chat_post",
        return_value=StreamingResponse(body(), media_type="text/event-stream"),
    )

    response = client.post(
        "/api/embed/v1/sessions/session-1/stream",
        json={"message": "Summarize today's shipments", "message_id": "turn-1"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert stream.await_args.args[0] == "session-1"
    assert stream.await_args.args[1].model_dump() == {
        "message": "Summarize today's shipments",
        "is_user_message": True,
        "context": None,
        "file_ids": None,
        "mode": None,
        "model": None,
        "message_id": "turn-1",
        "expert_kickoff": False,
    }
    assert stream.await_args.kwargs["user_id"] == PRINCIPAL.user_id
    context = stream.await_args.kwargs["ctx"]
    assert context.org_id == PRINCIPAL.organization_id
    assert context.team_id == PRINCIPAL.team_id


def test_title_fallback_is_scoped_and_derived_from_first_message(mocker):
    mocker.patch(
        "backend.api.features.partner_embed.routes.get_chat_session_metadata",
        return_value=SimpleNamespace(
            title=None,
            organization_id=PRINCIPAL.organization_id,
            team_id=PRINCIPAL.team_id,
            metadata=SimpleNamespace(
                source_platform=PRINCIPAL.partner_id,
                external_account_id=PRINCIPAL.external_account_id,
                external_capabilities=PRINCIPAL.capabilities,
            ),
        ),
    )
    update = mocker.patch(
        "backend.api.features.partner_embed.routes.update_session_title",
        return_value=True,
    )

    response = client.patch(
        "/api/embed/v1/sessions/session-1/title",
        json={"message": "Compare the active shipment lanes and flag the highest risk"},
    )

    assert response.status_code == 200
    assert response.json() == {"title": "Compare the active shipment lanes and..."}
    update.assert_awaited_once_with(
        "session-1",
        PRINCIPAL.user_id,
        "Compare the active shipment lanes and...",
        only_if_empty=True,
    )


def test_title_fallback_rejects_a_session_from_another_tenant(mocker):
    mocker.patch(
        "backend.api.features.partner_embed.routes.get_chat_session_metadata",
        return_value=SimpleNamespace(
            title=None,
            organization_id="another-org",
            team_id=PRINCIPAL.team_id,
            metadata=SimpleNamespace(
                source_platform=PRINCIPAL.partner_id,
                external_account_id=PRINCIPAL.external_account_id,
                external_capabilities=PRINCIPAL.capabilities,
            ),
        ),
    )
    update = mocker.patch(
        "backend.api.features.partner_embed.routes.update_session_title"
    )

    response = client.patch(
        "/api/embed/v1/sessions/session-1/title",
        json={"message": "Summarize today's shipments"},
    )

    assert response.status_code == 404
    update.assert_not_awaited()
