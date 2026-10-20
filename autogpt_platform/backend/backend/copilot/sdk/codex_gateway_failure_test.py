from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from aiohttp import ClientSession
from openai import AsyncOpenAI, RateLimitError
from pydantic import SecretStr

from backend.copilot.provider_failure import ProviderFailureKind
from backend.copilot.sdk.codex_compat_gateway import (
    CodexAnthropicGateway,
    _Conversation,
    _Failed,
    _TextDelta,
    gateway_for_lease,
)
from backend.data.model import OAuth2Credentials
from backend.integrations.codex.http_session import CodexHttpSession
from backend.integrations.codex.transport import (
    CodexCredentialIntegrityError,
    CodexCredentialLease,
    CodexTransportError,
    CodexTransportOverloadedError,
)

_RESETS_AT = 1788452827


def _spent_quota(_request: httpx.Request) -> httpx.Response:
    """What ChatGPT answers once a subscription's quota is spent."""
    return httpx.Response(
        429,
        headers={
            "x-codex-primary-used-percent": "100",
            "x-codex-primary-window-minutes": "300",
            "x-codex-primary-reset-at": "1788400000",
        },
        json={
            "error": {
                "type": "usage_limit_reached",
                "message": "The usage limit has been reached",
                "resets_at": _RESETS_AT,
            }
        },
    )


def _gateway() -> CodexAnthropicGateway:
    return CodexAnthropicGateway(
        agent_session=object(),  # type: ignore[arg-type]
        model="gpt-5.6-luna",
        transport=object(),  # type: ignore[arg-type]
    )


class TestTheGatewayNamesItsFailures:
    """A blanket 502 tells the CLI to try again. Most failures cannot."""

    @pytest.mark.parametrize(
        "exc, status, kind",
        [
            (
                CodexCredentialIntegrityError("bad"),
                401,
                ProviderFailureKind.INVALID_CREDENTIAL,
            ),
            (CodexTransportOverloadedError("busy"), 503, ProviderFailureKind.TRANSIENT),
        ],
    )
    def test_a_named_failure_gets_a_status_that_means_something(
        self, exc, status, kind
    ) -> None:
        gw = _gateway()
        response = gw._failure_response(_Failed(error=exc))
        assert response.status == status
        assert gw.last_failure is not None
        assert gw.last_failure.kind is kind

    def test_an_unrecognised_failure_keeps_the_honest_generic(self) -> None:
        # A wrong specific status would be worse than a generic one.
        gw = _gateway()
        response = gw._failure_response(_Failed(error=CodexTransportError("?")))
        assert response.status == 502
        assert gw.last_failure is None

    def test_the_failure_is_left_where_the_service_layer_can_read_it(self) -> None:
        # By the time this reaches the CLI it is a status and a sentence;
        # the typed exception only exists inside the gateway.
        gw = _gateway()
        assert gw.last_failure is None
        gw._failure_response(_Failed(error=CodexCredentialIntegrityError("bad")))
        assert gw.last_failure is not None
        assert gw.last_failure.auth_provider == "codex"

    @pytest.mark.asyncio
    async def test_a_midstream_failure_is_classified_for_the_service(self) -> None:
        gw = _gateway()
        conversation = _Conversation(id="conversation-1")
        await conversation.queue.put(
            _Failed(error=CodexCredentialIntegrityError("expired credential"))
        )
        response = MagicMock()
        response.write = AsyncMock()

        await gw._write_boundary(response, conversation, _TextDelta(text="partial"))

        assert gw.last_failure is not None
        assert gw.last_failure.kind is ProviderFailureKind.INVALID_CREDENTIAL
        wire = b"".join(call.args[0] for call in response.write.await_args_list)
        assert b'"type":"invalid_credential"' in wire


class TestASpentChatGPTQuota:
    """The switch offer needs the account; the limit card needs the reset.

    Without the account the client hides every ChatGPT connection, since it
    cannot tell which one refused, and drops the offer on reload. Without the
    reset it can only say "wait", not for how long.
    """

    @pytest.mark.asyncio
    async def test_names_the_account_and_when_it_lifts(self) -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_spent_quota)
        ) as http:
            session = CodexHttpSession(
                OAuth2Credentials(
                    provider="codex",
                    title="ChatGPT",
                    access_token=SecretStr("access"),
                    scopes=[],
                ),
                turn_timeout_seconds=30,
                tool_timeout_seconds=5,
                client=AsyncOpenAI(
                    api_key="test",
                    base_url="https://chatgpt.test",
                    http_client=http,
                    max_retries=0,
                ),
            )
            async with CodexAnthropicGateway(
                agent_session=session,
                credential_id="cred-chatgpt-2",
                model="gpt-5.6-terra",
            ) as gateway:
                async with ClientSession() as client:
                    response = await client.post(
                        f"{gateway.base_url}/v1/messages",
                        headers={"Authorization": f"Bearer {gateway.auth_token}"},
                        json={
                            "stream": True,
                            "messages": [{"role": "user", "content": "hello"}],
                        },
                    )
                    await response.read()

        assert response.status == 429
        failure = gateway.last_failure
        assert failure is not None
        assert failure.kind is ProviderFailureKind.USAGE_LIMIT
        part = failure.as_part()
        assert part["authProvider"] == "codex"
        assert part["credentialId"] == "cred-chatgpt-2"
        assert part["resetsAt"] == _RESETS_AT

    def test_a_gateway_built_for_a_lease_names_its_account(self) -> None:
        # The service builds the gateway from the lease as an agent session,
        # which carries no credential of its own to name.
        credentials = OAuth2Credentials(
            id="cred-chatgpt-1",
            provider="codex",
            title="ChatGPT",
            access_token=SecretStr("access"),
            scopes=[],
        )
        gateway = gateway_for_lease(
            CodexCredentialLease(credentials, MagicMock()),
            model="gpt-5.6-terra",
            effort="high",
        )
        limit = RateLimitError(
            "usage limit",
            response=httpx.Response(
                429, request=httpx.Request("POST", "https://chatgpt.test")
            ),
            body=None,
        )

        gateway._failure_response(_Failed(error=limit))

        assert gateway.last_failure is not None
        assert gateway.last_failure.credential_id == "cred-chatgpt-1"
