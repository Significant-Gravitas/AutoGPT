from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.provider_failure import ProviderFailureKind
from backend.copilot.sdk.codex_compat_gateway import (
    CodexAnthropicGateway,
    _Conversation,
    _Failed,
    _TextDelta,
)
from backend.integrations.codex.transport import (
    CodexCredentialIntegrityError,
    CodexTransportError,
    CodexTransportOverloadedError,
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
