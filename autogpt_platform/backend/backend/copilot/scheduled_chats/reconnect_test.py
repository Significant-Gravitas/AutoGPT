import asyncio

import pytest

from backend.copilot.scheduled_chats import reconnect
from backend.copilot.scheduled_chats.outcomes import classify_run, tool_failure_detail
from backend.copilot.tools.expert_scope import provider_slug
from backend.copilot.tools.helpers import _single_provider
from backend.copilot.tools.models import ErrorResponse
from backend.integrations.providers import ProviderName


@pytest.mark.asyncio
async def test_a_connect_does_not_wait_for_the_resume(mocker):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_resume(user_id: str, provider: str) -> None:
        started.set()
        await release.wait()

    mocker.patch.object(reconnect, "resume_after_reconnect", side_effect=slow_resume)

    task = reconnect.resume_after_reconnect_in_background("user-1", "discord")
    await started.wait()

    assert not task.done()
    release.set()
    await task


@pytest.mark.parametrize("provider", list(ProviderName))
def test_a_failing_tool_names_its_provider_as_the_connect_route_does(provider):
    """A reconnect resumes the chats whose streak names the provider just
    connected: the connect route's ProviderName value against what the
    failing tool put in ErrorResponse.details. One vocabulary, kept so."""
    output = ErrorResponse(
        message="The saved credential was refused.",
        error="credential_rejected",
        details=_single_provider({provider_slug(provider)}),
    ).model_dump_json()
    detail = tool_failure_detail(output)
    assert detail is not None

    outcome = classify_run(
        cancelled=False,
        turn_error_code=None,
        turn_errored=False,
        tool_failures=[("run_block", detail)],
        tool_successes=0,
    )

    assert outcome.failure is not None
    assert outcome.failure.kind == "connection"
    assert outcome.failure.provider == provider.value
