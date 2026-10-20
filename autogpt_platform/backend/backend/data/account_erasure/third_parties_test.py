"""PostHog and MailerLite: skipped with a reason when not configured, looked
up only on a dry run, deleted on an apply, and a no-op once gone."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data.account_erasure import third_parties
from backend.data.account_erasure.models import StepStatus, Subject

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"
SUBJECT = Subject(user_id=USER_ID, email="sam@example.com")


def _response(status: int, body: dict | None = None) -> MagicMock:
    response = MagicMock(status=status)
    response.json.return_value = body or {}
    return response


@pytest.fixture
def posthog(monkeypatch):
    secrets = SimpleNamespace(
        posthog_personal_api_key="phx_key",
        posthog_project_id="123",
        posthog_app_host="https://eu.posthog.com",
    )
    monkeypatch.setattr(
        third_parties, "Settings", lambda: SimpleNamespace(secrets=secrets)
    )
    client = MagicMock()
    client.get = AsyncMock(return_value=_response(200, {"results": [{"id": "p1"}]}))
    client.post = AsyncMock(return_value=_response(200))
    monkeypatch.setattr(third_parties, "_posthog", lambda: client)
    return SimpleNamespace(secrets=secrets, client=client)


@pytest.mark.asyncio
async def test_posthog_is_skipped_without_a_personal_key(posthog):
    posthog.secrets.posthog_personal_api_key = ""

    report = await third_parties.delete_posthog_person(SUBJECT, dry_run=False)

    assert report.status == StepStatus.SKIPPED
    assert "POSTHOG_PERSONAL_API_KEY" in report.note
    posthog.client.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_posthog_dry_run_only_looks_the_person_up(posthog):
    report = await third_parties.delete_posthog_person(SUBJECT, dry_run=True)

    assert report.counts == {"persons": 1}
    assert f"distinct_id={USER_ID}" in posthog.client.get.await_args.args[0]
    posthog.client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_posthog_person_is_deleted_with_events_and_recordings(posthog):
    report = await third_parties.delete_posthog_person(SUBJECT, dry_run=False)

    url = posthog.client.post.await_args.args[0]
    body = posthog.client.post.await_args.kwargs["json"]
    assert url == "https://eu.posthog.com/api/projects/123/persons/bulk_delete/"
    assert body == {
        "distinct_ids": [USER_ID],
        "delete_events": True,
        "delete_recordings": True,
    }
    assert report.status == StepStatus.DONE


@pytest.mark.asyncio
async def test_no_posthog_person_left_is_a_no_op(posthog):
    posthog.client.get = AsyncMock(return_value=_response(200, {"results": []}))

    report = await third_parties.delete_posthog_person(SUBJECT, dry_run=False)

    assert report.counts == {"persons": 0}
    posthog.client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_posthog_error_fails_the_step(posthog):
    posthog.client.post = AsyncMock(return_value=_response(403))

    with pytest.raises(RuntimeError, match="403"):
        await third_parties.delete_posthog_person(SUBJECT, dry_run=False)


@pytest.fixture
def mailerlite(monkeypatch):
    fake = SimpleNamespace(
        configured=MagicMock(return_value=True),
        find_subscriber_id=AsyncMock(return_value="ml_1"),
        forget_subscriber=AsyncMock(return_value=True),
    )
    monkeypatch.setattr(third_parties, "mailerlite", fake)
    return fake


@pytest.mark.asyncio
async def test_mailerlite_is_skipped_without_a_token(mailerlite):
    mailerlite.configured.return_value = False

    report = await third_parties.forget_mailerlite_subscriber(SUBJECT, dry_run=False)

    assert report.status == StepStatus.SKIPPED
    mailerlite.forget_subscriber.assert_not_awaited()


@pytest.mark.asyncio
async def test_mailerlite_without_an_email_is_skipped_not_guessed(mailerlite):
    report = await third_parties.forget_mailerlite_subscriber(
        Subject(user_id=USER_ID), dry_run=False
    )

    assert report.status == StepStatus.SKIPPED
    mailerlite.find_subscriber_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_mailerlite_dry_run_only_looks_up(mailerlite):
    report = await third_parties.forget_mailerlite_subscriber(SUBJECT, dry_run=True)

    assert report.counts == {"subscribers": 1}
    mailerlite.forget_subscriber.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_subscriber_is_forgotten_on_apply(mailerlite):
    report = await third_parties.forget_mailerlite_subscriber(SUBJECT, dry_run=False)

    mailerlite.forget_subscriber.assert_awaited_once_with("sam@example.com")
    assert report.counts == {"subscribers": 1}
    assert "sam@example.com" not in report.model_dump_json()
