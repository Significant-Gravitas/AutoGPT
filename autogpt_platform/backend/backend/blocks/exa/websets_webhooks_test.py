"""Request-building and parsing tests for the Websets webhook and event blocks."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from backend.blocks.exa._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.exa._webhook import ExaEventType
from backend.blocks.exa._websets_webhook_samples import (
    SAMPLE_ATTEMPT,
    SAMPLE_CREATED_WEBHOOK,
)
from backend.blocks.exa.websets_events import (
    ExaListWebsetEventsBlock,
    event_params,
    iso_utc,
)
from backend.blocks.exa.websets_webhook_attempts import (
    ExaListWebsetWebhookAttemptsBlock,
    attempt_params,
)
from backend.blocks.exa.websets_webhooks import (
    ExaCreateWebsetWebhookBlock,
    ExaUpdateWebsetWebhookBlock,
    build_update_body,
)
from backend.blocks.exa.websets_webhooks_models import (
    WebhookAttemptModel,
    event_type_values,
)

CREDS = {"credentials": TEST_CREDENTIALS_INPUT}


class TestEventTypes:
    def test_legacy_event_is_rejected_with_its_replacement(self):
        with pytest.raises(ValueError, match="import.completed"):
            event_type_values([ExaEventType.IMPORT_PROCESSING])

    def test_current_events_pass_through(self):
        assert event_type_values(
            [ExaEventType.MONITOR_RUN_COMPLETED, ExaEventType.WEBSET_IDLE]
        ) == ["monitor.run.completed", "webset.idle"]


class TestCreateWebhook:
    async def test_posts_url_events_and_metadata_and_returns_secret(self):
        block = ExaCreateWebsetWebhookBlock()
        block.call_exa = AsyncMock(return_value=SAMPLE_CREATED_WEBHOOK)
        inputs = block.input_schema(
            **CREDS,
            url="https://example.com/hook",
            events=[ExaEventType.WEBSET_ITEM_CREATED],
            metadata={"team": "research"},
        )
        outputs = {
            name: value
            async for name, value in block.run(inputs, credentials=TEST_CREDENTIALS)
        }
        args, kwargs = block.call_exa.call_args
        assert args[1:] == ("POST", "https://api.exa.ai/websets/v0/webhooks")
        assert kwargs["body"] == {
            "url": "https://example.com/hook",
            "events": ["webset.item.created"],
            "metadata": {"team": "research"},
        }
        assert outputs["secret"] == SAMPLE_CREATED_WEBHOOK["secret"]

    def test_needs_at_least_one_event(self):
        with pytest.raises(ValueError):
            ExaCreateWebsetWebhookBlock.Input(**CREDS, url="https://x", events=[])


class TestUpdateWebhook:
    def _input(self, **overrides):
        return ExaUpdateWebsetWebhookBlock.Input(**CREDS, webhook_id="w1", **overrides)

    def test_only_set_fields_are_sent(self):
        body = build_update_body(
            self._input(events=[ExaEventType.WEBSET_EXPORT_COMPLETED])
        )
        assert body == {"events": ["webset.export.completed"]}

    def test_nothing_to_update_is_an_error(self):
        with pytest.raises(ValueError, match="Nothing to update"):
            build_update_body(self._input(url="  "))


class TestAttempts:
    def test_filters_become_query_params(self):
        params = attempt_params(
            ExaListWebsetWebhookAttemptsBlock.Input(
                **CREDS,
                webhook_id="w1",
                event_type=ExaEventType.WEBSET_ITEM_CREATED,
                successful=False,
            )
        )
        assert params == [
            ("eventType", "webset.item.created"),
            ("successful", "false"),
            ("limit", "25"),
        ]

    def test_attempt_numbers_are_ints(self):
        attempt = WebhookAttemptModel.from_api({**SAMPLE_ATTEMPT, "attempt": 2.0})
        assert attempt.attempt == 2
        assert attempt.response_status_code == 500


class TestEvents:
    def test_types_repeat_and_dates_are_utc(self):
        params = event_params(
            ExaListWebsetEventsBlock.Input(
                **CREDS,
                types=[ExaEventType.WEBSET_IDLE, ExaEventType.MONITOR_RUN_COMPLETED],
                created_after=datetime(2026, 10, 1, 12, 0),
                created_before=datetime(
                    2026, 10, 2, 14, 0, tzinfo=timezone(timedelta(hours=2))
                ),
                limit=200,
            )
        )
        assert params == [
            ("types", "webset.idle"),
            ("types", "monitor.run.completed"),
            ("createdAfter", "2026-10-01T12:00:00Z"),
            ("createdBefore", "2026-10-02T12:00:00Z"),
            ("limit", "200"),
        ]

    def test_limit_is_capped_at_200(self):
        with pytest.raises(ValueError):
            ExaListWebsetEventsBlock.Input(**CREDS, limit=201)

    def test_no_date_means_no_param(self):
        assert iso_utc(None) is None
