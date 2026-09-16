from unittest.mock import MagicMock, patch

from backend.util import funnel_analytics
from backend.util.funnel_analytics import emit_funnel_event


def test_captures_to_posthog_with_the_user_as_distinct_id():
    client = MagicMock()
    with patch.object(funnel_analytics, "get_posthog_client", return_value=client):
        emit_funnel_event("user-1", "hire_completed", {"template_id": "tpl-1"})

    client.capture.assert_called_once_with(
        event="hire_completed",
        distinct_id="user-1",
        properties={"template_id": "tpl-1"},
    )


def test_a_data_index_becomes_the_posthog_dedup_key():
    client = MagicMock()
    with patch.object(funnel_analytics, "get_posthog_client", return_value=client):
        emit_funnel_event(
            "user-1",
            "briefing_delivered",
            {"briefing_id": "b-1"},
            "briefing_delivered:b-1",
        )

    assert client.capture.call_args.kwargs["properties"] == {
        "briefing_id": "b-1",
        "$insert_id": "briefing_delivered:b-1",
    }


def test_no_data_index_means_no_dedup_key():
    client = MagicMock()
    with patch.object(funnel_analytics, "get_posthog_client", return_value=client):
        emit_funnel_event("user-1", "expert_fired", {"expert_id": "e-1"})

    assert "$insert_id" not in client.capture.call_args.kwargs["properties"]


def test_breadcrumb_carries_the_event_and_its_payload():
    client = MagicMock()
    with (
        patch.object(funnel_analytics, "get_posthog_client", return_value=client),
        patch.object(funnel_analytics.sentry_sdk, "add_breadcrumb") as crumb,
    ):
        emit_funnel_event("user-1", "hire_failed", {"template_id": "tpl-1"})

    crumb.assert_called_once_with(
        category="funnel",
        message="hire_failed",
        data={"template_id": "tpl-1"},
        level="info",
    )


def test_breadcrumb_is_added_before_the_capture():
    """It has to land in the caller's scope, ahead of whatever may raise next."""
    calls = []
    client = MagicMock()
    client.capture.side_effect = lambda **_: calls.append("capture")
    with (
        patch.object(funnel_analytics, "get_posthog_client", return_value=client),
        patch.object(
            funnel_analytics.sentry_sdk,
            "add_breadcrumb",
            side_effect=lambda **_: calls.append("breadcrumb"),
        ),
    ):
        emit_funnel_event("user-1", "expert_thread_created", {"expert_id": "e-1"})

    assert calls == ["breadcrumb", "capture"]


def test_a_disabled_posthog_still_leaves_a_breadcrumb():
    with (
        patch.object(funnel_analytics, "get_posthog_client", return_value=None),
        patch.object(funnel_analytics.sentry_sdk, "add_breadcrumb") as crumb,
    ):
        emit_funnel_event("user-1", "expert_fired", {"expert_id": "e-1"})

    crumb.assert_called_once()


def test_a_failing_capture_never_raises_into_the_caller():
    client = MagicMock()
    client.capture.side_effect = RuntimeError("posthog unreachable")
    with patch.object(funnel_analytics, "get_posthog_client", return_value=client):
        emit_funnel_event("user-1", "hire_completed", {"template_id": "tpl-1"})


def test_a_failing_client_lookup_never_raises_into_the_caller():
    with patch.object(
        funnel_analytics,
        "get_posthog_client",
        side_effect=RuntimeError("settings unreadable"),
    ):
        emit_funnel_event("user-1", "hire_completed", {"template_id": "tpl-1"})


def test_the_payload_is_copied_not_shared_with_the_caller():
    client = MagicMock()
    payload = {"expert_id": "e-1"}
    with patch.object(funnel_analytics, "get_posthog_client", return_value=client):
        emit_funnel_event("user-1", "expert_fired", payload, "expert_fired:e-1")

    assert payload == {"expert_id": "e-1"}
