import uuid
from unittest.mock import MagicMock, patch

from backend.util import funnel_analytics
from backend.util.funnel_analytics import emit_funnel_event
from backend.util.posthog_events import PostHogEvent


def test_captures_to_posthog_with_the_user_as_distinct_id():
    client = MagicMock()
    with patch.object(funnel_analytics, "get_posthog_client", return_value=client):
        emit_funnel_event(
            "user-1", PostHogEvent.HIRE_COMPLETED, {"template_id": "tpl-1"}
        )

    client.capture.assert_called_once_with(
        event="hire_completed",
        distinct_id="user-1",
        properties={"template_id": "tpl-1"},
        uuid=None,
    )


def test_a_data_index_becomes_the_posthog_dedup_key():
    client = MagicMock()
    with patch.object(funnel_analytics, "get_posthog_client", return_value=client):
        emit_funnel_event(
            "user-1",
            PostHogEvent.BRIEFING_DELIVERED,
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
        emit_funnel_event("user-1", PostHogEvent.EXPERT_FIRED, {"expert_id": "e-1"})

    assert "$insert_id" not in client.capture.call_args.kwargs["properties"]


def test_breadcrumb_carries_the_event_and_its_payload():
    client = MagicMock()
    with (
        patch.object(funnel_analytics, "get_posthog_client", return_value=client),
        patch.object(funnel_analytics.sentry_sdk, "add_breadcrumb") as crumb,
    ):
        emit_funnel_event("user-1", PostHogEvent.HIRE_FAILED, {"template_id": "tpl-1"})

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
        emit_funnel_event("user-1", PostHogEvent.EXPERT_FIRED, {"expert_id": "e-1"})

    assert calls == ["breadcrumb", "capture"]


def test_a_disabled_posthog_still_leaves_a_breadcrumb():
    with (
        patch.object(funnel_analytics, "get_posthog_client", return_value=None),
        patch.object(funnel_analytics.sentry_sdk, "add_breadcrumb") as crumb,
    ):
        emit_funnel_event("user-1", PostHogEvent.EXPERT_FIRED, {"expert_id": "e-1"})

    crumb.assert_called_once()


def test_a_throwing_breadcrumb_still_lets_the_capture_through():
    """The two sinks are independent: losing Sentry must not lose PostHog."""
    client = MagicMock()
    with (
        patch.object(funnel_analytics, "get_posthog_client", return_value=client),
        patch.object(
            funnel_analytics.sentry_sdk,
            "add_breadcrumb",
            side_effect=RuntimeError("sentry unavailable"),
        ),
    ):
        emit_funnel_event(
            "user-1", PostHogEvent.HIRE_COMPLETED, {"template_id": "tpl-1"}
        )

    client.capture.assert_called_once()


def test_a_data_index_also_becomes_a_deterministic_event_uuid():
    """PostHog's ingestion keys on the event uuid, which the client otherwise
    randomises per call, so a retry needs the same one."""
    client = MagicMock()
    with patch.object(funnel_analytics, "get_posthog_client", return_value=client):
        emit_funnel_event(
            "user-1", PostHogEvent.BRIEFING_DELIVERED, {}, "briefing_delivered:b-1"
        )
        first = client.capture.call_args.kwargs["uuid"]
        emit_funnel_event(
            "user-1", PostHogEvent.BRIEFING_DELIVERED, {}, "briefing_delivered:b-1"
        )
        second = client.capture.call_args.kwargs["uuid"]

    assert first and first == second
    assert uuid.UUID(first).version == 5


def test_a_different_event_identity_gets_a_different_uuid():
    client = MagicMock()
    with patch.object(funnel_analytics, "get_posthog_client", return_value=client):
        emit_funnel_event(
            "user-1", PostHogEvent.BRIEFING_DELIVERED, {}, "briefing_delivered:b-1"
        )
        a = client.capture.call_args.kwargs["uuid"]
        emit_funnel_event(
            "user-2", PostHogEvent.BRIEFING_DELIVERED, {}, "briefing_delivered:b-1"
        )
        b = client.capture.call_args.kwargs["uuid"]

    assert a != b


def test_no_data_index_leaves_the_uuid_to_the_client():
    client = MagicMock()
    with patch.object(funnel_analytics, "get_posthog_client", return_value=client):
        emit_funnel_event("user-1", PostHogEvent.EXPERT_FIRED, {"expert_id": "e-1"})

    assert client.capture.call_args.kwargs["uuid"] is None


def test_a_failing_capture_never_raises_into_the_caller():
    client = MagicMock()
    client.capture.side_effect = RuntimeError("posthog unreachable")
    with patch.object(funnel_analytics, "get_posthog_client", return_value=client):
        emit_funnel_event(
            "user-1", PostHogEvent.HIRE_COMPLETED, {"template_id": "tpl-1"}
        )


def test_a_failing_client_lookup_never_raises_into_the_caller():
    with patch.object(
        funnel_analytics,
        "get_posthog_client",
        side_effect=RuntimeError("settings unreadable"),
    ):
        emit_funnel_event(
            "user-1", PostHogEvent.HIRE_COMPLETED, {"template_id": "tpl-1"}
        )


def test_the_payload_is_copied_not_shared_with_the_caller():
    client = MagicMock()
    payload = {"expert_id": "e-1"}
    with patch.object(funnel_analytics, "get_posthog_client", return_value=client):
        emit_funnel_event(
            "user-1", PostHogEvent.EXPERT_FIRED, payload, "expert_fired:e-1"
        )

    assert payload == {"expert_id": "e-1"}


def test_the_event_goes_out_as_its_plain_name_and_keeps_its_uuid():
    """Moving names into an enum must not change what PostHog receives: the
    name is the same plain string, and the dedup uuid derived from it matches
    the one events sent before the enum existed already carry."""
    client = MagicMock()
    with patch.object(funnel_analytics, "get_posthog_client", return_value=client):
        emit_funnel_event(
            "user-1", PostHogEvent.BRIEFING_DELIVERED, {}, "briefing_delivered:b-1"
        )

    kwargs = client.capture.call_args.kwargs
    assert type(kwargs["event"]) is str
    assert kwargs["event"] == "briefing_delivered"
    assert kwargs["uuid"] == str(
        uuid.uuid5(
            uuid.NAMESPACE_URL, "user-1:briefing_delivered:briefing_delivered:b-1"
        )
    )
