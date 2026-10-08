from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.copilot.heartbeat import delivery
from backend.copilot.heartbeat.config import HeartbeatDelivery


async def _deliver(targets: HeartbeatDelivery, *, configured: bool = True):
    chats = AsyncMock()
    chats.append_plain_session_message.return_value = "main-thread"
    bus = MagicMock(publish=AsyncMock(), publish_event=AsyncMock())
    bridge = AsyncMock()
    bridge.send_dm_to_user.return_value = SimpleNamespace(ok=True, error=None)
    with (
        patch.object(delivery, "chat_db", return_value=chats),
        patch.object(delivery, "AsyncRedisNotificationEventBus", return_value=bus),
        patch(
            "backend.copilot.tools.chat_platform._any_chat_platform_configured",
            return_value=configured,
        ),
        patch(
            "backend.util.clients.get_copilot_chat_bridge_client",
            return_value=bridge,
        ),
    ):
        report = await delivery.deliver_alert("u", "hb", "Sync failed.", targets)
    return report, chats, bus, bridge


async def test_an_alert_lands_in_the_main_thread_and_notifies():
    report, chats, bus, bridge = await _deliver(HeartbeatDelivery())
    assert report.thread_session_id == "main-thread"
    kwargs = chats.append_plain_session_message.await_args.kwargs
    assert kwargs["content"] == "Sync failed."
    assert kwargs["metadata"]["kind"] == delivery.ALERT_MESSAGE_KIND
    event = bus.publish.await_args.args[0]
    payload = event.payload.model_dump()
    assert payload["type"] == "copilot_completion"
    assert payload["session_id"] == "main-thread"
    assert payload["source"] == "heartbeat"
    bridge.send_dm_to_user.assert_not_awaited()


async def test_a_retried_delivery_posts_the_same_message_id():
    _, first, _, _ = await _deliver(HeartbeatDelivery())
    _, second, _, _ = await _deliver(HeartbeatDelivery())
    assert (
        first.append_plain_session_message.await_args.kwargs["message_id"]
        == second.append_plain_session_message.await_args.kwargs["message_id"]
    )


async def test_push_off_still_notifies_the_open_app():
    report, _, bus, _ = await _deliver(HeartbeatDelivery(push=False))
    assert report.notified
    bus.publish.assert_not_awaited()
    bus.publish_event.assert_awaited_once()


async def test_without_the_thread_the_notification_opens_the_heartbeat_chat():
    report, chats, bus, _ = await _deliver(HeartbeatDelivery(copilot_thread=False))
    chats.append_plain_session_message.assert_not_awaited()
    assert bus.publish.await_args.args[0].payload.model_dump()["session_id"] == "hb"
    assert report.thread_session_id is None


async def test_linked_chat_platforms_get_a_dm():
    report, _, _, bridge = await _deliver(
        HeartbeatDelivery(chat_platforms=["discord", "slack"])
    )
    assert report.chat_platforms == ["discord", "slack"]
    assert bridge.send_dm_to_user.await_count == 2
    assert bridge.send_dm_to_user.await_args.kwargs["user_id"] == "u"


async def test_no_bot_configured_skips_the_chat_platforms():
    report, _, _, bridge = await _deliver(
        HeartbeatDelivery(chat_platforms=["discord"]), configured=False
    )
    assert report.chat_platforms == []
    bridge.send_dm_to_user.assert_not_awaited()


async def test_a_failed_thread_post_still_notifies():
    chats = AsyncMock()
    chats.append_plain_session_message.side_effect = RuntimeError("db down")
    bus = MagicMock(publish=AsyncMock(), publish_event=AsyncMock())
    with (
        patch.object(delivery, "chat_db", return_value=chats),
        patch.object(delivery, "AsyncRedisNotificationEventBus", return_value=bus),
    ):
        report = await delivery.deliver_alert("u", "hb", "x", HeartbeatDelivery())
    assert report.thread_session_id is None
    assert report.notified
