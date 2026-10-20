import io
from unittest.mock import AsyncMock

from PIL import Image

from backend.copilot.baseline import service
from backend.copilot.baseline.continuation_note_test import _sent_messages
from backend.copilot.baseline.service_cancellation_test import baseline_io
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.tool_image import prepare_image

__all__ = ["baseline_io"]


async def test_requeued_approved_image_survives_turn_start_budget(
    monkeypatch, baseline_io
):
    output = io.BytesIO()
    Image.new("RGB", (20, 20), "blue").save(output, format="PNG")
    image = prepare_image(output.getvalue())
    session = ChatSession.new("user-1", dry_run=False)
    session.title = "Image inspection"
    session.messages = [
        ChatMessage(role="user", content="Inspect the page", sequence=0)
    ]
    pending = PendingMessage(content="Approved image", model_image=image)
    monkeypatch.setattr(
        service, "drain_pending_safe", AsyncMock(return_value=[pending])
    )
    monkeypatch.setattr(
        service, "persist_pending_as_user_rows", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        service, "build_turn_budget_block", AsyncMock(return_value="Live budget: 10.\n")
    )
    messages = await _sent_messages(monkeypatch, session, message=None)
    assert isinstance(messages[-1]["content"], list)
    assert messages[-1]["content"][0] == {"type": "text", "text": "Live budget: 10.\n"}
    assert messages[-1]["content"][-1] == image.openai()
    assert image.data_base64 not in session.model_dump_json()
