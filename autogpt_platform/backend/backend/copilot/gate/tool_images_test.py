import io
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image
from prisma.enums import ReviewStatus

from backend.copilot import pending_message_helpers
from backend.copilot.gate import held
from backend.copilot.gate.content import ContentVerdict
from backend.copilot.gate.reads import trusted_read
from backend.copilot.gate.reads_test import _Fetch, _session, rows
from backend.copilot.pending_messages import (
    PendingMessage,
    format_pending_as_user_message,
)
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.copilot.sdk.tool_adapter import (
    _make_truncating_wrapper,
    cap_late_tool_result,
    create_tool_handler,
    set_execution_context,
)
from backend.copilot.tool_image import HeldImage, ToolImage, prepare_image
from backend.copilot.tools.base import BaseTool
from backend.copilot.tools.models import ToolResponseBase
from backend.copilot.transcript_builder import TranscriptBuilder

__all__ = ["rows"]


def test_wrapped_workspace_image_retains_installed_skill_trust(image):
    text = '{"type":"workspace_file_metadata","path":"/skills/example/chart.png"}'
    wrapped = HeldImage(text=text, image=image).model_dump_json()
    assert trusted_read("read_workspace_file", {"file_id": "f"}, wrapped)


class _ImageRead(_Fetch):
    def __init__(self, image: ToolImage):
        super().__init__("Image returned", name="read_workspace_file")
        self.image = image

    async def _execute(self, user_id, session, **kwargs) -> ToolResponseBase:
        return (await super()._execute(user_id, session, **kwargs)).with_image(
            self.image
        )


@pytest.fixture
def image():
    output = io.BytesIO()
    Image.effect_noise((512, 512), 100).convert("RGB").save(output, format="PNG")
    return prepare_image(output.getvalue())


async def test_approved_action_keeps_image_with_non_json_metadata(monkeypatch, image):
    session = _session()
    call = held.HeldCall(
        review_id="review-1",
        tool_name="read_workspace_file",
        tool_call_id="call-image",
        args={"file_id": "f"},
    )
    tool = AsyncMock(spec=BaseTool)
    tool.execute.return_value = StreamToolOutputAvailable(
        toolCallId=call.tool_call_id,
        output={"created_at": datetime(2026, 1, 1, tzinfo=UTC), "pages": 1},
    ).with_image(image)
    consume = AsyncMock(return_value=True)
    monkeypatch.setattr(held.review_store, "consume", consume)

    outcome, output = await held._run("user-1", session, call, tool)

    assert outcome == "approved"
    released = HeldImage.model_validate_json(output)
    assert released.image == image
    assert json.loads(released.text) == {
        "created_at": "2026-01-01 00:00:00+00:00",
        "pages": 1,
    }
    tool.execute.assert_awaited_once_with(
        "user-1", session, call.tool_call_id, file_id="f"
    )
    consume.assert_awaited_once_with(call.review_id, "user-1")


@pytest.mark.parametrize("engine", ["sdk", "baseline"])
@pytest.mark.parametrize("delivery", ["repeat_read", "late_result", "failed_cap"])
async def test_approved_image_is_delivered_once_without_refetch(
    rows, image, engine, delivery
):
    session = _session()
    set_execution_context("user-1", session)
    tool = _ImageRead(image)
    wrapper = _make_truncating_wrapper(create_tool_handler(tool), tool.name)
    judge = AsyncMock(return_value=ContentVerdict(held=True))
    with patch("backend.copilot.gate.reads.judge_content", judge):
        if engine == "sdk":
            result = await wrapper({"file_id": "f"})
            assert result["isError"]
            assert all(block["type"] != "image" for block in result["content"])
        else:
            result = await tool.execute("user-1", session, "call-image", file_id="f")
            assert not result.success
            assert result.image is None
    assert judge.call_args.kwargs["images"][0].data_base64 == image.data_base64
    (review,) = rows.rows.values()
    assert image.data_base64 in review.payload["content"]
    rows.answer(ReviewStatus.APPROVED)
    if delivery != "repeat_read":
        cap = cap_late_tool_result if engine == "sdk" else str
        late = await _late_result(
            session, _failing_cap if delivery == "failed_cap" else cap
        )
        assert late.model_image == image
        assert len(late.content) < 2000
        retried = PendingMessage.model_validate_json(late.model_dump_json())
        assert retried.model_image == image
        model_message = format_pending_as_user_message(retried)
        assert model_message["content"][1] == image.openai()
    elif engine == "sdk":
        released = await wrapper({"file_id": "f"})
        assert released["content"][-1] == image.mcp()
    else:
        released = await tool.execute("user-1", session, "call-next", file_id="f")
        assert released.image == image
    assert tool.runs == 1
    assert rows.rows == {}


@pytest.mark.parametrize("answer", ["rejected", "expired"])
async def test_unapproved_image_never_reaches_late_model_message(rows, image, answer):
    session = _session()
    tool = _ImageRead(image)
    with patch(
        "backend.copilot.gate.reads.judge_content",
        AsyncMock(return_value=ContentVerdict(held=True)),
    ):
        await tool.execute("user-1", session, "call-image", file_id="f")
    rows.answer(
        ReviewStatus.REJECTED if answer == "rejected" else ReviewStatus.APPROVED
    )
    (review,) = rows.rows.values()
    if answer == "expired":
        review.reviewed_at = datetime.now(UTC) - timedelta(hours=2)
    late = await _late_result(session)
    assert late.model_image is None
    assert image.data_base64 not in late.content
    assert tool.runs == 1


async def test_sandbox_image_survives_content_gate_and_late_approval(rows, image):
    session = _session()
    set_execution_context("user-1", session)
    reader = AsyncMock(
        return_value={
            "content": [{"type": "text", "text": "Rendered page"}, image.mcp()],
            "isError": False,
        }
    )
    wrapper = _make_truncating_wrapper(reader, "read_file")
    judge = AsyncMock(return_value=ContentVerdict(held=True))
    with patch("backend.copilot.gate.reads.judge_content", judge):
        result = await wrapper({"file_path": "/home/user/page.png"})
    assert result["isError"]
    assert judge.call_args.kwargs["images"][0].data_base64 == image.data_base64
    rows.answer(ReviewStatus.APPROVED)
    late = await _late_result(session)
    assert late.model_image == image
    assert reader.await_count == 1


async def _late_result(session, cap=cap_late_tool_result):
    delivered: list[PendingMessage] = []
    statuses = [
        status
        async for status in held.resolve_answered(
            "user-1", session, delivered.append, cap=cap
        )
    ]
    assert not statuses
    (result,) = delivered
    return result


def _failing_cap(_text: str) -> str:
    raise ValueError("capping failed")


async def test_expired_image_is_not_released_on_identical_read(rows, image):
    session = _session()
    tool = _ImageRead(image)
    with patch(
        "backend.copilot.gate.reads.judge_content",
        AsyncMock(return_value=ContentVerdict(held=True)),
    ):
        await tool.execute("user-1", session, "call-image", file_id="f")
    rows.answer(ReviewStatus.APPROVED)
    (review,) = rows.rows.values()
    review.reviewed_at = datetime.now(UTC) - timedelta(hours=2)
    result = await tool.execute("user-1", session, "call-retry", file_id="f")
    assert not result.success
    assert result.image is None
    assert tool.runs == 1


@pytest.mark.parametrize("persisted", [True, False])
async def test_pending_image_persistence_keeps_pixels_out_of_chat_rows(
    monkeypatch, image, persisted
):
    session = _session()
    session.messages = []
    transcript = TranscriptBuilder()
    message = PendingMessage(content="Approved image", model_image=image)

    async def persist(session, log_prefix):
        if persisted:
            session.messages[-1].sequence = 0
        return session

    requeue = AsyncMock()
    monkeypatch.setattr(pending_message_helpers, "persist_session_safe", persist)
    monkeypatch.setattr(pending_message_helpers, "push_pending_message", requeue)
    result = await pending_message_helpers.persist_pending_as_user_rows(
        session, transcript, [message], log_prefix="test"
    )
    assert result is persisted
    assert image.data_base64 not in session.model_dump_json()
    if persisted:
        assert session.messages[-1].content == message.content
        assert image.data_base64 in transcript.to_jsonl()
        requeue.assert_not_awaited()
    else:
        assert not session.messages
        assert image.data_base64 not in transcript.to_jsonl()
        requeue.assert_awaited_once_with(session.session_id, message)
