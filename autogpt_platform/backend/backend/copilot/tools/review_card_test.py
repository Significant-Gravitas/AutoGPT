"""Tests for what a review output previews and where it says the card is."""

import json
import re

import pytest

from backend.copilot.model import ChatSessionMetadata
from backend.copilot.tools.base import _DIGEST_THRESHOLD
from backend.copilot.tools.models import ReviewRequiredResponse
from backend.copilot.tools.review_card import review_card_location


def _review(input_data: dict) -> ReviewRequiredResponse:
    return ReviewRequiredResponse(
        message="Needs review.",
        block_id="b-1",
        block_name="Gmail Send",
        review_id="r-1",
        input_data=input_data,
    )


class TestInputPreview:
    """The card reads the review's input from the review itself, so the stream
    carries only enough of it for the model to recognise its own call."""

    def test_a_short_input_is_carried_as_is(self):
        input_data = {"to": "ada@example.com", "options": {"cc": ["b@x.com"]}}
        dumped = json.loads(_review(input_data).model_dump_json(exclude_none=True))

        assert dumped["input_data"] == input_data
        assert "input_preview" not in dumped

    def test_a_long_string_is_replaced_by_its_size_at_any_depth(self):
        long = "x" * 600
        review = _review(
            {"body": long, "attachments": [{"name": "a.txt", "text": long}], "n": 3}
        )

        assert review.input_data == {
            "body": "<600 chars>",
            "attachments": [{"name": "a.txt", "text": "<600 chars>"}],
            "n": 3,
        }
        assert review.input_preview is True

    def test_a_field_too_wide_to_preview_is_shown_as_its_size(self):
        review = _review({f"field_{i}": ["v" * 400] * 20 for i in range(5)})

        assert all(
            re.fullmatch(r"<[\d,]+ chars>", value)
            for value in review.input_data.values()
        )
        assert review.input_preview is True

    def test_many_short_fields_stay_short_together(self):
        """No one field is long, but together they would carry a digest-sized
        output: the fields that fit are kept, in order, the rest are sized."""
        input_data = {f"field_{i:02}": "v" * 450 for i in range(40)}
        review = _review(input_data)

        kept = [key for key, value in review.input_data.items() if value == "v" * 450]
        assert kept == [f"field_{i:02}" for i in range(len(kept))]
        assert 0 < len(kept) < 40
        assert all(
            review.input_data[key] == "<450 chars>"
            for key in input_data
            if key not in kept
        )
        assert len(review.model_dump_json()) < _DIGEST_THRESHOLD


class TestCardLocation:
    @pytest.mark.parametrize(
        "metadata",
        [
            None,
            ChatSessionMetadata(origin="interactive"),
            ChatSessionMetadata(origin="automation"),
            # A handoff makes the thread the user's own chat.
            ChatSessionMetadata(
                delegated_by_session_id="parent", handed_off_from_expert_id="maria"
            ),
        ],
        ids=["unknown", "web chat", "scheduled run", "handed off"],
    )
    def test_a_chat_the_user_opens_has_the_card_at_its_bottom(self, metadata):
        location = review_card_location(metadata)

        assert "bottom of this chat" in location
        assert "Needs you on Home" in location
        assert "Never say it is in another chat" in location

    @pytest.mark.parametrize(
        "metadata, says",
        [
            (
                ChatSessionMetadata(source_platform="discord"),
                "No approval card is posted on this platform",
            ),
            (
                ChatSessionMetadata(delegated_by_session_id="parent"),
                "this thread, which the user is not watching",
            ),
        ],
        ids=["chat platform", "delegated thread"],
    )
    def test_elsewhere_it_points_to_home_and_never_to_this_chat(self, metadata, says):
        location = review_card_location(metadata)

        assert says in location
        assert "Needs you on Home" in location
        assert "bottom of this chat" not in location
        assert "don't repeat this call to raise it again" in location.lower()
