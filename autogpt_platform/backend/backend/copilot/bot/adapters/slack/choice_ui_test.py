"""Tests for the Slack choice-button block builder and action-id codec."""

from .choice_ui import choice_blocks, parse_action_id


class TestChoiceBlocks:
    def test_builds_a_button_per_option_with_encoded_action_ids(self):
        blocks = choice_blocks("Which region?", "abcdef012345", ["US", "EU"])
        assert blocks[0] == {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "Which region?"},
        }
        elements = blocks[1]["elements"]
        assert [e["text"]["text"] for e in elements] == ["US", "EU"]
        assert elements[0]["action_id"] == "qans:abcdef012345:0"
        assert elements[1]["action_id"] == "qans:abcdef012345:1"

    def test_button_text_truncates_to_slack_cap(self):
        blocks = choice_blocks("Q?", "abcdef012345", ["x" * 200])
        assert len(blocks[1]["elements"][0]["text"]["text"]) == 75


class TestParseActionId:
    def test_round_trips_token_and_index(self):
        blocks = choice_blocks("Q?", "abcdef012345", ["US", "EU"])
        action_id = blocks[1]["elements"][1]["action_id"]
        assert parse_action_id(action_id) == ("abcdef012345", 1)

    def test_rejects_malformed_action_id(self):
        assert parse_action_id("not-a-choice-action") is None
        assert parse_action_id("qans:short:0") is None
        assert parse_action_id("qans:abcdef012345:notanumber") is None
