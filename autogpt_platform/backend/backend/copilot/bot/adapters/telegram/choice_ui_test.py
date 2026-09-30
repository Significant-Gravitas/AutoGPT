"""Tests for the Telegram choice-button keyboard builder and callback codec."""

from .choice_ui import choice_keyboard, parse_callback_data


class TestChoiceKeyboard:
    def test_builds_one_button_per_row_with_encoded_callback_data(self):
        keyboard = choice_keyboard("abcdef012345", ["US", "EU"])
        rows = keyboard["inline_keyboard"]
        assert [row[0]["text"] for row in rows] == ["US", "EU"]
        assert rows[0][0]["callback_data"] == "qans:abcdef012345:0"
        assert rows[1][0]["callback_data"] == "qans:abcdef012345:1"

    def test_button_text_truncates_and_callback_data_stays_under_64_bytes(self):
        keyboard = choice_keyboard("abcdef012345", ["x" * 200])
        button = keyboard["inline_keyboard"][0][0]
        assert len(button["text"]) == 64
        assert len(button["callback_data"].encode()) <= 64


class TestParseCallbackData:
    def test_round_trips_token_and_index(self):
        keyboard = choice_keyboard("abcdef012345", ["US", "EU"])
        data = keyboard["inline_keyboard"][1][0]["callback_data"]
        assert parse_callback_data(data) == ("abcdef012345", 1)

    def test_rejects_malformed_callback_data(self):
        assert parse_callback_data("not-a-choice-callback") is None
        assert parse_callback_data("qans:short:0") is None
        assert parse_callback_data("qans:abcdef012345:notanumber") is None
