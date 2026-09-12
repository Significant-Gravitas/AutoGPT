"""Tests for the Teams choice-card builder and Action.Submit value codec."""

from .choice_ui import choice_card, parse_choice_value


class TestChoiceCard:
    def test_one_action_submit_per_option(self):
        card = choice_card("Which region?", "abcdef012345", ["US", "EU"])
        content = card["content"]
        assert content["body"][0]["text"] == "Which region?"
        actions = content["actions"]
        assert [a["title"] for a in actions] == ["US", "EU"]
        assert actions[0]["data"] == {"qans_token": "abcdef012345", "qans_index": 0}
        assert actions[1]["data"] == {"qans_token": "abcdef012345", "qans_index": 1}

    def test_title_truncates_to_card_button_cap(self):
        card = choice_card("Q?", "abcdef012345", ["x" * 200])
        assert len(card["content"]["actions"][0]["title"]) == 60


class TestParseChoiceValue:
    def test_round_trips_token_and_index(self):
        card = choice_card("Q?", "abcdef012345", ["US", "EU"])
        value = card["content"]["actions"][1]["data"]
        assert parse_choice_value(value) == ("abcdef012345", 1)

    def test_non_dict_value_returns_none(self):
        assert parse_choice_value(None) is None
        assert parse_choice_value("not a dict") is None

    def test_missing_or_wrong_typed_fields_return_none(self):
        assert parse_choice_value({}) is None
        assert parse_choice_value({"qans_token": "", "qans_index": 0}) is None
        assert parse_choice_value({"qans_token": "tok", "qans_index": "0"}) is None
        assert parse_choice_value({"qans_token": "tok", "qans_index": None}) is None

    def test_bool_index_is_rejected(self):
        # bool is an int subclass -- must not silently pass as index 0/1.
        assert parse_choice_value({"qans_token": "tok", "qans_index": True}) is None
