import json

from . import fixture
from .examples import load_examples


def test_every_turn_opens_with_the_business_and_the_team():
    message = fixture.first_message(1, "Hire an SEO expert")

    assert message.startswith("<user_context>")
    assert "Brightwave" in message
    assert all(expert.name in message for expert in fixture.TEAM)
    assert message.endswith("Hire an SEO expert")


def test_what_the_user_attached_or_pasted_arrives_with_the_message():
    csv = fixture.first_message(24, "Clean this CSV, dedupe it and email it to sales")
    pasted = fixture.first_message(40, "Translate this paragraph to Hindi")

    assert "[Attached files]\n- file_id=file-customers" in csv
    assert "meeting time for your whole team" in pasted


def test_the_fixture_only_names_real_examples():
    ids = {e.id for e in load_examples()}

    assert set(fixture.PASTED) | set(fixture.ATTACHED) | set(fixture.ANSWERS) <= ids


def test_the_board_says_why_a_blocked_task_stopped():
    board = json.loads(fixture.account_result("list_my_tasks", {}) or "{}")

    assert "WordPress returned 401" in board["message"]


def test_lookups_outside_the_account_fall_through_to_the_shared_stub():
    assert fixture.account_result("web_search", {}) is None
