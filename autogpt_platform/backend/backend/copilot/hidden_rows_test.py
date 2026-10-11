from backend.copilot.hidden_rows import is_hidden_from_model, visible_to_model
from backend.copilot.model import ChatMessage


def test_only_an_explicit_flag_hides_a_row():
    assert is_hidden_from_model({"hidden_from_model": True})
    assert not is_hidden_from_model({"hidden_from_model": "true"})
    assert not is_hidden_from_model({"kind": "expert_run"})
    assert not is_hidden_from_model(None)


def test_visible_to_model_keeps_order_and_drops_only_flagged_rows():
    rows = [
        ChatMessage(role="user", content="a"),
        ChatMessage(
            role="assistant", content="b", metadata={"hidden_from_model": True}
        ),
        ChatMessage(role="assistant", content="c", metadata={"kind": "expert_run"}),
    ]

    assert [m.content for m in visible_to_model(rows)] == ["a", "c"]
