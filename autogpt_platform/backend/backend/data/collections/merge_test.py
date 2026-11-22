"""The rule for a person's edit against the next automated write, cell by cell."""

from .merge import RowState, accept_pending, automated_write, new_row, person_edit


def _priya_corrected() -> RowState:
    """An invoice the workflow wrote at 3400, which Priya corrected to 340."""
    written = new_row({"invoice_no": "INV-7", "amount": 3400}, by_person=False)
    return person_edit(written, {"amount": 340}).state


def test_an_unchanged_source_value_leaves_the_persons_correction() -> None:
    merged = automated_write(
        _priya_corrected(), {"invoice_no": "INV-7", "amount": 3400}
    )

    assert merged.state.data["amount"] == 340
    assert merged.state.pending == {}
    assert merged.changed is False


def test_a_new_source_value_lands_where_no_person_edited() -> None:
    row = new_row({"sku": "TT-1", "on_hand": 12}, by_person=False)

    merged = automated_write(row, {"sku": "TT-1", "on_hand": 9})

    assert merged.state.data["on_hand"] == 9
    assert merged.state.pending == {}
    assert merged.held == []


def test_a_new_source_value_over_a_persons_edit_is_held_as_pending() -> None:
    merged = automated_write(_priya_corrected(), {"amount": 3500})

    assert merged.state.data["amount"] == 340
    assert merged.state.pending == {"amount": 3500}
    assert merged.held == ["amount"]
    # The held value is now the source's, so the next identical write is quiet.
    again = automated_write(merged.state, {"amount": 3500})
    assert again.changed is False
    assert again.state.pending == {"amount": 3500}


def test_accepting_a_pending_value_applies_it_and_ends_the_override() -> None:
    held = automated_write(_priya_corrected(), {"amount": 3500}).state

    accepted = accept_pending(held, ["amount"]).state

    assert accepted.data["amount"] == 3500
    assert accepted.pending == {}
    assert "amount" not in accepted.manual
    assert automated_write(accepted, {"amount": 3600}).state.data["amount"] == 3600


def test_a_source_that_catches_up_with_the_person_ends_the_override() -> None:
    merged = automated_write(_priya_corrected(), {"amount": 340})

    assert merged.state.data["amount"] == 340
    assert merged.state.manual == []
    assert automated_write(merged.state, {"amount": 360}).state.data["amount"] == 360


def test_a_persons_edit_drops_a_value_held_against_their_earlier_one() -> None:
    held = automated_write(_priya_corrected(), {"amount": 3500}).state

    edited = person_edit(held, {"amount": 345}).state

    assert edited.data["amount"] == 345
    assert edited.pending == {}


def test_a_write_touches_only_the_fields_it_names() -> None:
    row = person_edit(
        new_row({"profile": "p/1", "fit": 70}, by_person=False), {"notes": "Call Fri"}
    ).state

    merged = automated_write(row, {"profile": "p/1", "fit": 82})

    assert merged.state.data == {"profile": "p/1", "fit": 82, "notes": "Call Fri"}


def test_a_row_a_person_adds_is_theirs_until_a_source_writes_it() -> None:
    row = new_row({"what": "Send quote", "done": False}, by_person=True)

    assert row.manual == ["done", "what"]
    assert automated_write(row, {"done": True}).state.pending == {"done": True}
