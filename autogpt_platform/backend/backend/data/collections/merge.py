"""Who wins a cell: a person's edit against the next automated write.

An automated write changes a cell only when its value differs from the one it
brought last time; where a person has edited the cell since, the person's
value stays and the newer one is held as pending, one click from applied.
"""

from typing import Any

from pydantic import BaseModel, Field


class RowState(BaseModel):
    data: dict[str, Any] = Field(default_factory=dict)
    # Per field, the value an automated write last brought, applied or held.
    source: dict[str, Any] = Field(default_factory=dict)
    pending: dict[str, Any] = Field(default_factory=dict)
    manual: list[str] = Field(default_factory=list)


class Merged(BaseModel):
    state: RowState
    changed: bool
    held: list[str] = Field(default_factory=list)


def automated_write(row: RowState, values: dict[str, Any]) -> Merged:
    """*values* written by a workflow, an expert or Otto, field by field."""
    state = row.model_copy(deep=True)
    manual = set(state.manual)
    changed, held = False, []
    for name, value in values.items():
        if name in state.source and state.source[name] == value:
            continue
        state.source[name] = value
        if name not in manual:
            state.data[name] = value
            state.pending.pop(name, None)
        elif state.data.get(name) == value:
            # The source caught up with the person: nothing left to hold.
            manual.discard(name)
            state.pending.pop(name, None)
        else:
            state.pending[name] = value
            held.append(name)
        changed = True
    state.manual = sorted(manual)
    return Merged(state=state, changed=changed, held=held)


def person_edit(row: RowState, values: dict[str, Any]) -> Merged:
    """*values* typed by the owner: they stand until the source brings a new one."""
    state = row.model_copy(deep=True)
    manual = set(state.manual)
    for name, value in values.items():
        state.data[name] = value
        state.pending.pop(name, None)
        manual.add(name)
    state.manual = sorted(manual)
    return Merged(state=state, changed=bool(values))


def accept_pending(row: RowState, names: list[str]) -> Merged:
    """The held automated values for *names* applied, as if no one had edited them."""
    state = row.model_copy(deep=True)
    manual = set(state.manual)
    accepted = [name for name in names if name in state.pending]
    for name in accepted:
        state.data[name] = state.pending.pop(name)
        manual.discard(name)
    state.manual = sorted(manual)
    return Merged(state=state, changed=bool(accepted))


def new_row(values: dict[str, Any], *, by_person: bool) -> RowState:
    if by_person:
        return RowState(data=dict(values), manual=sorted(values))
    return RowState(data=dict(values), source=dict(values))
