"""Every path that starts a copilot turn must produce an enforced envelope.

The whole per-tree design rests on one property: a turn that reaches the
executor carries a :class:`TurnEnvelope`. A turn without one is unenforced end
to end — ``BaseTool.execute`` no-ops and its first spawn is minted as a fresh
root — and that is invisible to every other test, because such a turn behaves
exactly like a pre-feature turn.

These tests guard the property structurally rather than by listing known
callers, so a new entry point added later cannot quietly opt out.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

from backend.copilot.executor.utils import enqueue_copilot_turn

# .../backend/backend — the package root the sweep walks.
_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_enqueue_requires_an_envelope_argument() -> None:
    """Required and keyword-only, so omitting it is a TypeError at the call
    site rather than a silently unenforced turn at runtime."""
    parameter = inspect.signature(enqueue_copilot_turn).parameters["envelope"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_no_call_site_enqueues_a_turn_without_an_envelope() -> None:
    """AST sweep of the whole backend: every call to ``enqueue_copilot_turn``
    passes ``envelope=`` explicitly (or forwards ``**kwargs``)."""
    offenders: list[str] = []
    for path in _PACKAGE_ROOT.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:  # not ours to police
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name) else None
            )
            if name != "enqueue_copilot_turn":
                continue
            passes_envelope = any(
                kw.arg == "envelope" or kw.arg is None for kw in node.keywords
            )
            if not passes_envelope:
                offenders.append(f"{path.relative_to(_PACKAGE_ROOT)}:{node.lineno}")
    assert not offenders, (
        "these call sites start a turn with no envelope, which leaves it "
        f"unenforced and mints its first spawn as a root: {offenders}"
    )


def test_no_engine_re_entry_drops_the_envelope() -> None:
    """The engines re-enter themselves (auto-continue), and those calls are a
    second way to produce an unenforced turn: the parameter defaults to None,
    which clears the contextvar for the rest of the turn even though the turn
    genuinely has an envelope. Swept for the same reason as the enqueue sites.
    """
    offenders: list[str] = []
    targets = {"stream_chat_completion_sdk", "stream_chat_completion_baseline"}
    for path in _PACKAGE_ROOT.rglob("*.py"):
        if path.name.endswith("_test.py"):
            continue
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name) else None
            )
            if name not in targets:
                continue
            if not any(kw.arg == "envelope" or kw.arg is None for kw in node.keywords):
                offenders.append(f"{path.relative_to(_PACKAGE_ROOT)}:{node.lineno}")
    assert not offenders, (
        "these engine calls omit envelope=, which defaults to None and leaves "
        f"the rest of the turn unenforced: {offenders}"
    )


def test_chat_platform_turns_are_rooted_and_born_tainted() -> None:
    """A chat-platform message is authored off-platform by someone who need
    not be the account owner, so its turn roots a tree and carries taint."""
    source = (_PACKAGE_ROOT / "platform_linking" / "chat.py").read_text()
    assert "envelope=root_envelope(turn_id, tainted=True)" in source
