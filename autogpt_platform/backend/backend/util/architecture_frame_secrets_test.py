import ast
import json
from collections.abc import Iterator

import pytest

from backend.util import metrics
from backend.util.architecture_test import BACKEND_ROOT, _iter_backend_py_files

# ---------------------------------------------------------------------------
# Rule: a secret's plain text sits in a frame only where Sentry masks it
# ---------------------------------------------------------------------------
#
# Motivation: Sentry error events carry every frame's variables, and the event
# scrubber (backend.util.metrics) masks a secret there by its variable's name or
# mapping key, or by a value the request carried or whose shape gives it away.
# A saved credential's secret is a SecretStr until `.get_secret_value()` unwraps
# it; wherever its text is then held, as a local, a keyword argument (which is
# the callee's parameter) or a dict's entry, the scrubber must mask it. Each
# holder is checked by running the scrubber on it. The rule follows the text
# through assignments and string building within a function, not into calls,
# so a value a call derives from it, such as its base64, is not followed.

_SECRET = "q7Wm2Lp9Xv4Rt8Kc"

# "path:name" of a holder that may keep a secret unmasked, and why.
_UNMASKED_HOLDERS = {
    "blocks/google_maps.py:key": (
        "googlemaps.Client's parameter; masking every `key` would hide most frames"
    ),
    "integrations/webhooks/telegram.py:url": (
        "masked by its shape, a bot token in a Telegram API path (url_redaction.py)"
    ),
}


def test_a_secrets_text_is_held_only_where_sentry_masks_it():
    offenders = [
        f"  backend/{path}:{line} {name}"
        for path, name, line in _unmasked_holders_in_backend()
        if f"{path}:{name}" not in _UNMASKED_HOLDERS
    ]

    assert not offenders, (
        "These hold a secret's text unmasked, so a Sentry error event would carry "
        "it among the frame's variables. Name them so the scrubber masks them, or "
        "add the name to _SENTRY_SENSITIVE_FIELDS in backend/util/metrics.py:\n"
        + "\n".join(sorted(offenders))
    )


def test_every_unmasked_holder_allowed_is_still_there():
    found = {f"{path}:{name}" for path, name, _ in _unmasked_holders_in_backend()}
    assert set(_UNMASKED_HOLDERS) <= found


def _unmasked_holders_in_backend() -> Iterator[tuple[str, str, int]]:
    for py in _iter_backend_py_files():
        if py.name.endswith("_test.py"):
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for name, line in _unmasked_holders(tree):
            yield py.relative_to(BACKEND_ROOT).as_posix(), name, line


def _unmasked_holders(tree: ast.Module) -> list[tuple[str, int]]:
    return [
        (name, line)
        for name, line, value in _secret_holders(tree)
        if not _scrubbed(name, value)
    ]


def _secret_holders(tree: ast.Module) -> Iterator[tuple[str, int, object]]:
    """Each name holding a secret's text, its line, and the value as rendered
    with ``_SECRET`` in the secret's place."""
    scopes = [tree, *_functions(tree)]
    for scope in scopes:
        bindings = list(_bindings(scope))
        # Each local holding a secret's text, and its value as rendered.
        tainted: dict[str, object] = {}
        while True:
            newly = {
                name: rendered
                for name, _, value, local in bindings
                if local
                and name not in tainted
                and _holds(rendered := _render(value, tainted))
            }
            if not newly:
                break
            tainted |= newly
        for name, line, value, _ in bindings:
            rendered = _render(value, tainted)
            if _holds(rendered):
                yield name, line, rendered


def _scrubbed(name: str, value: object) -> bool:
    """Whether the scrubber masks ``value`` as a frame's variable ``name``."""
    frame = {"function": "f", "vars": {name: _as_sent(value)}}
    event = {"exception": {"values": [{"stacktrace": {"frames": [frame]}}]}}
    return _SECRET not in json.dumps(metrics._before_send(event, {}))


def _functions(tree: ast.Module) -> list[ast.AST]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _bindings(scope: ast.AST) -> Iterator[tuple[str, int, ast.expr, bool]]:
    """Names bound in ``scope`` and their values: assigned locals (``True``),
    and keyword arguments, each the callee's parameter (``False``)."""
    for node in _walk_scope(scope):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                yield from _paired(target, node.value)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value:
            yield from _paired(node.target, node.value)
        elif isinstance(node, ast.NamedExpr):
            yield from _paired(node.target, node.value)
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg:
                    yield keyword.arg, node.lineno, keyword.value, False


def _walk_scope(scope: ast.AST) -> Iterator[ast.AST]:
    """``scope``'s nodes, leaving out the functions and classes defined in it."""
    pending = list(ast.iter_child_nodes(scope))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield node
        pending.extend(ast.iter_child_nodes(node))


def _paired(
    target: ast.expr, value: ast.expr
) -> Iterator[tuple[str, int, ast.expr, bool]]:
    if isinstance(target, ast.Name):
        yield target.id, target.lineno, value, True
    elif isinstance(target, (ast.Tuple, ast.List)):
        if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(
            target.elts
        ):
            for part, part_value in zip(target.elts, value.elts):
                yield from _paired(part, part_value)
        else:
            for part in target.elts:
                yield from _paired(part, value)


def _render(node: ast.AST, tainted: dict[str, object]) -> object:
    """``node``'s value with ``_SECRET`` for a secret's text; ``None`` for what a
    call returns or anything else not followed."""
    if _is_unwrap(node):
        return _SECRET
    if isinstance(node, ast.Name) and node.id in tainted:
        return tainted[node.id]
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else repr(node.value)
    if isinstance(node, ast.JoinedStr):
        return "".join(
            _text(
                _render(
                    part.value if isinstance(part, ast.FormattedValue) else part,
                    tainted,
                )
            )
            for part in node.values
        )
    if isinstance(node, ast.BinOp):
        return _text(_render(node.left, tainted)) + _text(_render(node.right, tainted))
    if isinstance(node, (ast.IfExp, ast.BoolOp)):
        options = (
            [node.body, node.orelse] if isinstance(node, ast.IfExp) else node.values
        )
        rendered = [_render(option, tainted) for option in options]
        return next((value for value in rendered if _holds(value)), rendered[0])
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_render(element, tainted) for element in node.elts]
    if isinstance(node, ast.Dict):
        return {
            key.value if isinstance(key, ast.Constant) else "key": _render(
                value, tainted
            )
            for key, value in zip(node.keys, node.values)
            if key is not None
        }
    return None


def _is_unwrap(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_secret_value"
    )


def _holds(value: object) -> bool:
    return _SECRET in json.dumps(value, default=str)


def _text(value: object) -> str:
    return value if isinstance(value, str) else "x"


def _as_sent(value: object) -> object:
    """``value`` as the SDK serializes a frame's variable: text as its repr."""
    if isinstance(value, list):
        return [_as_sent(element) for element in value]
    if isinstance(value, dict):
        return {str(key): _as_sent(element) for key, element in value.items()}
    return repr(value)


@pytest.mark.parametrize(
    "source, flagged",
    [
        ("smtp_login = creds.username.get_secret_value()", ["smtp_login"]),
        ("password = creds.password.get_secret_value()", []),
        ("connect(smtp_login=creds.username.get_secret_value())", ["smtp_login"]),
        ("connect(api_key=creds.api_key.get_secret_value())", []),
        (
            "def f(creds):\n"
            "    user = creds.username.get_secret_value()\n"
            "    joined = f'{user}:x'\n",
            ["joined", "user"],
        ),
        ("token = a.get_secret_value() if a else None", []),
        ("login = a.get_secret_value() if a else None", ["login"]),
        ("pair = (creds.username.get_secret_value(), 'x')", ["pair"]),
        ("params = {'appkey': creds.api_key.get_secret_value()}", ["params"]),
        ("params = {'api_key': creds.api_key.get_secret_value()}", []),
        (
            "headers = {'Authorization': 'Bearer ' + creds.token.get_secret_value()}",
            [],
        ),
        # Not followed into a call: the client holds the secret, not its text.
        ("client = Client(creds.api_key.get_secret_value())", []),
        ("encoded = b64encode(creds.password.get_secret_value())", []),
    ],
)
def test_the_secret_holder_scan_reads_each_form(source: str, flagged: list[str]):
    assert sorted(name for name, _ in _unmasked_holders(ast.parse(source))) == (flagged)


def test_a_new_holder_in_a_scanned_module_is_flagged():
    """The SMTP login is held under names the scrubber masks; a new one under
    a name it doesn't is caught."""
    source = (BACKEND_ROOT / "blocks/email_block.py").read_text(encoding="utf-8")
    assert not _unmasked_holders(ast.parse(source))
    added = source + "\n\ndef f(c):\n    login = c.username.get_secret_value()\n"
    assert _unmasked_holders(ast.parse(added)) == [("login", len(added.splitlines()))]
