import ast
from collections.abc import Iterator

import pytest

from backend.util.architecture_test import BACKEND_ROOT, _iter_backend_py_files

# ---------------------------------------------------------------------------
# Rule: a webhook's ID reaches a log line or an error message only masked
# ---------------------------------------------------------------------------
#
# Motivation: a webhook's ID is the credential in its ingress URL; whoever holds
# it can post deliveries to it. A logging call's or a raised exception's
# arguments show it through `mask_credential`, whose prefix still finds the
# record. Elsewhere, such as in the ingress URL or an event bus channel, the
# ID itself is the point, so no file or function is exempt. The rule knows the
# ID as `webhook_id`, `x.webhook_id` and `webhook.id`, however an argument
# builds its text from it; another name for it is not recognised.

_LOG_METHODS = frozenset(
    {"debug", "info", "warning", "error", "exception", "critical", "log"}
)


def test_webhook_ids_are_logged_and_raised_only_masked():
    offenders = [
        f"  backend/{py.relative_to(BACKEND_ROOT).as_posix()}:{lineno}"
        for py in _iter_backend_py_files()
        if not py.name.endswith("_test.py")
        for lineno in _unmasked_webhook_ids(ast.parse(py.read_text(encoding="utf-8")))
    ]

    assert not offenders, (
        "A webhook's ID is the credential in its ingress URL. Pass it to a log "
        "line or an error message as `mask_credential(webhook_id)` "
        "(backend.util.url_redaction):\n" + "\n".join(sorted(offenders))
    )


def _unmasked_webhook_ids(tree: ast.Module) -> list[int]:
    """Lines where a logging call's or a raised exception's arguments carry a
    webhook's ID unmasked, as an argument or inside an f-string. A raised
    exception's cause is checked too, since a traceback prints it."""
    found: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _LOG_METHODS
        ):
            calls = [node]
        elif isinstance(node, ast.Raise):
            calls = [
                part for part in (node.exc, node.cause) if isinstance(part, ast.Call)
            ]
        else:
            continue
        found += [
            hit.lineno
            for call in calls
            for argument in [*call.args, *(keyword.value for keyword in call.keywords)]
            for hit in _raw_ids(argument)
        ]
    return found


def _raw_ids(node: ast.AST) -> Iterator[ast.expr]:
    """The webhook-ID expressions in `node`, outside any `mask_credential` call
    and the ID `loggable_event_type` masks in an event type."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "mask_credential":
            return
        if node.func.id == "loggable_event_type":
            shown = [
                *node.args[:1],
                *(k.value for k in node.keywords if k.arg != "webhook_id"),
            ]
            for argument in shown:
                yield from _raw_ids(argument)
            return
    if isinstance(node, ast.expr) and _is_webhook_id(node):
        yield node
        return
    for child in ast.iter_child_nodes(node):
        yield from _raw_ids(child)


def _is_webhook_id(node: ast.expr) -> bool:
    """`webhook_id`, `x.webhook_id` or `webhook.id`, the names the backend uses."""
    if isinstance(node, ast.Name):
        return node.id == "webhook_id"
    if isinstance(node, ast.Attribute):
        return node.attr == "webhook_id" or (
            node.attr == "id"
            and isinstance(node.value, ast.Name)
            and node.value.id == "webhook"
        )
    return False


@pytest.mark.parametrize(
    "source, flagged",
    [
        ('logger.info(f"Webhook #{webhook_id} not found")', True),
        ('raise NotFoundError(f"Webhook #{webhook.id} not found")', True),
        ("raise ValueError(webhook_id)", True),
        ('raise HTTPException(404, detail=f"#{webhook_id}")', True),
        ('logger.info("Pruned %s", event.webhook_id)', True),
        ('logger.info("Webhook #%s" % webhook_id)', True),
        ('logger.info("Webhook #" + webhook.id)', True),
        ('logger.info("Webhook #{}".format(webhook_id))', True),
        ('logger.info(f"Webhook #{mask_credential(webhook_id)} not found")', False),
        ('logger.info("Pruned %s", mask_credential(webhook.id))', False),
        ("raise ValueError(mask_credential(webhook_id))", False),
        ('logger.debug("%s", loggable_event_type(event_type, webhook_id))', False),
        ('logger.debug("%s", loggable_event_type(webhook_id, webhook_id))', True),
        ('raise RuntimeError("Setup failed") from ValueError(webhook_id)', True),
        (
            'raise RuntimeError("Setup failed") from '
            "ValueError(mask_credential(webhook_id))",
            False,
        ),
        ('logger.debug(f"Provider webhook #{webhook.provider_webhook_id}")', False),
        # Where the ID itself is the point, such as the ingress URL.
        ('return f"{base}/webhooks/{webhook_id}/ingress"', False),
        ('channel = f"{webhook_id}/{event_type}"', False),
    ],
)
def test_the_webhook_id_scan_reads_each_form(source: str, flagged: bool):
    assert bool(_unmasked_webhook_ids(ast.parse(source))) is flagged


@pytest.mark.parametrize(
    "module",
    [
        "integrations/webhooks/utils.py",
        "data/integrations.py",
        "blocks/airtable/_api.py",
    ],
)
def test_a_module_that_builds_with_a_webhook_id_is_still_scanned(module: str):
    """The ingress URL, the event bus's channels and Airtable's own URLs are
    built from a webhook's ID and pass; a log line beside them does not."""
    source = (BACKEND_ROOT / module).read_text(encoding="utf-8")
    assert not _unmasked_webhook_ids(ast.parse(source))
    added = source + '\n\ndef added(webhook_id):\n    logger.info(f"#{webhook_id}")\n'
    assert _unmasked_webhook_ids(ast.parse(added)) == [len(added.splitlines())]
