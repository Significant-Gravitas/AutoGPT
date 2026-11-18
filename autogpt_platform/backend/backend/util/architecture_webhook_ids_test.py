import ast

import pytest

from backend.util.architecture_test import BACKEND_ROOT, _iter_backend_py_files

# ---------------------------------------------------------------------------
# Rule: a webhook's ID reaches a log line or an error message only masked
# ---------------------------------------------------------------------------
#
# Motivation: a webhook's ID is the credential in its ingress URL; whoever holds
# it can post deliveries to it. Logs and error messages show it through
# `mask_credential`, whose prefix still finds the record.

# Where the ID itself is the point: the ingress URL and the event bus channels
# hold it. Airtable's API names Airtable's own webhook IDs.
_ALLOWED = frozenset(
    {
        "integrations/webhooks/utils.py",
        "data/integrations.py",
        "blocks/airtable/_api.py",
    }
)
_LOG_METHODS = frozenset(
    {"debug", "info", "warning", "error", "exception", "critical", "log"}
)


def test_webhook_ids_are_logged_and_raised_only_masked():
    offenders = [
        f"  backend/{rel}:{lineno}"
        for py in _iter_backend_py_files()
        if not py.name.endswith("_test.py")
        and (rel := py.relative_to(BACKEND_ROOT).as_posix()) not in _ALLOWED
        for lineno in _unmasked_webhook_ids(ast.parse(py.read_text(encoding="utf-8")))
    ]

    assert not offenders, (
        "A webhook's ID is the credential in its ingress URL. Show it in a log "
        "line or an error message as `mask_credential(webhook_id)` "
        "(backend.util.url_redaction):\n" + "\n".join(sorted(offenders))
    )


def _unmasked_webhook_ids(tree: ast.Module) -> list[int]:
    """Lines that put a webhook's ID, unmasked, into an f-string or a logging
    call's arguments."""
    found = [
        part.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr)
        for part in node.values
        if isinstance(part, ast.FormattedValue) and _is_webhook_id(part.value)
    ]
    found += [
        argument.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _LOG_METHODS
        for argument in node.args
        if _is_webhook_id(argument)
    ]
    return found


def _is_webhook_id(node: ast.expr) -> bool:
    """`webhook_id`, `x.webhook_id` or `webhook.id`."""
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
        ('log(f"Webhook #{webhook_id} not found")', True),
        ('raise NotFoundError(f"Webhook #{webhook.id} not found")', True),
        ('logger.info("Pruned %s", event.webhook_id)', True),
        ('log(f"Webhook #{mask_credential(webhook_id)} not found")', False),
        ('logger.info("Pruned %s", mask_credential(webhook.id))', False),
        ('log(f"Provider webhook #{webhook.provider_webhook_id}")', False),
    ],
)
def test_the_webhook_id_scan_reads_each_form(source: str, flagged: bool):
    assert bool(_unmasked_webhook_ids(ast.parse(source))) is flagged
