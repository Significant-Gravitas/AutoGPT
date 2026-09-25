"""A chat's checkout record in its private runtime directory.

``checkout.json`` is the checkout in progress, mirrored to the durable
``ledger`` before every change; ``sensitive`` seals the browser while anything
in it may hold a card. Called under ``runtime.browser_operation``'s lock.
"""

import time
from pathlib import Path

from backend.util.link_checkout import ledger
from backend.util.link_checkout.models import CheckoutIntent


def assert_observable(directory: Path) -> None:
    if (directory / "sensitive").exists():
        raise RuntimeError(
            "This payment browser is sealed. Reconcile Link status and reset the "
            "payment browser before browsing again."
        )


def mark_sensitive(directory: Path) -> None:
    with (directory / "sensitive").open("x"):
        pass


def unseal(directory: Path) -> None:
    """Let the chat browse again once nothing that held a card is left."""
    (directory / "sensitive").unlink(missing_ok=True)


def has_checkout(directory: Path) -> bool:
    return (directory / "checkout.json").exists()


def save_intent(directory: Path, intent: CheckoutIntent) -> None:
    if has_checkout(directory):
        raise RuntimeError("This browser already has a checkout; use its checkout ID")
    ledger.save(intent)
    (directory / "checkout.json").write_text(intent.model_dump_json())


def replace_intent(directory: Path, intent: CheckoutIntent) -> None:
    ledger.save(intent)
    temporary = directory / "checkout.pending"
    temporary.write_text(intent.model_dump_json())
    temporary.replace(directory / "checkout.json")


def read_intent(
    directory: Path,
    checkout_id: str,
    user_id: str,
    key: str,
    *,
    for_status: bool = False,
) -> CheckoutIntent:
    intent = CheckoutIntent.model_validate_json(
        (directory / "checkout.json").read_bytes()
    )
    if (intent.id, intent.user_id, intent.session_id) != (checkout_id, user_id, key):
        raise RuntimeError("Checkout not found for this user and browser session")
    if not for_status and (intent.expires_at <= time.time() or intent.attempted):
        raise RuntimeError(
            "Checkout expired or already attempted; do not retry payment"
        )
    return intent


def current_intent(directory: Path) -> CheckoutIntent:
    return CheckoutIntent.model_validate_json(
        (directory / "checkout.json").read_bytes()
    )


def consume_intent(directory: Path, intent: CheckoutIntent) -> None:
    """Record the single payment attempt before any card is retrieved."""
    current = read_intent(directory, intent.id, intent.user_id, intent.session_id)
    current.attempted = True
    replace_intent(directory, current)
    mark_sensitive(directory)


def archive_intent(directory: Path, intent: CheckoutIntent) -> None:
    """Close a checkout for good, so the chat can start another one."""
    ledger.retire(intent.session_id, intent.id)
    (directory / "checkout.json").replace(directory / f"completed-{intent.id}.json")
    for name in ("sensitive", "receipt.json", "status.json"):
        (directory / name).unlink(missing_ok=True)
