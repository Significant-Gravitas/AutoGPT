"""A durable record of each chat's checkout, outside the memory-backed runtime.

Holds the checkout (never a token or card field) so a replacement broker can
restore it after a restart. The attempt is recorded here before any card is
retrieved, so a checkout recorded as attempted comes back attempted and
sealed: whether the lost process got as far as paying is unknown, and the only
safe continuation is reconciling with Link, never paying again. One recorded
as not attempted never reached a card, and the fields it pinned died with the
old browser, so it is closed instead of restored.
"""

import hashlib
import os
from pathlib import Path

from backend.util.link_checkout.config import ledger_dir
from backend.util.link_checkout.models import CheckoutIntent


def directory(session_id: str) -> Path | None:
    configured = ledger_dir()
    if not configured:
        return None
    root = Path(configured)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or root.stat().st_mode & 0o077:
        raise RuntimeError("Checkout ledger must be private")
    child = root / hashlib.sha256(session_id.encode()).hexdigest()
    child.mkdir(mode=0o700, exist_ok=True)
    if child.is_symlink() or child.stat().st_mode & 0o077:
        raise RuntimeError("Checkout ledger must be private")
    return child


def save(intent: CheckoutIntent) -> None:
    root = directory(intent.session_id)
    if root is None:
        return
    target = root / "active.json"
    temporary = root / "active.pending"
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w") as output:
        output.write(intent.model_dump_json())
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(target)
    sync(root)


def restore(session_id: str, runtime_directory: Path) -> None:
    root = directory(session_id)
    if (
        root is None
        or not (root / "active.json").exists()
        or (runtime_directory / "checkout.json").exists()
    ):
        return
    intent = CheckoutIntent.model_validate_json((root / "active.json").read_bytes())
    if intent.session_id != session_id:
        raise RuntimeError("Checkout ledger mismatch")
    if not intent.attempted:
        retire(session_id, intent.id)
        return
    (runtime_directory / "checkout.json").write_text(intent.model_dump_json())
    (runtime_directory / "sensitive").touch(exist_ok=True)


def retire(session_id: str, checkout_id: str) -> None:
    root = directory(session_id)
    if root is None:
        return
    active = root / "active.json"
    if active.exists():
        intent = CheckoutIntent.model_validate_json(active.read_bytes())
        if intent.id != checkout_id:
            raise RuntimeError("Checkout ledger mismatch")
        active.replace(root / f"completed-{checkout_id}.json")
        sync(root)


def sync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
