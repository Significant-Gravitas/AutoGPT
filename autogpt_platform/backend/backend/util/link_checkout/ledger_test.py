import pytest

from backend.util.link_checkout import ledger
from backend.util.link_checkout.checkout_record import has_checkout, read_intent


@pytest.fixture
def ledger_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECKOUT_BROKER_LEDGER_DIR", str(tmp_path / "ledger"))
    fresh = tmp_path / "fresh-runtime"
    fresh.mkdir()
    return fresh


def test_container_loss_restores_an_attempt_only_for_reconciliation(ledger_dir, intent):
    intent.attempted = True
    ledger.save(intent)

    ledger.restore(intent.session_id, ledger_dir)

    assert (ledger_dir / "sensitive").exists()
    with pytest.raises(RuntimeError):
        read_intent(ledger_dir, intent.id, intent.user_id, intent.session_id)
    recovered = read_intent(
        ledger_dir, intent.id, intent.user_id, intent.session_id, for_status=True
    )
    assert recovered.spend_request_id == intent.spend_request_id
    assert recovered.attempted
    ledger.retire(intent.session_id, intent.id)
    assert not (ledger.directory(intent.session_id) / "active.json").exists()


def test_a_checkout_that_never_reached_a_card_is_closed_not_restored(
    ledger_dir, intent
):
    """Its pinned fields died with the old browser, and nothing was charged.
    Restoring it as an attempt left a chat that could never be cleared: a
    checkout still waiting for the customer has no Link request to reconcile."""
    intent.spend_request_id = None
    ledger.save(intent)

    ledger.restore(intent.session_id, ledger_dir)

    assert not has_checkout(ledger_dir)
    assert not (ledger_dir / "sensitive").exists()
    root = ledger.directory(intent.session_id)
    assert root is not None and not (root / "active.json").exists()
