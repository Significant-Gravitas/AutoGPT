import pytest

from backend.api.features.partner_embed.models import ProvisionPartnerIdentityRequest
from backend.api.features.partner_embed.provisioning import derive_shadow_identity_ids


def identity(
    *,
    subject: str = "user-123",
    account_id: str = "forwarder-42",
) -> ProvisionPartnerIdentityRequest:
    return ProvisionPartnerIdentityRequest(
        partner_id="logistics-partner",
        external_subject=subject,
        external_account_id=account_id,
        display_name="Jordan Avery",
        account_name="Acme Forwarding",
        is_admin=True,
    )


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    """This pure identity suite does not need the integration server."""
    yield


def test_shadow_ids_are_stable_for_the_same_partner_identity():
    assert derive_shadow_identity_ids(identity()) == derive_shadow_identity_ids(
        identity()
    )


def test_user_identity_is_isolated_between_customer_accounts():
    first = derive_shadow_identity_ids(identity(account_id="forwarder-1"))
    second = derive_shadow_identity_ids(identity(account_id="forwarder-2"))

    assert first.user_id != second.user_id
    assert first.organization_id != second.organization_id
    assert first.team_id != second.team_id


def test_different_external_subjects_cannot_collide():
    first = derive_shadow_identity_ids(identity(subject="user-1"))
    second = derive_shadow_identity_ids(identity(subject="user-2"))

    assert first.user_id != second.user_id
    assert first.organization_id == second.organization_id
