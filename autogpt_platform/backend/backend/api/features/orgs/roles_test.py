"""Only the owner makes someone an admin or billing manager, or takes it away."""

import pytest
from autogpt_libs.auth.models import RequestContext

from backend.api.features.orgs.roles import require_owner_for_role_change
from backend.util.exceptions import NotAuthorizedError


def _actor(owner: bool) -> RequestContext:
    return RequestContext(
        user_id="actor",
        org_id="org",
        team_id=None,
        is_org_owner=owner,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=False,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )


@pytest.mark.parametrize(
    "wants_admin, wants_billing",
    [(True, None), (None, True), (False, None)],
    ids=["grant-admin", "grant-billing", "revoke-admin"],
)
def test_an_admin_cannot_change_roles(wants_admin, wants_billing):
    with pytest.raises(NotAuthorizedError, match="owner"):
        require_owner_for_role_change(
            _actor(owner=False),
            is_admin=wants_admin is False,
            is_billing_manager=False,
            wants_admin=wants_admin,
            wants_billing_manager=wants_billing,
        )


def test_the_owner_can():
    require_owner_for_role_change(
        _actor(owner=True),
        is_admin=False,
        is_billing_manager=False,
        wants_admin=True,
        wants_billing_manager=True,
    )


@pytest.mark.parametrize("wants_admin, wants_billing", [(None, None), (False, False)])
def test_an_admin_can_invite_or_keep_a_plain_member(wants_admin, wants_billing):
    require_owner_for_role_change(
        _actor(owner=False),
        is_admin=False,
        is_billing_manager=False,
        wants_admin=wants_admin,
        wants_billing_manager=wants_billing,
    )
