import pydantic
import pytest

from backend.api.features.api_keys.model import (
    CreateAPIKeyRequest,
    UpdatePermissionsRequest,
)
from backend.data.auth.api_key import OAUTH_ONLY_PERMISSIONS, APIKeyPermission
from backend.util.text_limits import (
    MAX_DESCRIPTION_LENGTH,
    MAX_ID_LENGTH,
    MAX_NAME_LENGTH,
)


@pytest.mark.parametrize("permission", sorted(OAUTH_ONLY_PERMISSIONS))
def test_an_api_key_cannot_be_given_an_oauth_only_permission(
    permission: APIKeyPermission,
) -> None:
    """CONNECT_LOCAL_EXECUTOR is checked only on the companion's OAuth token;
    on an API key it would do nothing and suggest it could."""
    with pytest.raises(pydantic.ValidationError, match="only be granted through"):
        CreateAPIKeyRequest(
            name="key", permissions=[APIKeyPermission.READ_GRAPH, permission]
        )
    with pytest.raises(pydantic.ValidationError, match="only be granted through"):
        UpdatePermissionsRequest(permissions=[permission])


def test_other_permissions_still_go_on_api_keys() -> None:
    request = CreateAPIKeyRequest(
        name="key",
        permissions=[APIKeyPermission.READ_GRAPH, APIKeyPermission.USE_TOOLS],
    )

    assert request.permissions == [
        APIKeyPermission.READ_GRAPH,
        APIKeyPermission.USE_TOOLS,
    ]


EVERY_PERMISSION = list(APIKeyPermission)
# Repeats of a grantable permission, one more than there are permissions.
ONE_TOO_MANY = [APIKeyPermission.IDENTITY] * (len(EVERY_PERMISSION) + 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("permissions", ONE_TOO_MANY),
        ("name", "n" * (MAX_NAME_LENGTH + 1)),
        ("description", "d" * (MAX_DESCRIPTION_LENGTH + 1)),
        ("team_id", "t" * (MAX_ID_LENGTH + 1)),
    ],
    ids=["permissions", "name", "description", "team_id"],
)
def test_creating_a_key_refuses_an_oversized_field(field: str, value) -> None:
    body = {"name": "key", "permissions": [APIKeyPermission.IDENTITY], field: value}

    with pytest.raises(pydantic.ValidationError) as error:
        CreateAPIKeyRequest(**body)

    assert error.value.errors()[0]["loc"] == (field,)


def test_updating_a_key_refuses_more_permissions_than_exist() -> None:
    with pytest.raises(pydantic.ValidationError):
        UpdatePermissionsRequest(permissions=ONE_TOO_MANY)


def test_a_key_may_carry_every_api_key_permission_and_fields_at_their_bound() -> None:
    allowed = [p for p in EVERY_PERMISSION if p not in OAUTH_ONLY_PERMISSIONS]

    request = CreateAPIKeyRequest(
        name="n" * MAX_NAME_LENGTH,
        permissions=allowed,
        description="d" * MAX_DESCRIPTION_LENGTH,
        team_id="t" * MAX_ID_LENGTH,
    )

    assert request.permissions == allowed
    assert UpdatePermissionsRequest(permissions=allowed).permissions == allowed
