import pydantic
import pytest

from backend.api.features.api_keys.model import (
    CreateAPIKeyRequest,
    UpdatePermissionsRequest,
)
from backend.data.auth.api_key import OAUTH_ONLY_PERMISSIONS, APIKeyPermission


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
