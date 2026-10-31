"""Auth-discriminator guard tests for the AnySearch blocks.

Every block honours the same guard pair: ``auth=anonymous`` drops any
still-selected credential and ``auth=api_key`` without a credential errors
before the request method is ever invoked. ``_AUTH_CASES`` is the matrix.
"""

import pytest

from backend.blocks.anysearch._api import AnySearchAuth
from backend.blocks.anysearch.test_helpers import (
    _AUTH_CASES,
    TEST_CREDENTIALS,
    _collect,
    _mock_block,
    _spy,
)


def test_anonymous_auth_makes_credentials_optional():
    """auth=anonymous marks the credentials field credential-free - the
    same discriminator mechanism AutoPilot uses for its platform transport."""
    for block_cls, minimal_input, _method in _AUTH_CASES:
        input_data = block_cls.Input.model_validate(
            {**minimal_input, "auth": "anonymous"}
        )
        assert input_data.auth == AnySearchAuth.ANONYMOUS
        assert input_data.credentials is None
        info = block_cls.Input.get_credentials_fields_info()["credentials"]
        assert "anonymous" in info.credential_free_discriminator_values


@pytest.mark.parametrize("block_cls,minimal_input,method", _AUTH_CASES)
def test_api_key_auth_still_requires_credentials(block_cls, minimal_input, method):
    """auth=api_key keeps the credentials field required."""
    info = block_cls.Input.get_credentials_fields_info()["credentials"]
    mapping = info.discriminator_mapping or {}
    assert "api_key" in mapping
    assert "anonymous" not in mapping
    assert info.requires_credentials("api_key")
    assert not info.requires_credentials("anonymous")


def test_auth_defaults_to_anonymous():
    """anonymous is the default auth - the zero-setup tier; selecting
    api_key is an explicit opt-in."""
    for block_cls, minimal_input, _method in _AUTH_CASES:
        input_data = block_cls.Input.model_validate(minimal_input)
        assert input_data.auth == AnySearchAuth.ANONYMOUS


@pytest.mark.asyncio
@pytest.mark.parametrize("block_cls,minimal_input,method", _AUTH_CASES)
async def test_api_key_auth_without_credentials_errors(
    block_cls, minimal_input, method
):
    captured = []
    block = block_cls()
    _mock_block(block, {method: _spy(captured)})
    with pytest.raises(ValueError, match="credentials are required"):
        await _collect(block, {**minimal_input, "auth": "api_key"}, credentials=None)
    assert captured == []


@pytest.mark.asyncio
@pytest.mark.parametrize("block_cls,minimal_input,method", _AUTH_CASES)
async def test_anonymous_auth_ignores_selected_credential(
    block_cls, minimal_input, method
):
    captured = []
    block = block_cls()
    _mock_block(block, {method: _spy(captured)})
    await _collect(
        block, {**minimal_input, "auth": "anonymous"}, credentials=TEST_CREDENTIALS
    )
    assert captured and all(c is None for c in captured)


@pytest.mark.asyncio
@pytest.mark.parametrize("block_cls,minimal_input,method", _AUTH_CASES)
async def test_api_key_auth_forwards_credentials(block_cls, minimal_input, method):
    captured = []
    block = block_cls()
    _mock_block(block, {method: _spy(captured)})
    await _collect(
        block, {**minimal_input, "auth": "api_key"}, credentials=TEST_CREDENTIALS
    )
    assert captured and all(c is TEST_CREDENTIALS for c in captured)
