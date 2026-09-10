import pytest

from backend.blocks.stripe_link._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.stripe_link.profile import StripeLinkGetUserInfoBlock


@pytest.mark.asyncio
async def test_user_info_normalizes_null_profile_fields():
    block = StripeLinkGetUserInfoBlock()

    async def null_profile(credentials, method, path, body=None):
        return {
            "name": None,
            "first_name": None,
            "last_name": None,
            "email": None,
            "phone": None,
        }

    object.__setattr__(block, "_link_api_request", null_profile)
    input_data = block.Input.model_validate({"credentials": TEST_CREDENTIALS_INPUT})

    outputs = {
        name: value
        async for name, value in block.run(input_data, credentials=TEST_CREDENTIALS)
    }

    assert outputs == {
        "name": "",
        "first_name": "",
        "last_name": "",
        "email": "",
        "phone": "",
    }
