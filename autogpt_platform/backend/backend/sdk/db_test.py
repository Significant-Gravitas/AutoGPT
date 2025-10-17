from datetime import datetime

import pytest
from prisma.models import ProviderRegistry as PrismaProviderRegistry

from backend.sdk.db import find_delta_providers, is_providers_different
from backend.sdk.provider import ProviderRegister


@pytest.mark.asyncio
def test_is_providers_different_same():
    current_provider = PrismaProviderRegistry(
        name="test_provider",
        with_oauth=True,
        client_id_env_var="TEST_CLIENT_ID",
        client_secret_env_var="TEST_CLIENT_SECRET",
        with_api_key=True,
        api_key_env_var="TEST_API_KEY",
        with_user_password=True,
        username_env_var="TEST_USERNAME",
        password_env_var="TEST_PASSWORD",
        updatedAt=datetime.now(),
    )
    new_provider = ProviderRegister(
        name="test_provider",
        with_oauth=True,
        client_id_env_var="TEST_CLIENT_ID",
        client_secret_env_var="TEST_CLIENT_SECRET",
        with_api_key=True,
        api_key_env_var="TEST_API_KEY",
        with_user_password=True,
        username_env_var="TEST_USERNAME",
        password_env_var="TEST_PASSWORD",
    )
    assert not is_providers_different(current_provider, new_provider)


@pytest.mark.asyncio
def test_is_providers_different_different():
    current_provider = PrismaProviderRegistry(
        name="test_provider",
        with_oauth=True,
        client_id_env_var="TEST_CLIENT_ID",
        client_secret_env_var="TEST_CLIENT_SECRET",
        with_api_key=True,
        api_key_env_var="TEST_API_KEY",
        with_user_password=True,
        username_env_var="TEST_USERNAME",
        password_env_var="TEST_PASSWORD",
        updatedAt=datetime.now(),
    )
    new_provider = ProviderRegister(
        name="test_provider",
        with_oauth=False,
        with_api_key=True,
        api_key_env_var="TEST_API_KEY",
        with_user_password=True,
        username_env_var="TEST_USERNAME",
        password_env_var="TEST_PASSWORD",
    )
    assert is_providers_different(current_provider, new_provider)


@pytest.mark.asyncio
def test_find_delta_providers():
    current_providers = {
        "test_provider": PrismaProviderRegistry(
            name="test_provider",
            with_oauth=True,
            client_id_env_var="TEST_CLIENT_ID",
            client_secret_env_var="TEST_CLIENT_SECRET",
            with_api_key=True,
            api_key_env_var="TEST_API_KEY",
            with_user_password=True,
            username_env_var="TEST_USERNAME",
            password_env_var="TEST_PASSWORD",
            updatedAt=datetime.now(),
        ),
        "test_provider_2": PrismaProviderRegistry(
            name="test_provider_2",
            with_oauth=True,
            client_id_env_var="TEST_CLIENT_ID_2",
            client_secret_env_var="TEST_CLIENT_SECRET_2",
            with_api_key=True,
            api_key_env_var="TEST_API_KEY_2",
            with_user_password=True,
            username_env_var="TEST_USERNAME_2",
            password_env_var="TEST_PASSWORD_2",
            updatedAt=datetime.now(),
        ),
    }
    new_providers = {
        "test_provider": ProviderRegister(
            name="test_provider",
            with_oauth=True,
            client_id_env_var="TEST_CLIENT_ID",
            client_secret_env_var="TEST_CLIENT_SECRET",
            with_api_key=True,
            api_key_env_var="TEST_API_KEY",
            with_user_password=True,
            username_env_var="TEST_USERNAME",
            password_env_var="TEST_PASSWORD",
        ),
        "test_provider_2": ProviderRegister(
            name="test_provider_2",
            with_oauth=False,
            with_api_key=True,
            api_key_env_var="TEST_API_KEY_2",
            with_user_password=True,
            username_env_var="TEST_USERNAME_2",
            password_env_var="TEST_PASSWORD_2",
        ),
        "test_provider_3": ProviderRegister(
            name="test_provider_3",
            with_oauth=True,
            client_id_env_var="TEST_CLIENT_ID_3",
            client_secret_env_var="TEST_CLIENT_SECRET_3",
            with_api_key=False,
            with_user_password=True,
            username_env_var="TEST_USERNAME_3",
            password_env_var="TEST_PASSWORD_3",
        ),
    }
    assert find_delta_providers(current_providers, new_providers) == {
        "test_provider_2": new_providers["test_provider_2"],
        "test_provider_3": new_providers["test_provider_3"],
    }
