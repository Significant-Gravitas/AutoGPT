from typing import Dict

from prisma import Prisma
from prisma.models import ProviderRegistry as PrismaProviderRegistry

from backend.sdk.provider import ProviderRegister


def is_providers_different(
    current_provider: PrismaProviderRegistry, new_provider: ProviderRegister
) -> bool:
    """
    Compare a current provider (as stored in the database) against a new provider registration
    and determine if they are different. This is done by converting the database model to a
    ProviderRegister and checking for equality (all fields compared).

    Args:
        current_provider (PrismaProviderRegistry): The provider as stored in the database.
        new_provider (ProviderRegister): The provider specification to compare.

    Returns:
        bool: True if the providers differ, False if they are effectively the same.
    """
    current_provider_register = ProviderRegister(
        name=current_provider.name,
        with_oauth=current_provider.with_oauth,
        client_id_env_var=current_provider.client_id_env_var,
        client_secret_env_var=current_provider.client_secret_env_var,
        with_api_key=current_provider.with_api_key,
        api_key_env_var=current_provider.api_key_env_var,
        with_user_password=current_provider.with_user_password,
        username_env_var=current_provider.username_env_var,
        password_env_var=current_provider.password_env_var,
    )
    if current_provider_register == new_provider:
        return False
    return True


def find_delta_providers(
    current_providers: Dict[str, PrismaProviderRegistry],
    providers: Dict[str, ProviderRegister],
) -> Dict[str, ProviderRegister]:
    """
    Identify providers that are either new or updated compared to the current providers list.

    Args:
        current_providers (Dict[str, PrismaProviderRegistry]): Dictionary of current provider models keyed by provider name.
        providers (Dict[str, ProviderRegister]): Dictionary of new provider registrations keyed by provider name.

    Returns:
        Dict[str, ProviderRegister]: Providers that need to be added/updated in the registry.
            - Includes providers not in current_providers.
            - Includes providers where the data differs from what's in current_providers.
    """
    provider_update = {}
    for name, provider in providers.items():
        if name not in current_providers:
            provider_update[name] = provider
        else:
            if is_providers_different(current_providers[name], provider):
                provider_update[name] = provider

    return provider_update


async def get_providers() -> Dict[str, PrismaProviderRegistry]:
    """
    Retrieve all provider registries from the database.

    Returns:
        Dict[str, PrismaProviderRegistry]: Dictionary of all current providers, keyed by provider name.
    """
    async with Prisma() as prisma:
        providers = await prisma.providerregistry.find_many()
        return {
            provider.name: PrismaProviderRegistry(**provider.model_dump())
            for provider in providers
        }


async def upsert_providers_change_bulk(providers: Dict[str, ProviderRegister]):
    """
    Bulk upsert providers into the database after checking for changes.

    Args:
        providers (Dict[str, ProviderRegister]): Dictionary of new provider registrations keyed by provider name.
    """
    current_providers = await get_providers()
    provider_update = find_delta_providers(current_providers, providers)
    """Async version of bulk upsert providers with all fields using transaction for atomicity"""
    async with Prisma() as prisma:
        async with prisma.tx() as tx:
            results = []
            for name, provider in provider_update.items():
                result = await tx.providerregistry.upsert(
                    where={"name": name},
                    data={
                        "create": {
                            "name": name,
                            "with_oauth": provider.with_oauth,
                            "client_id_env_var": provider.client_id_env_var,
                            "client_secret_env_var": provider.client_secret_env_var,
                            "with_api_key": provider.with_api_key,
                            "api_key_env_var": provider.api_key_env_var,
                            "with_user_password": provider.with_user_password,
                            "username_env_var": provider.username_env_var,
                            "password_env_var": provider.password_env_var,
                        },
                        "update": {
                            "with_oauth": provider.with_oauth,
                            "client_id_env_var": provider.client_id_env_var,
                            "client_secret_env_var": provider.client_secret_env_var,
                            "with_api_key": provider.with_api_key,
                            "api_key_env_var": provider.api_key_env_var,
                            "with_user_password": provider.with_user_password,
                            "username_env_var": provider.username_env_var,
                            "password_env_var": provider.password_env_var,
                        },
                    },
                )
                results.append(result)
            return results
