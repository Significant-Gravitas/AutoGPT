from typing import Dict

from prisma import Prisma

from backend.sdk.provider import ProviderRegister


async def upsert_providers_bulk(providers: Dict[str, ProviderRegister]):
    """Async version of bulk upsert providers with all fields using transaction for atomicity"""
    async with Prisma() as prisma:
        async with prisma.tx() as tx:
            results = []
            for name, provider in providers.items():
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
