"""The credentials a block run needs, loaded for the user who runs it.

Shared by the graph executor and the direct block-execute routes, so a block gets
its credentials the same way however it is run.
"""

import inspect
import logging
from typing import Any

from redis.asyncio.lock import Lock as AsyncRedisLock

from backend.blocks._base import AnyBlockSchema, BlockSchema
from backend.data.block import BlockInput
from backend.data.db_accessors import experts_db
from backend.data.model import (
    Credentials,
    CredentialsMetaInput,
    OAuth2Credentials,
    is_credentials_field_name,
)
from backend.integrations.codex.access import enforce_codex_access
from backend.integrations.credential_lease import CredentialLease
from backend.integrations.credentials_store import (
    is_system_credential,
    provider_matches,
)
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.util.logging import TruncatedLogger
from backend.util.request import HTTPClientError

from .auto_credentials import acquire_auto_credentials

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[BlockCredentials]")


class CredentialsUnavailableError(ValueError):
    """A credentials field names a credential the user can't run the block with."""

    def __init__(self, field_name: str, message: str):
        super().__init__(message)
        self.field_name = field_name


class DirectBlockCredentialsError(ValueError):
    """A direct block run's input does not give the block the credentials it needs."""


class BlockCredentials:
    """The credentials a block run holds; ``release`` hands back their leases and locks."""

    def __init__(
        self,
        exec_kwargs: dict[str, Any],
        leases: list[CredentialLease],
        locks: list[AsyncRedisLock],
    ):
        self.exec_kwargs = exec_kwargs
        self.leases = leases
        self.locks = locks

    async def release(self, log: TruncatedLogger = logger) -> None:
        await _release_credential_leases(self.leases, log)
        for lock in self.locks:
            if lock and (await lock.locked()) and (await lock.owned()):
                try:
                    await lock.release()
                except Exception as e:
                    log.error(f"Failed to release credentials lock: {e}")


async def acquire_block_credentials(
    input_model: type[BlockSchema],
    input_data: BlockInput,
    creds_manager: IntegrationCredentialsManager,
    user_id: str,
    *,
    expert_id: str | None = None,
    dry_run_credentials: Credentials | None = None,
    log: TruncatedLogger = logger,
) -> BlockCredentials:
    """Load the credentials ``input_data`` names, as ``run()`` keyword arguments.

    Normalizes the credentials fields of ``input_data`` in place. OAuth credentials
    come with a system-wide lease that stops them being refreshed or replaced during
    execution. ⚠️ This means an OAuth credential can only be used by one (running)
    block at a time; simultaneous execution of blocks using the same one is not
    supported.
    """
    credential_fields_info = input_model.get_credentials_fields_info()
    exec_kwargs: dict[str, Any] = {}
    credential_leases: dict[str, CredentialLease] = {}
    runtime_credential_leases: dict[str, CredentialLease] = {}
    expert_allowed_credentials: set[str] | None = None
    try:
        # Handle regular credentials fields
        for field_name, input_type in input_model.get_credentials_fields().items():
            # Dry-run platform credentials bypass the credential store.
            # Keep the existing credential metadata so _execute's input_schema(**...)
            # doesn't fail on the required field.  If no metadata is present,
            # synthesize a minimal placeholder from the platform credentials.
            if dry_run_credentials is not None:
                if input_data.get(field_name) is None:
                    input_data[field_name] = {
                        "id": dry_run_credentials.id,
                        "provider": dry_run_credentials.provider,
                        "type": dry_run_credentials.type,
                        "title": dry_run_credentials.title,
                    }
                exec_kwargs[field_name] = dry_run_credentials
                continue

            field_value = input_data.get(field_name)
            if not field_value or (
                isinstance(field_value, dict) and not field_value.get("id")
            ):
                # No credentials configured — nullify so JSON schema validation
                # doesn't choke on the empty default `{}`.
                input_data[field_name] = None
                continue  # Block runs without credentials

            credentials_meta = input_type(**field_value)
            # An expert run uses only the credentials granted to it.
            if expert_id is not None and not is_system_credential(credentials_meta.id):
                if expert_allowed_credentials is None:
                    expert_allowed_credentials = set(
                        await experts_db().expert_allowed_credential_ids(
                            user_id, expert_id
                        )
                    )
                if credentials_meta.id not in expert_allowed_credentials:
                    raise CredentialsUnavailableError(
                        field_name,
                        f"Credentials #{credentials_meta.id} in field "
                        f"'{field_name}' have not been granted to this expert. "
                        f"Grant them on the expert's Integrations page.",
                    )
            # Write normalized values back so JSON schema validation also passes
            # (model_validator may have fixed legacy formats like "ProviderName.MCP")
            input_data[field_name] = credentials_meta.model_dump(mode="json")
            field_info = credential_fields_info[field_name]
            if field_info.credential_reference_only:
                credentials = await creds_manager.get(
                    user_id,
                    credentials_meta.id,
                )
                if not (
                    credentials is not None
                    and provider_matches(
                        credentials.provider,
                        credentials_meta.provider,
                    )
                    and credentials.type == credentials_meta.type
                ):
                    raise CredentialsUnavailableError(
                        field_name,
                        f"Credentials #{credentials_meta.id} for user #{user_id} not found",
                    )
                if provider_matches(credentials.provider, "codex"):
                    await enforce_codex_access(user_id)
                continue
            try:
                credentials, lease = await _get_node_credentials(
                    creds_manager, user_id, credentials_meta
                )
            except ValueError as e:
                # Credential was deleted or doesn't exist.
                # If the field has a default, run without credentials.
                if input_model.model_fields[field_name].default is not None:
                    log.warning(
                        f"Credentials #{credentials_meta.id} not found, "
                        "running without (field has default)"
                    )
                    input_data[field_name] = None
                    continue
                raise CredentialsUnavailableError(field_name, str(e)) from e
            if lease is not None:
                credential_leases[field_name] = lease
            if not (
                provider_matches(
                    credentials.provider,
                    credentials_meta.provider,
                )
                and credentials.type == credentials_meta.type
            ):
                raise CredentialsUnavailableError(
                    field_name,
                    f"Credentials #{credentials_meta.id} for user #{user_id} not found",
                )
            if provider_matches(credentials.provider, "codex"):
                await enforce_codex_access(user_id)
            exec_kwargs[field_name] = credentials
            if lease is not None and _uses_provider_runtime(lease):
                runtime_credential_leases[field_name] = lease

        # Handle auto-generated credentials (e.g., from GoogleDriveFileInput)
        auto_extra_kwargs, auto_locks = await acquire_auto_credentials(
            input_model=input_model,
            input_data=input_data,
            creds_manager=creds_manager,
            user_id=user_id,
            expert_id=expert_id,
        )
        exec_kwargs.update(auto_extra_kwargs)
    except BaseException:
        await _release_credential_leases(list(credential_leases.values()), log)
        raise

    if runtime_credential_leases:
        exec_kwargs["credential_leases"] = runtime_credential_leases
    return BlockCredentials(
        exec_kwargs, list(credential_leases.values()), list(auto_locks)
    )


async def acquire_direct_block_credentials(
    block: AnyBlockSchema, input_data: BlockInput, user_id: str
) -> BlockCredentials:
    """The caller's stored credentials for a block run outside any graph.

    Raises:
        DirectBlockCredentialsError: A credentials field the block's ``run()``
            requires is empty, or names a credential the caller doesn't have or
            one of another provider or type. The app answers it with a 400.
    """
    try:
        credentials = await acquire_block_credentials(
            block.input_schema, input_data, IntegrationCredentialsManager(), user_id
        )
    except CredentialsUnavailableError as e:
        raise DirectBlockCredentialsError(
            f"The credentials in field '{e.field_name}' are not in your account, "
            "or are not of the provider and type that field takes."
        ) from e
    except HTTPClientError as e:
        raise DirectBlockCredentialsError(
            "A credential this block needs was refused by its provider. "
            "Reconnect it, then try again."
        ) from e
    if unfilled := _unfilled_credentials_fields(block, credentials.exec_kwargs):
        await credentials.release()
        raise DirectBlockCredentialsError(
            f"This block needs credentials in field {', '.join(map(repr, unfilled))}: "
            "pass the id, provider and type of one of your stored credentials."
        )
    return credentials


def _unfilled_credentials_fields(
    block: AnyBlockSchema, exec_kwargs: dict[str, Any]
) -> list[str]:
    """Input fields whose credentials ``block.run`` requires but did not get."""
    picker_fields = {
        kwarg: info["field_name"]
        for kwarg, info in block.input_schema.get_auto_credentials_fields().items()
    }
    params = inspect.signature(block.run).parameters
    return [
        picker_fields.get(name, name)
        for name, param in params.items()
        if is_credentials_field_name(name)
        and param.default is param.empty
        and name not in exec_kwargs
    ]


async def _get_node_credentials(
    creds_manager: IntegrationCredentialsManager,
    user_id: str,
    credentials_meta: CredentialsMetaInput,
) -> tuple[Credentials, CredentialLease | None]:
    """Load a node's credentials, leased only if they can change mid-run.

    Only OAuth credentials change under a running block: a refresh replaces the
    tokens, and a provider runtime checkpoints rotated ones through the lease.
    A lease on any other kind would protect nothing, and would make every other
    run of the same credential wait for this one. The caller
    checks that the stored type matches ``credentials_meta``, so an OAuth
    credential never runs unleased.
    """
    if credentials_meta.type == "oauth2":
        lease = await creds_manager.acquire_lease(user_id, credentials_meta.id)
        return lease.credentials, lease
    credentials = await creds_manager.get(user_id, credentials_meta.id)
    if credentials is None:
        raise ValueError(
            f"Credentials #{credentials_meta.id} for user #{user_id} not found"
        )
    return credentials, None


async def _release_credential_leases(
    leases: list[CredentialLease], log: TruncatedLogger
) -> None:
    for lease in leases:
        try:
            await lease.release()
        except Exception as error:
            log.error(f"Failed to release credentials lease: {error}")


def _uses_provider_runtime(lease: CredentialLease) -> bool:
    credentials = lease.credentials
    return (
        isinstance(credentials, OAuth2Credentials)
        and credentials.refresh_strategy == "provider_runtime"
    )
