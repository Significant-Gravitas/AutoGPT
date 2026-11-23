"""The credentials a block runs with.

A block's credentials input names one of the user's stored credentials by
provider, type and id, and the block's ``run`` gets them as the keyword argument
of the same name. Loading them from the store is the ownership check: it looks
only among the user's own credentials and the platform's, so another user's
credential id finds nothing. They must also be the provider and type the input
names, an API key must not have expired, and ChatGPT credentials need a plan
that allows them.

The graph executor runs each node on what ``acquire_block_credentials`` gives
it. Direct block execution (``backend.api.features.blocks.direct_run``) makes
the same checks with ``check_credentials_input`` before it charges for a run,
then runs the block with ``execute_with_credentials``, on
``acquire_block_credentials`` too.
"""

from typing import Any

from pydantic_core import PydanticUndefined
from redis.asyncio.lock import Lock as AsyncRedisLock

from backend.blocks._base import AnyBlockSchema, BlockSchema
from backend.data.block import BlockInput, BlockOutput
from backend.data.db_accessors import experts_db
from backend.data.execution import ExecutionContext
from backend.data.model import (
    APIKeyCredentials,
    Credentials,
    CredentialsFieldInfo,
    CredentialsMetaInput,
    OAuth2Credentials,
)
from backend.executor.auto_credentials import acquire_auto_credentials
from backend.integrations.codex.access import enforce_codex_access
from backend.integrations.credential_lease import CredentialLease
from backend.integrations.credentials_store import (
    is_system_credential,
    provider_matches,
)
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.util.logging import TruncatedLogger


class BlockCredentials:
    """The credentials one block run was given: the keyword arguments that pass
    them to ``run``, and the leases and locks it holds on them until it ends."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}
        self.leases: list[CredentialLease] = []
        self.locks: list[AsyncRedisLock] = []

    async def release(self, log_metadata: TruncatedLogger) -> None:
        await _release_credential_leases(self.leases, log_metadata)
        for lock in self.locks:
            if lock and (await lock.locked()) and (await lock.owned()):
                try:
                    await lock.release()
                except Exception as e:
                    log_metadata.error(f"Failed to release credentials lock: {e}")


async def acquire_block_credentials(
    input_model: type[BlockSchema],
    input_data: BlockInput,
    *,
    user_id: str,
    creds_manager: IntegrationCredentialsManager,
    log_metadata: TruncatedLogger,
    expert_id: str | None = None,
    user_timezone: str = "UTC",
    check_expiry: bool = True,
    substitute: Credentials | None = None,
) -> BlockCredentials:
    """Load the credentials each credentials input of *input_data* names, for a
    run that releases them when it ends.

    OAuth credentials come with a system-wide lease that stops them being
    refreshed or replaced during the run. ⚠️ This means an OAuth credential can
    only be used by one (running) block at a time.

    Each credentials input is written back normalized, or as None when it
    names no credentials, so the block's schema accepts it. ``expert_id`` holds
    the run to the credentials granted to that expert. ``check_expiry=False``
    lets an expired API key through, for a run that won't use it.
    ``substitute`` gives every credentials input those credentials instead, as
    a dry run on the platform's key does.

    Releases what it took before raising what ``check_credentials_input``
    raises, or a picker input's error.
    """
    credentials = BlockCredentials()
    runtime_leases: dict[str, CredentialLease] = {}
    loader = _CredentialsLoader(
        user_id=user_id,
        creds_manager=creds_manager,
        log_metadata=log_metadata,
        expert_id=expert_id,
        user_timezone=user_timezone,
        check_expiry=check_expiry,
        lease=True,
    )
    fields_info = input_model.get_credentials_fields_info()
    try:
        for field_name, input_type in input_model.get_credentials_fields().items():
            # Dry-run platform credentials bypass the credential store.
            # Keep the existing credential metadata so _execute's input_schema(**...)
            # doesn't fail on the required field.  If no metadata is present,
            # synthesize a minimal placeholder from the platform credentials.
            if substitute is not None:
                if input_data.get(field_name) is None:
                    input_data[field_name] = {
                        "id": substitute.id,
                        "provider": substitute.provider,
                        "type": substitute.type,
                        "title": substitute.title,
                    }
                credentials.kwargs[field_name] = substitute
                continue

            loaded, lease = await loader.load(
                input_model, field_name, input_type, fields_info[field_name], input_data
            )
            if lease is not None:
                credentials.leases.append(lease)
                if _uses_provider_runtime(lease):
                    runtime_leases[field_name] = lease
            if loaded is not None:
                credentials.kwargs[field_name] = loaded

        # Handle auto-generated credentials (e.g., from GoogleDriveFileInput)
        auto_kwargs, auto_locks = await acquire_auto_credentials(
            input_model=input_model,
            input_data=input_data,
            creds_manager=creds_manager,
            user_id=user_id,
            expert_id=expert_id,
        )
        credentials.kwargs.update(auto_kwargs)
        credentials.locks.extend(auto_locks)
    except BaseException:
        await _release_credential_leases(credentials.leases, log_metadata)
        raise

    if runtime_leases:
        credentials.kwargs["credential_leases"] = runtime_leases
    return credentials


async def execute_with_credentials(
    block: AnyBlockSchema,
    input_data: BlockInput,
    *,
    user_id: str,
    execution_context: ExecutionContext,
    log_metadata: TruncatedLogger,
) -> BlockOutput:
    """*block*'s outputs, run outside a graph on the credentials its inputs name,
    which are loaded on the loop that iterates this and released when the block
    stops."""
    credentials = await acquire_block_credentials(
        block.input_schema,
        input_data,
        user_id=user_id,
        creds_manager=IntegrationCredentialsManager(),
        log_metadata=log_metadata,
        user_timezone=execution_context.user_timezone,
    )
    try:
        async for output in block.execute(
            input_data,
            user_id=user_id,
            execution_context=execution_context,
            **credentials.kwargs,
        ):
            yield output
    finally:
        await credentials.release(log_metadata)


async def check_credentials_input(
    input_model: type[BlockSchema],
    field_name: str,
    input_data: BlockInput,
    *,
    user_id: str,
    creds_manager: IntegrationCredentialsManager,
    log_metadata: TruncatedLogger,
    user_timezone: str = "UTC",
) -> None:
    """Check the credentials one credentials input names, as
    ``acquire_block_credentials`` would load them, without leasing them: a
    lease another run holds is waited on only for a token's refresh, and then
    for a few seconds at most.

    Writes the input back normalized, or as None when it names no credentials.

    Raises:
        ValueError: the credentials aren't the user's, or aren't the provider
            and type the input names; also a malformed input (a pydantic
            ``ValidationError``), an expired API key (``CredentialsExpiredError``)
            and a platform key no block may use (``SharedCredentialsRefusedError``).
        HTTPClientError: the provider refused to refresh an OAuth token.
        TimeoutError: an OAuth token due a refresh is leased to another run.
        UserPaywalledError: ChatGPT credentials, on a plan that doesn't allow them.
    """
    loader = _CredentialsLoader(
        user_id=user_id,
        creds_manager=creds_manager,
        log_metadata=log_metadata,
        expert_id=None,
        user_timezone=user_timezone,
        check_expiry=True,
        lease=False,
    )
    await loader.load(
        input_model,
        field_name,
        input_model.get_credentials_fields()[field_name],
        input_model.get_credentials_fields_info()[field_name],
        input_data,
    )


def names_credentials(field_value: Any) -> bool:
    """Whether a credentials input names any credentials to load."""
    return bool(field_value) and not (
        isinstance(field_value, dict) and not field_value.get("id")
    )


class _CredentialsLoader:
    """Loads the credentials a run's credentials inputs name, for one user."""

    def __init__(
        self,
        *,
        user_id: str,
        creds_manager: IntegrationCredentialsManager,
        log_metadata: TruncatedLogger,
        expert_id: str | None,
        user_timezone: str,
        check_expiry: bool,
        lease: bool,
    ) -> None:
        self.user_id = user_id
        self.creds_manager = creds_manager
        self.log_metadata = log_metadata
        self.expert_id = expert_id
        self.user_timezone = user_timezone
        self.check_expiry = check_expiry
        self.lease = lease
        self._expert_allowed_credentials: set[str] | None = None

    async def load(
        self,
        input_model: type[BlockSchema],
        field_name: str,
        input_type: type[CredentialsMetaInput],
        field_info: CredentialsFieldInfo,
        input_data: BlockInput,
    ) -> tuple[Credentials | None, CredentialLease | None]:
        """The credentials *field_name* names, with their lease if they are
        leased. None when the input names none, so the block runs without them,
        and for an input that only references its credentials."""
        user_id = self.user_id
        field_value = input_data.get(field_name)
        if not names_credentials(field_value):
            # No credentials configured — nullify so JSON schema validation
            # doesn't choke on the empty default `{}`.
            input_data[field_name] = None
            return None, None  # Block runs without credentials

        credentials_meta = input_type.model_validate(field_value)
        # An expert run uses only the credentials granted to it.
        await self._check_granted_to_expert(credentials_meta, field_name)
        # Write normalized values back so JSON schema validation also passes
        # (model_validator may have fixed legacy formats like "ProviderName.MCP")
        input_data[field_name] = credentials_meta.model_dump(mode="json")
        if field_info.credential_reference_only:
            credentials = await self._get(credentials_meta.id)
            if not (
                credentials is not None
                and provider_matches(
                    credentials.provider,
                    credentials_meta.provider,
                )
                and credentials.type == credentials_meta.type
            ):
                raise ValueError(
                    f"Credentials #{credentials_meta.id} for user #{user_id} not found"
                )
            if provider_matches(credentials.provider, "codex"):
                await enforce_codex_access(user_id)
            return None, None

        try:
            credentials, lease = await self._load(credentials_meta)
        except ValueError:
            # Credential was deleted or doesn't exist.
            # If the field has a default, run without credentials; a required
            # field's default is PydanticUndefined, which is no default.
            default = input_model.model_fields[field_name].default
            if default is not None and default is not PydanticUndefined:
                self.log_metadata.warning(
                    f"Credentials #{credentials_meta.id} not found, "
                    "running without (field has default)"
                )
                input_data[field_name] = None
                return None, None
            raise
        try:
            if not (
                provider_matches(
                    credentials.provider,
                    credentials_meta.provider,
                )
                and credentials.type == credentials_meta.type
            ):
                raise ValueError(
                    f"Credentials #{credentials_meta.id} for user #{user_id} not found"
                )
            if (
                self.check_expiry
                and isinstance(credentials, APIKeyCredentials)
                and credentials.is_expired()
            ):
                # Outside the `except ValueError` above, so a field with a
                # default can't fall back to running without the key.
                raise credentials.expired_error(self.user_timezone)
            if provider_matches(credentials.provider, "codex"):
                await enforce_codex_access(user_id)
        except BaseException:
            if lease is not None:
                await _release_credential_leases([lease], self.log_metadata)
            raise
        return credentials, lease

    async def _check_granted_to_expert(
        self, credentials_meta: CredentialsMetaInput, field_name: str
    ) -> None:
        if self.expert_id is None or is_system_credential(credentials_meta.id):
            return
        if self._expert_allowed_credentials is None:
            self._expert_allowed_credentials = set(
                await experts_db().expert_allowed_credential_ids(
                    self.user_id, self.expert_id
                )
            )
        if credentials_meta.id not in self._expert_allowed_credentials:
            raise ValueError(
                f"Credentials #{credentials_meta.id} in field "
                f"'{field_name}' have not been granted to this expert. "
                f"Grant them on the expert's Integrations page."
            )

    async def _get(self, credentials_id: str) -> Credentials | None:
        if self.lease:
            return await self.creds_manager.get(self.user_id, credentials_id)
        # Without a lease, refreshing a token waits on a run that holds its
        # lease for a few seconds at most (TimeoutError), not until it ends.
        return await self.creds_manager.get(self.user_id, credentials_id, lock=False)

    async def _load(
        self, credentials_meta: CredentialsMetaInput
    ) -> tuple[Credentials, CredentialLease | None]:
        if not self.lease:
            credentials = await self._get(credentials_meta.id)
            if credentials is None:
                raise ValueError(
                    f"Credentials #{credentials_meta.id} for user #{self.user_id} "
                    "not found"
                )
            return credentials, None
        return await _get_node_credentials(
            self.creds_manager, self.user_id, credentials_meta
        )


async def _get_node_credentials(
    creds_manager: IntegrationCredentialsManager,
    user_id: str,
    credentials_meta: CredentialsMetaInput,
) -> tuple[Credentials, CredentialLease | None]:
    """Load a node's credentials, leased only if they can change mid-run.

    Only OAuth credentials change under a running block: a refresh replaces the
    tokens, and a provider runtime checkpoints rotated ones through the lease.
    A lease on any other kind would protect nothing, and would make every other
    node on the same credential wait for this one, up to the lock's 60 s TTL,
    including the tools an Orchestrator runs on its own API key. The caller
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
    leases: list[CredentialLease], log_metadata: TruncatedLogger
) -> None:
    for lease in leases:
        try:
            await lease.release()
        except Exception as error:
            log_metadata.error(f"Failed to release credentials lease: {error}")


def _uses_provider_runtime(lease: CredentialLease) -> bool:
    credentials = lease.credentials
    return (
        isinstance(credentials, OAuth2Credentials)
        and credentials.refresh_strategy == "provider_runtime"
    )
