"""Auto-credential resolution for picker-style block inputs (e.g. Google Drive).

Shared between the graph executor (``backend/executor/manager.py``) and the
CoPilot direct-block-execution path (``backend/copilot/tools/helpers.py``)
so both handle ``_credentials_id`` payloads identically.
"""

import logging
from dataclasses import dataclass
from typing import Any

from redis.asyncio.lock import Lock as AsyncRedisLock

from backend.blocks._base import BlockSchema
from backend.integrations.creds_manager import IntegrationCredentialsManager

logger = logging.getLogger(__name__)


class MissingAutoCredentialsError(ValueError):
    """Raised when a picker-style field lacks a usable ``_credentials_id``.

    Distinct from generic ``ValueError`` so callers (e.g. the CoPilot
    run_block path) can branch on "needs picker interaction" and return a
    structured response instead of a bare error.
    """


@dataclass(frozen=True)
class AutoCredentialFieldInfo:
    cred_id: str | None
    provider: str
    file_name: str
    field_name: str
    field_present: bool
    missing_credentials_id: bool = False
    explicit_none_credentials_id: bool = False
    invalid_credentials_id: bool = False
    missing_file_selection: bool = False
    invalid_value_type: str | None = None


def parse_auto_credential_field(
    field_name: str,
    info: dict[str, Any],
    field_data: Any,
    *,
    field_present_in_input: bool = True,
) -> AutoCredentialFieldInfo:
    provider = info.get("config", {}).get("provider", "external service")
    file_name = (
        field_data.get("name", "selected file")
        if isinstance(field_data, dict)
        else "selected file"
    )

    if isinstance(field_data, dict):
        if "_credentials_id" not in field_data:
            return AutoCredentialFieldInfo(
                cred_id=None,
                provider=provider,
                file_name=file_name,
                field_name=field_name,
                field_present=field_present_in_input,
                missing_credentials_id=True,
            )

        cred_id = field_data.get("_credentials_id")
        if cred_id is None:
            return AutoCredentialFieldInfo(
                cred_id=None,
                provider=provider,
                file_name=file_name,
                field_name=field_name,
                field_present=field_present_in_input,
                explicit_none_credentials_id=True,
            )
        if not isinstance(cred_id, str) or not cred_id.strip():
            return AutoCredentialFieldInfo(
                cred_id=None,
                provider=provider,
                file_name=file_name,
                field_name=field_name,
                field_present=field_present_in_input,
                invalid_credentials_id=True,
            )

        return AutoCredentialFieldInfo(
            cred_id=cred_id,
            provider=provider,
            file_name=file_name,
            field_name=field_name,
            field_present=field_present_in_input,
        )

    if field_data is None and not field_present_in_input:
        return AutoCredentialFieldInfo(
            cred_id=None,
            provider=provider,
            file_name=file_name,
            field_name=field_name,
            field_present=False,
        )

    if field_data is None or field_data == "":
        return AutoCredentialFieldInfo(
            cred_id=None,
            provider=provider,
            file_name=file_name,
            field_name=field_name,
            field_present=field_present_in_input,
            missing_file_selection=True,
        )

    return AutoCredentialFieldInfo(
        cred_id=None,
        provider=provider,
        file_name=file_name,
        field_name=field_name,
        field_present=field_present_in_input,
        invalid_value_type=type(field_data).__name__,
    )


async def acquire_auto_credentials(
    input_model: type[BlockSchema],
    input_data: dict[str, Any],
    creds_manager: IntegrationCredentialsManager,
    user_id: str,
) -> tuple[dict[str, Any], list[AsyncRedisLock]]:
    """Resolve ``auto_credentials`` from ``GoogleDriveFileField``-style inputs.

    Returns:
        (extra_exec_kwargs, locks): kwargs to inject into block execution,
        and credential locks to release after execution completes.

    Raises:
        MissingAutoCredentialsError: when a field is missing or lacks a
            ``_credentials_id``. Caller can decide whether to surface a
            picker UI or fail outright.
        ValueError: for other validation failures (invalid cred id, etc.).
    """
    extra_exec_kwargs: dict[str, Any] = {}
    locks: list[AsyncRedisLock] = []

    try:
        for kwarg_name, info in input_model.get_auto_credentials_fields().items():
            field_name = info["field_name"]
            field_data = input_data.get(field_name)
            parsed = parse_auto_credential_field(
                field_name=field_name,
                info=info,
                field_data=field_data,
                field_present_in_input=field_name in input_data,
            )

            if not parsed.field_present or parsed.explicit_none_credentials_id:
                continue
            if parsed.missing_credentials_id:
                raise MissingAutoCredentialsError(
                    f"Authentication missing for '{parsed.file_name}' in field "
                    f"'{parsed.field_name}'. The CoPilot chat will render the "
                    f"{parsed.provider.capitalize()} picker inline — pick the "
                    f"file there; re-invoking `run_block` with a bare "
                    f"id/URL will not authenticate."
                )
            if parsed.missing_file_selection:
                raise MissingAutoCredentialsError(
                    f"No file selected for '{parsed.field_name}'. The CoPilot chat "
                    f"will render the {parsed.provider.capitalize()} picker inline "
                    f"— pick the file there; `run_block` will re-run "
                    f"automatically with the populated value."
                )
            if parsed.invalid_credentials_id:
                raise ValueError(
                    f"{parsed.provider.capitalize()} credential id for "
                    f"'{parsed.file_name}' in field '{parsed.field_name}' is empty "
                    f"or invalid. Please open the agent in the "
                    f"builder and re-select the file."
                )
            if parsed.invalid_value_type:
                raise ValueError(
                    f"Invalid {parsed.invalid_value_type} value for "
                    f"'{parsed.field_name}': this field expects a picker-populated "
                    f"object carrying the user's credentials, not a bare "
                    f"value. Please re-select the file via the picker to "
                    f"provide {parsed.provider.capitalize()} authentication."
                )
            if not parsed.cred_id:
                continue

            try:
                credentials, lock = await creds_manager.acquire(user_id, parsed.cred_id)
                locks.append(lock)
                extra_exec_kwargs[kwarg_name] = credentials
            except ValueError:
                raise ValueError(
                    f"{parsed.provider.capitalize()} credentials for "
                    f"'{parsed.file_name}' in field '{parsed.field_name}' are not "
                    f"available in your account. "
                    f"This can happen if the agent was created by "
                    f"another user or the credentials were deleted. "
                    f"Please open the agent in the builder and "
                    f"re-select the file to authenticate with your "
                    f"own account."
                )
    except BaseException:
        # Release any locks already acquired so failures on later fields
        # don't strand earlier credentials until Redis TTL expires them.
        for lock in locks:
            try:
                await lock.release()
            except Exception as release_exc:
                logger.warning(
                    "Failed to release auto-credential lock after "
                    "acquisition error: %s",
                    release_exc,
                )
        raise

    return extra_exec_kwargs, locks
