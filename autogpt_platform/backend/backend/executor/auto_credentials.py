"""Auto-credential resolution for picker-style block inputs (e.g. Google Drive).

Shared between the graph executor (``backend/executor/manager.py``) and the
CoPilot direct-block-execution path (``backend/copilot/tools/helpers.py``)
so both handle ``_credentials_id`` payloads identically.
"""

import logging
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
            provider = info.get("config", {}).get("provider", "external service")
            field_data = input_data.get(field_name)

            if field_data and isinstance(field_data, dict):
                if "_credentials_id" in field_data:
                    cred_id = field_data["_credentials_id"]
                    if cred_id is None:
                        # Explicitly None means the value is being chained in
                        # at execution time from an upstream block — skip.
                        continue
                    if not isinstance(cred_id, str) or not cred_id.strip():
                        file_name = field_data.get("name", "selected file")
                        raise ValueError(
                            f"{provider.capitalize()} credential id for "
                            f"'{file_name}' in field '{field_name}' is empty "
                            f"or invalid. Please open the agent in the "
                            f"builder and re-select the file."
                        )
                    file_name = field_data.get("name", "selected file")
                    try:
                        credentials, lock = await creds_manager.acquire(
                            user_id, cred_id
                        )
                        locks.append(lock)
                        extra_exec_kwargs[kwarg_name] = credentials
                    except ValueError:
                        raise ValueError(
                            f"{provider.capitalize()} credentials for "
                            f"'{file_name}' in field '{field_name}' are not "
                            f"available in your account. "
                            f"This can happen if the agent was created by "
                            f"another user or the credentials were deleted. "
                            f"Please open the agent in the builder and "
                            f"re-select the file to authenticate with your "
                            f"own account."
                        )
                else:
                    file_name = field_data.get("name", "selected file")
                    raise MissingAutoCredentialsError(
                        f"Authentication missing for '{file_name}' in field "
                        f"'{field_name}'. The CoPilot chat will render the "
                        f"{provider.capitalize()} picker inline — pick the "
                        f"file there; re-invoking `run_block` with a bare "
                        f"id/URL will not authenticate."
                    )
            elif field_data is None and field_name not in input_data:
                # Field not in input_data at all = connected from upstream, skip
                pass
            elif field_data is None or field_data == "":
                raise MissingAutoCredentialsError(
                    f"No file selected for '{field_name}'. The CoPilot chat "
                    f"will render the {provider.capitalize()} picker inline "
                    f"— pick the file there; `run_block` will re-run "
                    f"automatically with the populated value."
                )
            else:
                raise ValueError(
                    f"Invalid {type(field_data).__name__} value for "
                    f"'{field_name}': this field expects a picker-populated "
                    f"object carrying the user's credentials, not a bare "
                    f"value. Please re-select the file via the picker to "
                    f"provide {provider.capitalize()} authentication."
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
