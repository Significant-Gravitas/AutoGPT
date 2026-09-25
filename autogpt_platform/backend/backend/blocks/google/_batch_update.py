"""Shared pieces of the raw batch-update blocks for Google Docs and Sheets."""

from googleapiclient.errors import HttpError

from backend.util.exceptions import BlockExecutionError


def batch_update_error(
    exc: HttpError, product: str, block_name: str, block_id: str
) -> BlockExecutionError:
    """Turn a batchUpdate error into a message the user (or the copilot) can act on.

    A 400 keeps Google's own message: it names the request and field that were
    wrong, which is what's needed to fix the request list.
    """
    if exc.status_code == 400:
        message = f"Google rejected the {product} update, so nothing was changed: {exc.reason}"
    elif exc.status_code == 404:
        message = (
            f"Couldn't find that {product} file, or the connected Google account "
            "can't open it."
        )
    elif exc.status_code == 403 and "insufficient" in str(exc.reason).lower():
        message = (
            "The connected Google account hasn't granted the access this block "
            "needs. Reconnect Google and approve it."
        )
    elif exc.status_code == 403:
        message = (
            f"The connected Google account can't edit this {product} file: {exc.reason}"
        )
    else:
        message = f"Google {product} API error {exc.status_code}: {exc.reason}"
    return BlockExecutionError(
        message=message, block_name=block_name, block_id=block_id
    )
