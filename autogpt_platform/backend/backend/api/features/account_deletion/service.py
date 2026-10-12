"""Placeholder: the deletion service is written in the next stage."""

from prisma.enums import AccountDeletionSource


class EmailMismatchError(Exception):
    """The typed confirmation email isn't the account's email."""


class PlanCancellationError(Exception):
    """Stripe refused to cancel the plan; nothing else was changed."""


async def delete_account(
    user_id: str, confirm_email: str, source: AccountDeletionSource
) -> None:
    raise NotImplementedError
