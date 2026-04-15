import logging
import typing

from autogpt_libs.auth import get_user_id, requires_admin_user
from fastapi import APIRouter, Body, Security
from prisma.enums import CreditTransactionType

from backend.data.credit import admin_get_user_history, get_user_credit_model
from backend.util.json import SafeJson

from .model import AddUserCreditsResponse, UserHistoryResponse

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/admin",
    tags=["credits", "admin"],
    dependencies=[Security(requires_admin_user)],
)


@router.post(
    "/add_credits", response_model=AddUserCreditsResponse, summary="Add Credits to User"
)
async def add_user_credits(
    user_id: typing.Annotated[str, Body()],
    amount: typing.Annotated[int, Body()],
    comments: typing.Annotated[str, Body()],
    admin_user_id: str = Security(get_user_id),
):
    logger.info(
        f"Admin user {admin_user_id} is adding {amount} credits to user {user_id}"
    )
    user_credit_model = await get_user_credit_model(user_id)
    new_balance, transaction_key = await user_credit_model._add_transaction(
        user_id,
        amount,
        transaction_type=CreditTransactionType.GRANT,
        metadata=SafeJson({"admin_id": admin_user_id, "reason": comments}),
    )
    return {
        "new_balance": new_balance,
        "transaction_key": transaction_key,
    }


@router.get(
    "/users_history",
    response_model=UserHistoryResponse,
    summary="Get All Users History",
)
async def admin_get_all_user_history(
    admin_user_id: str = Security(get_user_id),
    search: typing.Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    transaction_filter: typing.Optional[CreditTransactionType] = None,
):
    """ """
    logger.info(f"Admin user {admin_user_id} is getting grant history")

    try:
        resp = await admin_get_user_history(
            page=page,
            page_size=page_size,
            search=search,
            transaction_filter=transaction_filter,
        )
        logger.info(f"Admin user {admin_user_id} got {len(resp.history)} grant history")
        return resp
    except Exception as e:
        logger.exception(f"Error getting grant history: {e}")
        raise e
