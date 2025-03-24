import logging
import typing

from autogpt_libs.auth import requires_admin_user
from autogpt_libs.auth.depends import get_user_id
from backend.server.model import Pagination
from backend.server.v2.admin.model import (
    AddUserCreditsResponse,
    GrantHistoryResponse,
    UserBalanceResponse,
)
from fastapi import APIRouter, Body, Depends
from prisma.enums import CreditTransactionType

from backend.data.credit import admin_get_user_balances, get_user_credit_model

logger = logging.getLogger(__name__)

_user_credit_model = get_user_credit_model()


router = APIRouter(
    prefix="/admin",
    tags=["credits", "admin"],
    dependencies=[Depends(requires_admin_user)],
)


@router.post("/add_credits", response_model=AddUserCreditsResponse)
async def add_user_credits(
    user_id: typing.Annotated[str, Body()],
    amount: typing.Annotated[int, Body()],
    comments: typing.Annotated[str, Body()],
    admin_user: typing.Annotated[
        str,
        Depends(get_user_id),
    ],
):
    """ """
    logger.info(f"Admin user {admin_user} is adding {amount} credits to user {user_id}")
    new_balance, transaction_key = await _user_credit_model._add_transaction(
        user_id,
        amount,
        CreditTransactionType.GRANT,
    )
    return {
        "success": True,
        "new_balance": new_balance,
        "transaction_key": transaction_key,
    }


@router.get(
    "/user_balances",
    dependencies=[Depends(requires_admin_user)],
    response_model=UserBalanceResponse,
)
async def get_user_balances(
    admin_user: typing.Annotated[
        str,
        Depends(get_user_id),
    ],
    search: typing.Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
):
    """ """
    logger.info(f"Admin user {admin_user} is getting user balances")

    try:
        resp = await admin_get_user_balances(
            page=page,
            page_size=page_size,
            search=search,
        )
        logger.info(f"Admin user {admin_user} got {len(resp.balances)} user balances")
        return resp
    except Exception as e:
        logger.exception(f"Error getting user balances: {e}")
        raise


@router.get(
    "/grant_history",
    response_model=GrantHistoryResponse,
)
async def get_grant_history(
    admin_user: typing.Annotated[
        str,
        Depends(get_user_id),
    ],
    search: typing.Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
):
    """ """
    logger.info(f"Admin user {admin_user} is getting grant history")

    return GrantHistoryResponse(
        grants=[],
        pagination=Pagination(
            total_items=0,
            total_pages=0,
            current_page=page,
            page_size=page_size,
        ),
    )
