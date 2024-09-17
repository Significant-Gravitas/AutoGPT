import typing
import fastapi

import autogpt_libs.auth.middleware
import autogpt_server.data.credit
import autogpt_server.data.user
import autogpt_server.server.utils


router = fastapi.APIRouter()

_user_credit_model = autogpt_server.data.credit.get_user_credit_model()


@router.get("/")
async def root():
    return {"message": "Welcome to the Autogpt Server API"}


@router.post("/auth/user")
async def get_or_create_user_route(
    user_data: dict = fastapi.Depends(autogpt_libs.auth.middleware.auth_middleware),
):
    user = await autogpt_server.data.user.get_or_create_user(user_data)
    return user.model_dump()


@router.get("/credits")
async def get_user_credits(
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_server.server.utils.get_user_id)
    ]
):
    return {"credits": await _user_credit_model.get_or_refill_credit(user_id)}