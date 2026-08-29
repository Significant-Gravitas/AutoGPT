import pydantic


class PushSubscriptionKeys(pydantic.BaseModel):
    p256dh: str = pydantic.Field(min_length=1, max_length=512)
    auth: str = pydantic.Field(min_length=1, max_length=512)


class PushSubscribeRequest(pydantic.BaseModel):
    endpoint: str = pydantic.Field(min_length=1, max_length=2048)
    keys: PushSubscriptionKeys
    user_agent: str | None = pydantic.Field(default=None, max_length=512)


class PushUnsubscribeRequest(pydantic.BaseModel):
    endpoint: str = pydantic.Field(min_length=1, max_length=2048)


class VapidPublicKeyResponse(pydantic.BaseModel):
    public_key: str
