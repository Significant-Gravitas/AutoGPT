from typing import Annotated

from autogpt_libs.auth.service import frontend_service_claims
from fastapi import Depends, Header, Request

CLIENT_IP_SCOPE = "client-ip"


async def client_ip(
    request: Request,
    token: Annotated[
        str | None, Header(alias="X-Client-IP-Token", include_in_schema=False)
    ] = None,
) -> str:
    """The calling visitor's IP, for per-visitor limits on public routes.

    Browser calls arrive from the frontend's server, so the frontend signs the
    visitor's IP into a short-lived service token. Without a valid one, the
    caller is keyed by the address it connected from.
    """
    if token:
        claims = await frontend_service_claims(token, CLIENT_IP_SCOPE)
        ip = claims.get("ip") if claims else None
        if isinstance(ip, str) and ip:
            return ip
    if forwarded := request.headers.get("cf-connecting-ip"):
        return forwarded
    return request.client.host if request.client else "unknown"


ClientIP = Annotated[str, Depends(client_ip)]
