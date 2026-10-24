from typing import Annotated

from autogpt_libs.auth.service import frontend_service_claims
from fastapi import Depends, Header, Request

CLIENT_IP_SCOPE = "client-ip"
CLIENT_IP_TOKEN_HEADER = "X-Client-IP-Token"


async def client_ip(
    request: Request,
    token: Annotated[
        str | None, Header(alias=CLIENT_IP_TOKEN_HEADER, include_in_schema=False)
    ] = None,
) -> str:
    """The calling visitor's IP, for per-visitor limits on public routes.

    Browser calls arrive from the frontend's server, so the frontend signs the
    visitor's IP into a short-lived service token. Without a valid one, the
    caller is keyed by its connecting address, which uvicorn reads from
    X-Forwarded-For only when the peer is a proxy listed in FORWARDED_ALLOW_IPS.
    """
    if token:
        claims = await frontend_service_claims(token, CLIENT_IP_SCOPE)
        ip = claims.get("ip") if claims else None
        if isinstance(ip, str) and ip:
            return ip
    return request.client.host if request.client else "unknown"


ClientIP = Annotated[str, Depends(client_ip)]
