from typing import Annotated

from autogpt_libs.auth.service import frontend_service_claims
from fastapi import Depends, Header

CLIENT_COUNTRY_SCOPE = "client-country"


async def attested_country(
    token: Annotated[
        str | None, Header(alias="X-Client-Country-Token", include_in_schema=False)
    ] = None,
) -> str | None:
    """The visitor's country, as the frontend proxy vouches for it, or None.

    The backend is reachable directly -- the browser already calls it with
    its own bearer token -- so a plain country header would be whatever the
    caller typed. The proxy instead sends what Vercel's edge geolocated inside
    a short-lived frontend service token, signed with the JWKS key only the
    frontend holds. Anything else -- no token, a forged or expired one, a
    user token -- is no country at all, which the trial offer's country rule
    treats as unknown and withholds. Hidden from the schema: it is
    proxy-to-backend plumbing, not API surface.
    """
    if not token:
        return None
    claims = await frontend_service_claims(token, CLIENT_COUNTRY_SCOPE)
    country = claims.get("country") if claims else None
    return country if isinstance(country, str) else None


ClientCountry = Annotated[str | None, Depends(attested_country)]
