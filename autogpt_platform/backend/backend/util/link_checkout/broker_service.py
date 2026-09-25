"""The broker service: one tenant's checkout broker behind mutual TLS.

Hosted AutoGPT runs one of these per provisioned user, on a network the agent
cannot reach, with egress only through the exact-host proxy (``egress``). The
controller authenticates twice: a client certificate at TLS, then a bearer
secret. Every request names its principal, and a principal other than this
broker's tenant is refused before anything runs.
"""

import hashlib
import hmac
import logging
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.util.link_checkout.broker_checkout import (
    complete_checkout,
    create_checkout,
    get_checkout,
    reconcile,
    reset_browser,
)
from backend.util.link_checkout.broker_commands import execute_browser
from backend.util.link_checkout.broker_protocol import (
    AuthorizedCheckout,
    BrowserCommand,
    BrowserOutput,
    CheckoutReference,
    CheckoutView,
    CreateCheckout,
    Principal,
)
from backend.util.link_checkout.refusals import CheckoutRefused


def create_app(tenant_id: str, secret: bytes) -> FastAPI:
    if not tenant_id or len(secret) < 32:
        raise ValueError(
            "A single tenant and strong controller credential are required"
        )
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

    async def authorize(authorization: Annotated[str, Header()] = "") -> None:
        supplied = hashlib.sha256(authorization.encode()).digest()
        expected = hashlib.sha256(b"Bearer " + secret).digest()
        if not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="Broker authorization required")

    def require_tenant(principal: Principal) -> None:
        if principal.user_id != tenant_id:
            raise HTTPException(status_code=403, detail="Checkout unavailable")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422, content={"detail": "Invalid broker request"}
        )

    @app.exception_handler(CheckoutRefused)
    async def refused(request: Request, exc: CheckoutRefused):
        # Fixed text (``refusals``), so the controller can pass it on.
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.middleware("http")
    async def restrict_body(request: Request, call_next):
        size = request.headers.get("content-length", "")
        if request.method != "POST" or not size.isdigit() or int(size) > 32768:
            return JSONResponse(
                status_code=413, content={"detail": "Unsupported request"}
            )
        try:
            response = await call_next(request)
        except Exception:
            logging.getLogger("link_checkout.broker").warning(
                "Checkout operation failed"
            )
            response = JSONResponse(
                status_code=409,
                content={"detail": "Checkout unavailable; reconcile before retrying"},
            )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.post(
        "/v1/browser", dependencies=[Security(authorize)], response_model=BrowserOutput
    )
    async def browser(request: BrowserCommand):
        require_tenant(request)
        return await execute_browser(request)

    @app.post(
        "/v1/checkout/create",
        dependencies=[Security(authorize)],
        response_model=CheckoutView,
    )
    async def create(request: CreateCheckout):
        require_tenant(request)
        return await create_checkout(request)

    @app.post(
        "/v1/checkout/get",
        dependencies=[Security(authorize)],
        response_model=CheckoutView,
    )
    async def get(request: CheckoutReference):
        require_tenant(request)
        return await get_checkout(request)

    @app.post(
        "/v1/checkout/complete",
        dependencies=[Security(authorize)],
        response_model=CheckoutView,
    )
    async def complete(request: AuthorizedCheckout):
        require_tenant(request)
        return await complete_checkout(request)

    @app.post(
        "/v1/checkout/status",
        dependencies=[Security(authorize)],
        response_model=CheckoutView,
    )
    async def status(request: AuthorizedCheckout):
        require_tenant(request)
        return await reconcile(request)

    @app.post(
        "/v1/checkout/reset",
        dependencies=[Security(authorize)],
        response_model=CheckoutView,
    )
    async def reset(request: CheckoutReference):
        require_tenant(request)
        return await reset_browser(request)

    return app


def configured_app() -> FastAPI:
    return create_app(
        os.environ["CHECKOUT_BROKER_TENANT_ID"],
        Path(os.environ["CHECKOUT_BROKER_SECRET_FILE"]).read_bytes().strip(),
    )
