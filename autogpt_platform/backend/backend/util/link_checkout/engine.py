"""The controller's entry point: where a chat's browser and checkout run.

With remote brokers configured (``broker_routing``), each provisioned user's
browser and checkout live in that user's broker, reached over mutual TLS;
users without one browse as usual and have no checkout. Otherwise, with
``COPILOT_LINK_PRIVATE_CHECKOUT=true``, the same broker code runs in-process on
this host for everyone. Either way the agent's browser commands and the
checkout go through one state machine, and every call carries the
authenticated caller set by ``caller`` around a tool call, never a principal a
tool argument could name.
"""

import base64
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from backend.util.link_checkout import broker_checkout, broker_client
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
from backend.util.link_checkout.broker_routing import configured as remote_broker
from backend.util.link_checkout.broker_routing import routed
from backend.util.link_checkout.config import private_checkout_requested
from backend.util.link_checkout.runtime import local_runtime_ready

_caller: ContextVar[Principal | None] = ContextVar("checkout_caller", default=None)


def enabled() -> bool:
    """Whether some user's browser tools can run in a private browser.

    Requested but on a host that fails the runtime checks (swap, core dumps,
    ``/dev/shm``), browsing stays ordinary and the checkout tools stay hidden;
    every step that could touch a card re-checks and fails closed regardless.
    """
    return remote_broker() or (private_checkout_requested() and local_runtime_ready())


def active_for(user_id: str | None) -> bool:
    """Whether this user's browser tools run in their private browser."""
    if not user_id:
        return False
    if remote_broker():
        return routed(user_id)
    return private_checkout_requested() and local_runtime_ready()


def remote() -> bool:
    return remote_broker()


def requested() -> bool:
    """Whether the operator asked for the private checkout at all, runtime
    ready or not. While it is, no path may hand the agent a card number."""
    return remote_broker() or private_checkout_requested()


@contextmanager
def caller(user_id: str | None, session_id: str, owner_id: str) -> Iterator[None]:
    if not user_id or user_id != owner_id:
        raise ValueError("Browser session does not belong to this user")
    token = _caller.set(Principal(user_id=user_id, session_id=session_id))
    try:
        yield
    finally:
        _caller.reset(token)


def current_caller(session_id: str) -> Principal:
    principal = _caller.get()
    if principal is None or principal.session_id != session_id:
        raise ValueError("Authenticated browser context required")
    return principal


def serves(session_id: str) -> bool:
    """Whether this chat's browser commands go to its private browser: set by
    ``caller`` for a user it is active for, around that user's tool call."""
    principal = _caller.get()
    return principal is not None and principal.session_id == session_id


async def run_browser_command(
    session_id: str, args: tuple[str, ...]
) -> tuple[int, str, str]:
    """``agent-browser`` semantics for the browser tools: a screenshot's last
    argument is the local file to write."""
    arguments = list(args)
    screenshot_path = Path(arguments.pop()) if arguments[0] == "screenshot" else None
    principal = current_caller(session_id)
    result = await _browser(BrowserCommand(**principal.model_dump(), args=arguments))
    if screenshot_path and result.code == 0:
        screenshot_path.write_bytes(base64.b64decode(result.image, validate=True))
    return result.code, result.output, result.error


async def _browser(command: BrowserCommand) -> BrowserOutput:
    if remote_broker():
        return BrowserOutput.model_validate(
            await broker_client.request("browser", command)
        )
    return await execute_browser(command)


async def create(request: CreateCheckout) -> CheckoutView:
    if remote_broker():
        return await _remote("checkout/create", request)
    return await broker_checkout.create_checkout(request)


async def get(request: CheckoutReference) -> CheckoutView:
    if remote_broker():
        return await _remote("checkout/get", request)
    return await broker_checkout.get_checkout(request)


async def complete(request: AuthorizedCheckout) -> CheckoutView:
    if remote_broker():
        return await _remote("checkout/complete", request)
    return await broker_checkout.complete_checkout(request)


async def status(request: AuthorizedCheckout) -> CheckoutView:
    if remote_broker():
        return await _remote("checkout/status", request)
    return await broker_checkout.reconcile(request)


async def reset(request: CheckoutReference) -> CheckoutView:
    if remote_broker():
        return await _remote("checkout/reset", request)
    return await broker_checkout.reset_browser(request)


async def _remote(operation: str, request: Principal) -> CheckoutView:
    return CheckoutView.model_validate(await broker_client.request(operation, request))
