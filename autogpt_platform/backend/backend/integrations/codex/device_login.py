"""A ChatGPT device-code sign-in, driven over HTTP.

Satisfies the session contract ``CodexLoginCoordinator`` already expects, so
the coordinator's Redis state, single-active-login rule and cancellation all
keep working -- only the mechanism underneath changes from a Codex subprocess
to ``auth.openai.com``.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Awaitable, TypeVar

from backend.integrations.codex.chatgpt_auth import (
    ChatGPTDeviceCode,
    CodexAuthError,
    bundle_from_tokens,
    exchange_authorization_code,
    poll_device_code,
    request_device_code,
)
from backend.integrations.codex.credential_codec import credentials_from_bundle
from backend.integrations.codex.http_client import account_snapshot
from backend.integrations.codex.models import (
    CodexDeviceCodeDetails,
    CodexLoginCompletion,
)

logger = logging.getLogger(__name__)

_MAX_INTERVAL_SECONDS = 60
_T = TypeVar("_T")


class CodexHttpDeviceLogin:
    """One pending sign-in. ``wait()`` resolves when the user approves it."""

    def __init__(
        self,
        device: ChatGPTDeviceCode,
        *,
        timeout_seconds: float,
        sleep=asyncio.sleep,
        now=lambda: datetime.now(timezone.utc),
    ) -> None:
        self._device = device
        self._timeout_seconds = timeout_seconds
        self._sleep = sleep
        self._now = now
        self._canceled = asyncio.Event()
        self.details = CodexDeviceCodeDetails(
            # Replaced by the coordinator with the id it minted.
            login_id="",
            verification_url=device.verification_url,
            user_code=device.user_code,
        )

    async def wait(self) -> CodexLoginCompletion:
        interval = self._device.interval
        deadline = self._deadline()

        while True:
            if self._canceled.is_set():
                raise CodexAuthError("ChatGPT sign-in was canceled")
            if self._now().timestamp() >= deadline:
                raise CodexAuthError("ChatGPT sign-in expired before it was approved")

            result = await self._unless_canceled(
                poll_device_code(self._device.device_auth_id, self._device.user_code)
            )

            if result.status == "approved":
                return await self._complete(result)
            if result.status == "denied":
                raise CodexAuthError("ChatGPT sign-in was declined")
            if result.status == "expired":
                raise CodexAuthError("ChatGPT sign-in expired before it was approved")
            if result.status == "slow_down":
                # Back off rather than keeping the old cadence, or the server
                # answers slow_down forever and the login never lands.
                interval = min(interval * 2, _MAX_INTERVAL_SECONDS)

            # Never sleep past the deadline: doing so reports the expiry up to a
            # full interval after it actually happened.
            remaining = deadline - self._now().timestamp()
            await self._unless_canceled(self._sleep(min(interval, max(remaining, 0))))

    async def _unless_canceled(self, operation: Awaitable[_T]) -> _T:
        """Run one poll/sleep while letting cancel interrupt it immediately."""
        operation_task = asyncio.ensure_future(operation)
        canceled_task = asyncio.create_task(self._canceled.wait())
        try:
            done, _ = await asyncio.wait(
                (operation_task, canceled_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if canceled_task in done:
                operation_task.cancel()
                raise CodexAuthError("ChatGPT sign-in was canceled")
            return await operation_task
        finally:
            if not operation_task.done():
                operation_task.cancel()
            canceled_task.cancel()
            await asyncio.gather(
                operation_task,
                canceled_task,
                return_exceptions=True,
            )

    async def _complete(self, result) -> CodexLoginCompletion:
        assert result.authorization_code and result.code_verifier
        tokens = await exchange_authorization_code(
            result.authorization_code, result.code_verifier
        )
        bundle = bundle_from_tokens(tokens)
        credentials = credentials_from_bundle(bundle)
        return CodexLoginCompletion(
            bundle=bundle,
            account=account_snapshot(credentials),
            # Quota is reported on inference responses, not at sign-in.
            rate_limits=None,
        )

    def _deadline(self) -> float:
        remaining = self._device.seconds_remaining(now=self._now())
        budget = self._timeout_seconds
        if remaining is not None and remaining > 0:
            # Never poll past the code's own expiry, however long we were
            # configured to wait.
            budget = min(budget, remaining)
        return self._now().timestamp() + budget

    async def cancel(self) -> None:
        self._canceled.set()

    async def close(self) -> None:
        self._canceled.set()


async def start_http_device_login(timeout_seconds: float) -> CodexHttpDeviceLogin:
    device = await request_device_code()
    logger.info("Started a ChatGPT device sign-in")
    return CodexHttpDeviceLogin(device, timeout_seconds=timeout_seconds)
