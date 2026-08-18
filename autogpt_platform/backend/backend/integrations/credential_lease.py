import asyncio
from collections.abc import Awaitable, Callable

from redis.asyncio.lock import Lock as AsyncRedisLock

from backend.data.model import Credentials

CheckpointCallback = Callable[[Credentials, AsyncRedisLock], Awaitable[None]]
DeleteCallback = Callable[[Credentials, AsyncRedisLock], Awaitable[None]]


class CredentialLease:
    def __init__(
        self,
        credentials: Credentials,
        lock: AsyncRedisLock,
        checkpoint_callback: CheckpointCallback,
        delete_callback: DeleteCallback | None = None,
    ) -> None:
        self.credentials = credentials
        self._lock = lock
        self._checkpoint_callback = checkpoint_callback
        self._delete_callback = delete_callback
        self._deleted = False
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._heartbeat_error: BaseException | None = None
        self._heartbeat_failed = asyncio.Event()

    def start_heartbeat(self) -> None:
        if self.credentials.type != "oauth2":
            return
        if self.credentials.refresh_strategy != "provider_runtime":
            return
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat())

    async def validate(self) -> None:
        if self._heartbeat_error is not None:
            raise RuntimeError(
                "Credential lease heartbeat failed"
            ) from self._heartbeat_error
        if not (await self._lock.locked()) or not (await self._lock.owned()):
            raise RuntimeError("Credential lease ownership was lost")

    async def wait_for_failure(self) -> None:
        if self._heartbeat_error is None:
            await self._heartbeat_failed.wait()
        raise RuntimeError(
            "Credential lease heartbeat failed"
        ) from self._heartbeat_error

    @property
    def failure(self) -> BaseException | None:
        return self._heartbeat_error

    async def checkpoint(self, updated: Credentials) -> None:
        if self._deleted:
            raise RuntimeError("Credential lease was deleted")
        await self.validate()
        if (
            updated.id != self.credentials.id
            or updated.provider != self.credentials.provider
        ):
            raise RuntimeError("Credential lease identity changed")
        await self._checkpoint_callback(updated, self._lock)
        self.credentials = updated

    async def delete(self) -> None:
        if self._delete_callback is None:
            raise RuntimeError("Credential lease does not support deletion")
        if self._deleted:
            return
        await self.validate()
        await self._delete_callback(self.credentials, self._lock)
        self._deleted = True

    async def release(self) -> None:
        heartbeat = self._heartbeat_task
        if heartbeat is not None:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        if (await self._lock.locked()) and (await self._lock.owned()):
            await self._lock.release()

    async def _heartbeat(self) -> None:
        timeout = float(self._lock.timeout or 60)
        interval = max(timeout / 3, 0.01)
        try:
            while True:
                await asyncio.sleep(interval)
                extended = await self._lock.extend(timeout, replace_ttl=True)
                if not extended:
                    raise RuntimeError("Credential lease extension was rejected")
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self._heartbeat_error = error
            self._heartbeat_failed.set()
