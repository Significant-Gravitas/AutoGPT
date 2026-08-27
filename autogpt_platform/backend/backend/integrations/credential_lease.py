import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable
from typing import TypeVar

from redis.asyncio.lock import Lock as AsyncRedisLock

from backend.data.model import Credentials

CheckpointCallback = Callable[[Credentials, AsyncRedisLock], Awaitable[None]]
DeleteCallback = Callable[[Credentials, AsyncRedisLock], Awaitable[None]]
T = TypeVar("T")


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


async def run_with_credential_lease_guard(
    action: Awaitable[T],
    leases: Iterable[CredentialLease],
) -> T:
    active_leases = tuple(leases)
    if not active_leases:
        return await action

    action_task = asyncio.ensure_future(action)
    failure_tasks = [
        asyncio.create_task(lease.wait_for_failure()) for lease in active_leases
    ]
    try:
        done, _ = await asyncio.wait(
            [action_task, *failure_tasks],
            return_when=asyncio.FIRST_COMPLETED,
        )
        failed = next((task for task in failure_tasks if task in done), None)
        if failed is not None:
            action_task.cancel()
            await asyncio.gather(action_task, return_exceptions=True)
            await failed
        result = await action_task
        for lease in active_leases:
            await lease.validate()
        return result
    finally:
        for task in failure_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*failure_tasks, return_exceptions=True)


async def iterate_with_credential_lease_guard(
    iterator: AsyncGenerator[T, None],
    leases: Iterable[CredentialLease],
) -> AsyncIterator[T]:
    try:
        while True:
            try:
                item = await run_with_credential_lease_guard(
                    anext(iterator),
                    leases,
                )
            except StopAsyncIteration:
                break
            yield item
    finally:
        await iterator.aclose()
