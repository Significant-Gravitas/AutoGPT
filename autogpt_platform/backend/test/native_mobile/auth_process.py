import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path


@asynccontextmanager
async def auth_harness(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    readiness_timeout: float = 40,
) -> AsyncIterator[str]:
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert process.stdout is not None
    ready: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    tail = bytearray()
    draining = asyncio.create_task(_drain_output(process.stdout, ready, tail))
    try:
        try:
            origin = await asyncio.wait_for(ready, readiness_timeout)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                "The production auth harness timed out before readiness:\n"
                + tail.decode(errors="replace")
            ) from exc
        yield origin
    finally:
        if not ready.done():
            ready.cancel()
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        await draining


async def _drain_output(
    stdout: asyncio.StreamReader, ready: asyncio.Future[str], tail: bytearray
) -> None:
    while chunk := await stdout.read(65536):
        output = bytes(tail) + chunk
        if not ready.done():
            for line in output.splitlines(keepends=True):
                if line.startswith(b"NATIVE_AUTH_READY ") and line.endswith(b"\n"):
                    parts = line.split()
                    if len(parts) == 2:
                        ready.set_result(parts[1].decode())
                        break
        tail[:] = output[-8192:]
    if not ready.done():
        ready.set_exception(
            RuntimeError(
                "The production auth harness exited before readiness:\n"
                + tail.decode(errors="replace")
            )
        )
