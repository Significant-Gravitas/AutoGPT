import asyncio
import shutil
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

_VALIDATOR = Path(__file__).with_name("openui-validator.cjs")
_TIMEOUT_SECONDS = 5


class ValidationResult(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    valid: bool
    error: str = Field(max_length=2000)


class ValidatorUnavailable(Exception):
    pass


async def validate_openui_source(source: str) -> ValidationResult:
    node = shutil.which("node")
    if not node:
        raise ValidatorUnavailable("Node runtime unavailable")
    try:
        process = await asyncio.create_subprocess_exec(
            node,
            "--max-old-space-size=96",
            "--stack-size=512",
            str(_VALIDATOR),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env={},
            cwd=_VALIDATOR.parent,
        )
    except OSError as error:
        raise ValidatorUnavailable("Cannot start validator") from error
    try:
        output, _ = await asyncio.wait_for(
            process.communicate(source.encode()), timeout=_TIMEOUT_SECONDS
        )
    except asyncio.CancelledError:
        await _terminate(process)
        raise
    except TimeoutError as error:
        await _terminate(process)
        raise ValidatorUnavailable("Validator timed out") from error
    if process.returncode != 0 or len(output) > 8192:
        raise ValidatorUnavailable("Validator failed")
    try:
        return ValidationResult.model_validate_json(output)
    except ValidationError as error:
        raise ValidatorUnavailable("Invalid validator response") from error


async def _terminate(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    await process.communicate()
