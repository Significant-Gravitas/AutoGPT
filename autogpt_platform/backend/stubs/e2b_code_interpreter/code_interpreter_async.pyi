"""Partial type stub for e2b_code_interpreter.AsyncSandbox."""

from typing import Any, Awaitable, Callable, Dict, Optional, Union

from e2b import AsyncSandbox as BaseAsyncSandbox

class AsyncSandbox(BaseAsyncSandbox):
    async def run_code(
        self,
        code: str,
        language: Optional[str] = None,
        context: Optional[Any] = None,
        on_stdout: Optional[Any] = None,
        on_stderr: Optional[Any] = None,
        on_result: Optional[Any] = None,
        on_error: Optional[
            Union[Callable[..., Any], Callable[..., Awaitable[Any]]]
        ] = None,
        envs: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ) -> Any: ...
