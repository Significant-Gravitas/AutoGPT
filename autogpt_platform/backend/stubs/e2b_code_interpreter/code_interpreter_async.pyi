"""Partial type stub for e2b_code_interpreter.AsyncSandbox."""

from typing import Any, Awaitable, Callable, Dict, Optional, Union

from e2b import AsyncSandbox as BaseAsyncSandbox
from typing_extensions import Self

class AsyncSandbox(BaseAsyncSandbox):
    @classmethod
    async def create(  # type: ignore[override]
        cls,
        template: Optional[str] = None,
        timeout: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
        envs: Optional[Dict[str, str]] = None,
        api_key: Optional[str] = None,
        domain: Optional[str] = None,
        debug: Optional[bool] = None,
        request_timeout: Optional[float] = None,
        lifecycle: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Self: ...
    @classmethod
    async def connect(  # type: ignore[override]
        cls,
        sandbox_id: str,
        api_key: Optional[str] = None,
        domain: Optional[str] = None,
        debug: Optional[bool] = None,
        request_timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Self: ...
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
