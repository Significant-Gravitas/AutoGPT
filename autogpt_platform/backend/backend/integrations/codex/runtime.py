import asyncio
import concurrent.futures
import os
import sys
import threading
from collections.abc import Awaitable, Callable
from contextlib import suppress
from importlib.metadata import version
from pathlib import Path
from typing import cast

from codex_cli_bin import bundled_codex_path
from openai_codex import (
    ApprovalMode,
    AsyncCodex,
    AsyncDeviceCodeLoginHandle,
    AsyncThread,
    CodexConfig,
    Sandbox,
    TurnResult,
)
from openai_codex._run import _collect_async_turn_result
from openai_codex.generated.v2_all import (
    ChatgptAccount,
    GetAccountRateLimitsResponse,
    RateLimitWindow,
    ReasoningEffort,
    ReasoningThreadItem,
    ThreadStartResponse,
)
from openai_codex.types import JsonObject
from openai_codex.models import Notification

from backend.integrations.codex.models import (
    CodexAccountSnapshot,
    CodexDeviceCodeDetails,
    CodexDynamicToolCall,
    CodexDynamicToolResult,
    CodexDynamicToolSpec,
    CodexInvocationRequest,
    CodexInvocationResult,
    CodexRateLimitsSnapshot,
    CodexRateLimitWindow,
    CodexTokenUsage,
)
from backend.integrations.codex.temporary_home import TemporaryCodexHome

OPENAI_CODEX_VERSION = "0.144.4"
CODEX_RUNTIME_VERSION = "0.144.4"
_RUNTIME_CLOSE_TIMEOUT_SECONDS = 3.0


class CodexRuntimeError(RuntimeError):
    pass


class CodexRuntimeDeviceLogin:
    def __init__(self, handle: AsyncDeviceCodeLoginHandle) -> None:
        self._handle = handle
        self.details = CodexDeviceCodeDetails(
            login_id=handle.login_id,
            verification_url=handle.verification_url,
            user_code=handle.user_code,
        )

    async def wait(self) -> bool:
        completion = await self._handle.wait()
        return completion.success

    async def cancel(self) -> None:
        await self._handle.cancel()


class CodexRuntime:
    def __init__(
        self,
        client: AsyncCodex,
        home: TemporaryCodexHome,
        *,
        close_timeout_seconds: float = _RUNTIME_CLOSE_TIMEOUT_SECONDS,
    ) -> None:
        self._client = client
        self._home = home
        self._close_timeout_seconds = close_timeout_seconds
        self._closed = False
        self._close_lock = asyncio.Lock()
        self._dynamic_tool_futures: set[concurrent.futures.Future[object]] = set()
        self._dynamic_tool_futures_lock = threading.Lock()

    @classmethod
    async def start(cls, home: TemporaryCodexHome) -> "CodexRuntime":
        assert_pinned_versions()
        config = build_runtime_config(home)
        client = AsyncCodex(config)
        _install_fail_closed_approval_handler(client)
        startup = asyncio.create_task(client.__aenter__())
        try:
            await asyncio.shield(startup)
        except asyncio.CancelledError:
            with suppress(BaseException):
                await _close_codex_client(client, _RUNTIME_CLOSE_TIMEOUT_SECONDS)
            await _cancel_task_bounded(startup, _RUNTIME_CLOSE_TIMEOUT_SECONDS)
            raise
        except BaseException:
            with suppress(BaseException):
                await _close_codex_client(client, _RUNTIME_CLOSE_TIMEOUT_SECONDS)
            raise
        return cls(client, home)

    async def start_device_code_login(self) -> CodexRuntimeDeviceLogin:
        handle = await self._client.login_chatgpt_device_code()
        return CodexRuntimeDeviceLogin(handle)

    async def account(
        self,
        *,
        refresh_token: bool = True,
    ) -> CodexAccountSnapshot:
        response = await self._client.account(refresh_token=refresh_token)
        if response.account is None:
            return CodexAccountSnapshot(
                connected=False,
                requires_openai_auth=response.requires_openai_auth,
            )
        account = response.account.root
        if account.type != "chatgpt":
            return CodexAccountSnapshot(
                connected=True,
                requires_openai_auth=response.requires_openai_auth,
                account_type=account.type,
            )
        chatgpt = cast(ChatgptAccount, account)
        return CodexAccountSnapshot(
            connected=True,
            requires_openai_auth=response.requires_openai_auth,
            account_type=chatgpt.type,
            email=chatgpt.email,
            plan_type=chatgpt.plan_type.value,
        )

    async def rate_limits(self) -> CodexRateLimitsSnapshot:
        response = await self._client._client.request(
            "account/rateLimits/read",
            None,
            response_model=GetAccountRateLimitsResponse,
        )
        limits = response.rate_limits
        credits = limits.credits
        return CodexRateLimitsSnapshot(
            plan_type=limits.plan_type.value if limits.plan_type else None,
            limit_id=limits.limit_id,
            limit_name=limits.limit_name,
            rate_limit_reached_type=(
                limits.rate_limit_reached_type.value
                if limits.rate_limit_reached_type
                else None
            ),
            primary=_rate_limit_window(limits.primary),
            secondary=_rate_limit_window(limits.secondary),
            has_credits=credits.has_credits if credits else None,
            unlimited_credits=credits.unlimited if credits else None,
            bucket_ids=sorted((response.rate_limits_by_limit_id or {}).keys()),
        )

    async def models(self) -> list[str]:
        response = await self._client.models(include_hidden=True)
        return [model.model for model in response.data]

    async def invoke(
        self,
        request: CodexInvocationRequest,
    ) -> CodexInvocationResult:
        thread = await self._start_toolless_thread(request)
        effort = ReasoningEffort(request.effort) if request.effort else None
        schema = cast(JsonObject | None, request.output_schema)
        turn_start = asyncio.create_task(
            thread.turn(
                request.prompt,
                approval_mode=ApprovalMode.deny_all,
                effort=effort,
                model=request.model,
                output_schema=schema,
                sandbox=Sandbox.read_only,
            )
        )
        try:
            turn = await asyncio.shield(turn_start)
        except asyncio.CancelledError:
            with suppress(BaseException):
                await asyncio.shield(self.close())
            await _cancel_task_bounded(
                turn_start,
                self._close_timeout_seconds,
            )
            raise
        try:
            result = await turn.run()
        except asyncio.CancelledError:
            with suppress(BaseException):
                await asyncio.wait_for(turn.interrupt(), timeout=2)
            if not self._closed:
                with suppress(BaseException):
                    await asyncio.shield(self.close())
            raise
        if result.final_response is None:
            raise CodexRuntimeError("Codex turn completed without a final response")
        return CodexInvocationResult(
            response_id=result.id,
            final_response=result.final_response,
            reasoning_summary=_reasoning_summary(result),
            status=result.status.value,
            duration_ms=result.duration_ms,
            usage=_token_usage(result),
        )

    async def invoke_agent(
        self,
        request: CodexInvocationRequest,
        dynamic_tools: list[CodexDynamicToolSpec],
        tool_handler: Callable[
            [CodexDynamicToolCall], Awaitable[CodexDynamicToolResult]
        ],
        event_handler: Callable[[Notification], Awaitable[None]] | None = None,
        *,
        tool_timeout_seconds: float,
    ) -> CodexInvocationResult:
        self._install_dynamic_tool_handler(
            tool_handler,
            timeout_seconds=tool_timeout_seconds,
        )
        thread = await self._start_thread(request, dynamic_tools)
        effort = ReasoningEffort(request.effort) if request.effort else None
        turn = await thread.turn(
            request.prompt,
            approval_mode=ApprovalMode.deny_all,
            effort=effort,
            model=request.model,
            sandbox=Sandbox.read_only,
        )
        stream = turn.stream()

        async def observed_stream():
            async for event in stream:
                if event_handler is not None:
                    await event_handler(event)
                yield event

        try:
            result = await _collect_async_turn_result(
                observed_stream(),
                turn_id=turn.id,
            )
        except asyncio.CancelledError:
            self._cancel_dynamic_tool_futures()
            with suppress(BaseException):
                await asyncio.wait_for(turn.interrupt(), timeout=2)
            raise
        finally:
            await self._close_agent_stream(stream)
        if result.final_response is None:
            raise CodexRuntimeError("Codex turn completed without a final response")
        return CodexInvocationResult(
            response_id=result.id,
            final_response=result.final_response,
            reasoning_summary=_reasoning_summary(result),
            status=result.status.value,
            duration_ms=result.duration_ms,
            usage=_token_usage(result),
        )

    async def logout(self) -> None:
        await self._client.logout()

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._cancel_dynamic_tool_futures()
            await _close_codex_client(self._client, self._close_timeout_seconds)

    async def _start_toolless_thread(
        self,
        request: CodexInvocationRequest,
    ) -> AsyncThread:
        return await self._start_thread(request, [])

    async def _start_thread(
        self,
        request: CodexInvocationRequest,
        dynamic_tools: list[CodexDynamicToolSpec],
    ) -> AsyncThread:
        params: JsonObject = {
            "approvalPolicy": "never",
            "approvalsReviewer": "user",
            "config": {
                "features": {
                    "apps": False,
                    "image_generation": False,
                    "multi_agent": False,
                    "multi_agent_v2": False,
                    "plugins": False,
                    "shell_tool": False,
                    "tool_suggest": False,
                },
                "web_search": "disabled",
            },
            "cwd": str(self._home.workspace_path),
            "dynamicTools": [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                }
                for tool in dynamic_tools
            ],
            "ephemeral": True,
            "environments": [],
            "runtimeWorkspaceRoots": [],
            "sandbox": "read-only",
            "selectedCapabilityRoots": [],
        }
        if request.instructions is not None:
            params["developerInstructions"] = request.instructions
        if request.model is not None:
            params["model"] = request.model
        started = await self._client._client.request(
            "thread/start",
            params,
            response_model=ThreadStartResponse,
        )
        return AsyncThread(self._client, started.thread.id)

    def _install_dynamic_tool_handler(
        self,
        handler: Callable[[CodexDynamicToolCall], Awaitable[CodexDynamicToolResult]],
        *,
        timeout_seconds: float,
    ) -> None:
        loop = asyncio.get_running_loop()

        def handle(method: str, params: JsonObject | None) -> JsonObject:
            if method != "item/tool/call":
                return _deny_server_request(method, params)
            try:
                call = CodexDynamicToolCall.model_validate(
                    {
                        "thread_id": (params or {}).get("threadId"),
                        "turn_id": (params or {}).get("turnId"),
                        "call_id": (params or {}).get("callId"),
                        "namespace": (params or {}).get("namespace"),
                        "tool": (params or {}).get("tool"),
                        "arguments": (params or {}).get("arguments"),
                    }
                )
            except Exception:
                return _dynamic_tool_error("codex_tool_request_invalid")

            async def execute_handler() -> CodexDynamicToolResult:
                return await handler(call)

            future = asyncio.run_coroutine_threadsafe(execute_handler(), loop)
            with self._dynamic_tool_futures_lock:
                self._dynamic_tool_futures.add(future)
            try:
                result = future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                future.cancel()
                return _dynamic_tool_error("codex_tool_execution_timeout")
            except BaseException:
                return _dynamic_tool_error("codex_tool_execution_failed")
            finally:
                with self._dynamic_tool_futures_lock:
                    self._dynamic_tool_futures.discard(future)
            return {
                "contentItems": [{"type": "inputText", "text": result.content}],
                "success": result.success,
            }

        try:
            sync_client = self._client._client._sync
        except AttributeError as exc:
            raise CodexRuntimeError(
                "Pinned Codex SDK approval-handler layout changed"
            ) from exc
        sync_client._approval_handler = handle

    def _cancel_dynamic_tool_futures(self) -> None:
        with self._dynamic_tool_futures_lock:
            futures = tuple(self._dynamic_tool_futures)
        for future in futures:
            future.cancel()

    async def _close_agent_stream(self, stream) -> None:
        closing = asyncio.create_task(stream.aclose())
        try:
            done, _ = await asyncio.wait(
                {closing},
                timeout=self._close_timeout_seconds,
            )
        except asyncio.CancelledError:
            closing.cancel()
            _detach_task(closing)
            with suppress(BaseException):
                await self.close()
            raise
        if closing in done:
            try:
                await closing
            except BaseException:
                with suppress(BaseException):
                    await self.close()
                raise
            return
        closing.cancel()
        _detach_task(closing)
        with suppress(BaseException):
            await self.close()
        raise CodexRuntimeError("codex_stream_shutdown_timeout")


async def _close_codex_client(
    client: AsyncCodex,
    timeout_seconds: float,
) -> None:
    process = _runtime_process(client)
    closing = asyncio.create_task(client.close())
    try:
        done, _ = await asyncio.wait({closing}, timeout=timeout_seconds)
        if closing in done:
            await closing
            return
        closing.cancel()
        _detach_task(closing)
        _force_stop_process(process)
    except asyncio.CancelledError:
        closing.cancel()
        _detach_task(closing)
        _force_stop_process(process)
        raise
    except BaseException:
        _force_stop_process(process)
        raise


def _runtime_process(client: AsyncCodex) -> object | None:
    try:
        return client._client._sync._proc
    except AttributeError:
        return None


def _force_stop_process(process: object | None) -> None:
    if process is None:
        return
    with suppress(BaseException):
        stdin = getattr(process, "stdin", None)
        if stdin is not None:
            stdin.close()
    with suppress(BaseException):
        process.terminate()  # type: ignore[attr-defined]
    with suppress(BaseException):
        if process.poll() is None:  # type: ignore[attr-defined]
            process.kill()  # type: ignore[attr-defined]


def _detach_task(task: asyncio.Task[object]) -> None:
    def consume_result(completed: asyncio.Task[object]) -> None:
        with suppress(BaseException):
            completed.result()

    task.add_done_callback(consume_result)


async def _cancel_task_bounded(
    task: asyncio.Task[object],
    timeout_seconds: float,
) -> None:
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=timeout_seconds)
    if task not in done:
        _detach_task(task)
        return
    with suppress(BaseException):
        task.result()


def build_runtime_config(home: TemporaryCodexHome) -> CodexConfig:
    launcher_path = str(Path(__file__).with_name("launcher.py").resolve())
    launch_args = (
        sys.executable,
        launcher_path,
        str(bundled_codex_path()),
        "--config",
        'cli_auth_credentials_store="file"',
        "--config",
        'forced_login_method="chatgpt"',
        "--config",
        "allow_login_shell=false",
        "--config",
        'web_search="disabled"',
        "--config",
        "features.shell_tool=false",
        "--config",
        "features.multi_agent=false",
        "--config",
        "features.multi_agent_v2=false",
        "--config",
        "features.apps=false",
        "--config",
        "features.plugins=false",
        "--config",
        "features.tool_suggest=false",
        "--config",
        "features.image_generation=false",
        "--config",
        "features.in_app_browser=false",
        "--config",
        "features.browser_use=false",
        "--config",
        "features.computer_use=false",
        "--config",
        "features.memories=false",
        "--config",
        "features.goals=false",
        "--config",
        "features.workspace_dependencies=false",
        "--config",
        "include_apps_instructions=false",
        "--config",
        "include_collaboration_mode_instructions=false",
        "--config",
        "include_environment_context=false",
        "--config",
        "include_permissions_instructions=false",
        "app-server",
        "--listen",
        "stdio://",
    )
    return CodexConfig(
        launch_args_override=launch_args,
        cwd=str(home.workspace_path),
        env=_runtime_environment(home),
        client_name="autogpt_platform",
        client_title="AutoGPT Platform",
        client_version="0.6.22",
        experimental_api=True,
    )


def assert_pinned_versions() -> None:
    sdk_version = version("openai-codex")
    runtime_version = version("openai-codex-cli-bin")
    if sdk_version != OPENAI_CODEX_VERSION or runtime_version != CODEX_RUNTIME_VERSION:
        raise CodexRuntimeError(
            "Codex SDK and runtime must both be pinned to version 0.144.4"
        )


def _install_fail_closed_approval_handler(client: AsyncCodex) -> None:
    try:
        sync_client = client._client._sync
    except AttributeError as exc:
        raise CodexRuntimeError(
            "Pinned Codex SDK approval-handler layout changed"
        ) from exc
    sync_client._approval_handler = _deny_server_request


def _deny_server_request(
    method: str,
    _params: JsonObject | None,
) -> JsonObject:
    if method in {
        "item/commandExecution/requestApproval",
        "item/fileChange/requestApproval",
    }:
        return {"decision": "decline"}
    if method in {"applyPatchApproval", "execCommandApproval"}:
        return {"decision": "denied"}
    if method == "item/permissions/requestApproval":
        return {"permissions": {}, "scope": "turn"}
    raise CodexRuntimeError(f"Unsupported Codex server request: {method}")


def _dynamic_tool_error(code: str) -> JsonObject:
    return {
        "contentItems": [{"type": "inputText", "text": code}],
        "success": False,
    }


def _runtime_environment(home: TemporaryCodexHome) -> dict[str, str]:
    home_path = str(home.path)
    temp_path = str(home.temp_path)
    environment = {
        "APPDATA": home_path,
        "CODEX_HOME": home_path,
        "CODEX_SQLITE_HOME": home_path,
        "HOME": home_path,
        "LOCALAPPDATA": home_path,
        "RUST_LOG": "warn",
        "TEMP": temp_path,
        "TMP": temp_path,
        "TMPDIR": temp_path,
        "USERPROFILE": home_path,
        "XDG_CACHE_HOME": str(Path(home_path) / "cache"),
        "XDG_CONFIG_HOME": str(Path(home_path) / "config"),
        "XDG_DATA_HOME": str(Path(home_path) / "data"),
    }
    return environment | _required_windows_environment()


def _required_windows_environment() -> dict[str, str]:
    names = ("COMSPEC", "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR")
    return {name: os.environ[name] for name in names if name in os.environ}


def _rate_limit_window(window: RateLimitWindow | None) -> CodexRateLimitWindow | None:
    if window is None:
        return None
    return CodexRateLimitWindow(
        used_percent=window.used_percent,
        window_duration_mins=window.window_duration_mins,
        resets_at=window.resets_at,
    )


def _token_usage(result: TurnResult) -> CodexTokenUsage | None:
    if result.usage is None:
        return None
    total = result.usage.total
    return CodexTokenUsage(
        input_tokens=total.input_tokens,
        cached_input_tokens=total.cached_input_tokens,
        output_tokens=total.output_tokens,
        reasoning_output_tokens=total.reasoning_output_tokens,
        total_tokens=total.total_tokens,
    )


def _reasoning_summary(result: TurnResult) -> str | None:
    summaries = []
    for item in result.items:
        if item.root.type != "reasoning":
            continue
        reasoning = cast(ReasoningThreadItem, item.root)
        summaries.extend(reasoning.summary or [])
    return "\n".join(summaries) or None
