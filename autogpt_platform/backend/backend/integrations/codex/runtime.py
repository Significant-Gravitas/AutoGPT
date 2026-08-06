import asyncio
import concurrent.futures
import os
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from importlib.metadata import version
from pathlib import Path
from types import MethodType
from typing import Any, cast

from codex_cli_bin import bundled_codex_path
from openai_codex import (
    ApprovalMode,
    AsyncCodex,
    AsyncDeviceCodeLoginHandle,
    AsyncThread,
    AsyncTurnHandle,
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
from openai_codex.models import Notification
from openai_codex.types import JsonObject

from backend.integrations.codex.models import (
    CodexAccountSnapshot,
    CodexDeviceCodeDetails,
    CodexDynamicToolCall,
    CodexDynamicToolResult,
    CodexDynamicToolSpec,
    CodexInvocationRequest,
    CodexInvocationResult,
    CodexModelInfo,
    CodexRateLimitsSnapshot,
    CodexRateLimitWindow,
    CodexReasoningEffort,
    CodexTokenUsage,
)
from backend.integrations.codex.temporary_home import TemporaryCodexHome

OPENAI_CODEX_VERSION = "0.144.4"
CODEX_RUNTIME_VERSION = "0.144.4"
_RUNTIME_CLOSE_TIMEOUT_SECONDS = 3.0
_SERVER_REQUEST_MAX_CONCURRENCY = 8


class _ConcurrentServerRequestDispatcher:
    def __init__(self, sync_client: Any, *, max_concurrency: int) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self._sync_client = sync_client
        self._capacity = threading.BoundedSemaphore(max_concurrency)
        self._closing = threading.Event()
        self._workers: set[threading.Thread] = set()
        self._workers_lock = threading.Lock()

    def dispatch(self, message: dict[str, object]) -> None:
        if self._closing.is_set():
            self._write_error(message, "Codex tool bridge is shutting down")
            return
        if not self._capacity.acquire(blocking=False):
            self._write_error(message, "Codex tool bridge is busy")
            return

        worker = threading.Thread(
            target=self._answer,
            args=(message,),
            name="autogpt-codex-server-request",
            daemon=True,
        )
        rejection: str | None = None
        with self._workers_lock:
            if self._closing.is_set():
                self._capacity.release()
                rejection = "Codex tool bridge is shutting down"
            else:
                self._workers.add(worker)
                try:
                    worker.start()
                except BaseException:
                    self._workers.discard(worker)
                    self._capacity.release()
                    rejection = "Codex tool bridge request failed"
        if rejection is not None:
            self._write_error(message, rejection)

    def close(self, timeout_seconds: float) -> bool:
        self._closing.set()
        deadline = time.monotonic() + max(timeout_seconds, 0)
        current = threading.current_thread()
        while True:
            with self._workers_lock:
                workers = tuple(worker for worker in self._workers if worker != current)
            if not workers:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            workers[0].join(timeout=remaining)

    def _answer(self, message: dict[str, object]) -> None:
        try:
            response = self._sync_client._handle_server_request(message)
            self._sync_client._write_message({"id": message["id"], "result": response})
        except BaseException:
            self._write_error(message, "Codex tool bridge request failed")
        finally:
            with self._workers_lock:
                self._workers.discard(threading.current_thread())
            self._capacity.release()

    def _write_error(self, message: dict[str, object], detail: str) -> None:
        if getattr(self._sync_client, "_proc", None) is None:
            return
        with suppress(BaseException):
            self._sync_client._write_message(
                {
                    "id": message.get("id"),
                    "error": {
                        "code": -32001,
                        "message": detail,
                    },
                }
            )


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
        self._poisoned = False
        self._close_lock = asyncio.Lock()
        self._dynamic_tool_handlers: dict[
            str,
            tuple[
                Callable[[CodexDynamicToolCall], Awaitable[CodexDynamicToolResult]],
                float,
            ],
        ] = {}
        self._dynamic_tool_futures: dict[
            str, set[concurrent.futures.Future[CodexDynamicToolResult]]
        ] = {}
        self._dynamic_tool_futures_lock = threading.Lock()

    @property
    def closed(self) -> bool:
        return self._closed or self._poisoned

    @classmethod
    async def start(cls, home: TemporaryCodexHome) -> "CodexRuntime":
        assert_pinned_versions()
        config = build_runtime_config(home)
        client = AsyncCodex(config)
        _install_concurrent_server_request_dispatcher(client)
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

    async def models(self) -> list[CodexModelInfo]:
        response = await self._client.models(include_hidden=True)
        return [
            CodexModelInfo(
                model=model.model,
                display_name=model.display_name,
                is_default=model.is_default,
                hidden=model.hidden,
                default_reasoning_effort=cast(
                    CodexReasoningEffort,
                    model.default_reasoning_effort.value,
                ),
                supported_reasoning_efforts=[
                    cast(CodexReasoningEffort, option.reasoning_effort.value)
                    for option in model.supported_reasoning_efforts
                ],
                input_modalities=[
                    modality.value for modality in (model.input_modalities or [])
                ],
            )
            for model in response.data
        ]

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
        thread = await self._start_thread(request, dynamic_tools)
        self._register_dynamic_tool_handler(
            thread.id,
            tool_handler,
            timeout_seconds=tool_timeout_seconds,
        )
        try:
            effort = ReasoningEffort(request.effort) if request.effort else None
            turn_start = asyncio.create_task(
                thread.turn(
                    request.prompt,
                    approval_mode=ApprovalMode.deny_all,
                    effort=effort,
                    model=request.model,
                    sandbox=Sandbox.read_only,
                )
            )
            try:
                turn = await asyncio.shield(turn_start)
            except asyncio.CancelledError:
                await self._stop_cancelled_turn_start(turn_start)
                raise
            stream = turn.stream()
            abnormal_exit = False

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
            except BaseException:
                abnormal_exit = True
                self._cancel_dynamic_tool_futures(thread.id)
                await self._interrupt_turn_or_close(turn)
                raise
            finally:
                try:
                    await self._close_agent_stream(stream)
                except BaseException:
                    if not abnormal_exit:
                        raise
                    await self._poison_runtime()
        finally:
            self._unregister_dynamic_tool_handler(thread.id)
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
                    "inputSchema": cast(JsonObject, tool.input_schema),
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

    def _register_dynamic_tool_handler(
        self,
        thread_id: str,
        handler: Callable[[CodexDynamicToolCall], Awaitable[CodexDynamicToolResult]],
        *,
        timeout_seconds: float,
    ) -> None:
        self._ensure_dynamic_tool_dispatcher()
        with self._dynamic_tool_futures_lock:
            self._dynamic_tool_handlers[thread_id] = (handler, timeout_seconds)
            self._dynamic_tool_futures.setdefault(thread_id, set())

    def _unregister_dynamic_tool_handler(self, thread_id: str) -> None:
        with self._dynamic_tool_futures_lock:
            self._dynamic_tool_handlers.pop(thread_id, None)
            futures = tuple(self._dynamic_tool_futures.pop(thread_id, ()))
        for future in futures:
            future.cancel()

    def _ensure_dynamic_tool_dispatcher(self) -> None:
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

            with self._dynamic_tool_futures_lock:
                registration = self._dynamic_tool_handlers.get(call.thread_id)
                futures = self._dynamic_tool_futures.get(call.thread_id)
                if self.closed or registration is None or futures is None:
                    return _dynamic_tool_error("codex_tool_handler_unavailable")
                handler, timeout_seconds = registration

                async def execute_handler() -> CodexDynamicToolResult:
                    return await handler(call)

                coroutine = execute_handler()
                try:
                    future = asyncio.run_coroutine_threadsafe(coroutine, loop)
                except BaseException:
                    coroutine.close()
                    return _dynamic_tool_error("codex_tool_execution_failed")
                futures.add(future)
            try:
                result = future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                future.cancel()
                return _dynamic_tool_error("codex_tool_execution_timeout")
            except BaseException:
                return _dynamic_tool_error("codex_tool_execution_failed")
            finally:
                with self._dynamic_tool_futures_lock:
                    futures = self._dynamic_tool_futures.get(call.thread_id)
                    if futures is not None:
                        futures.discard(future)
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
        if getattr(sync_client, "_autogpt_dynamic_tool_dispatcher", False):
            return
        sync_client._approval_handler = handle
        setattr(sync_client, "_autogpt_dynamic_tool_dispatcher", True)

    def _cancel_dynamic_tool_futures(self, thread_id: str | None = None) -> None:
        with self._dynamic_tool_futures_lock:
            if thread_id is None:
                futures = tuple(
                    future
                    for group in self._dynamic_tool_futures.values()
                    for future in group
                )
            else:
                futures = tuple(self._dynamic_tool_futures.get(thread_id, ()))
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

    async def _stop_cancelled_turn_start(
        self,
        turn_start: asyncio.Task[AsyncTurnHandle],
    ) -> None:
        try:
            done, _ = await asyncio.wait(
                {turn_start},
                timeout=self._close_timeout_seconds,
            )
        except BaseException:
            await self._poison_runtime()
            return
        if turn_start not in done:
            await self._poison_runtime()
            await _cancel_task_bounded(
                cast(asyncio.Task[object], turn_start),
                self._close_timeout_seconds,
            )
            return
        try:
            turn = turn_start.result()
        except BaseException:
            await self._poison_runtime()
            return
        await self._interrupt_turn_or_close(turn)

    async def _interrupt_turn_or_close(self, turn: AsyncTurnHandle) -> bool:
        interrupting = asyncio.create_task(turn.interrupt())
        try:
            done, _ = await asyncio.wait(
                {interrupting},
                timeout=min(2.0, self._close_timeout_seconds),
            )
        except BaseException:
            interrupting.cancel()
            _detach_task(cast(asyncio.Task[object], interrupting))
            await self._poison_runtime()
            return False
        if interrupting in done:
            try:
                interrupting.result()
            except BaseException:
                await self._poison_runtime()
                return False
            return True
        await _cancel_task_bounded(
            cast(asyncio.Task[object], interrupting),
            self._close_timeout_seconds,
        )
        await self._poison_runtime()
        return False

    async def _poison_runtime(self) -> None:
        self._poisoned = True
        closing = asyncio.create_task(self.close())
        try:
            done, _ = await asyncio.wait(
                {closing},
                timeout=self._close_timeout_seconds,
            )
        except BaseException:
            _detach_task(cast(asyncio.Task[object], closing))
            return
        if closing not in done:
            _detach_task(cast(asyncio.Task[object], closing))
            return
        with suppress(BaseException):
            closing.result()


def _install_concurrent_server_request_dispatcher(client: AsyncCodex) -> None:
    try:
        sync_client = client._client._sync
        sync_client._read_message
        sync_client._write_message
        sync_client._handle_server_request
        sync_client._coerce_notification
        sync_client._router
    except AttributeError as exc:
        raise CodexRuntimeError("Pinned Codex SDK reader layout changed") from exc

    if getattr(sync_client, "_autogpt_server_request_dispatcher", None) is not None:
        return
    dispatcher = _ConcurrentServerRequestDispatcher(
        sync_client,
        max_concurrency=_SERVER_REQUEST_MAX_CONCURRENCY,
    )
    setattr(sync_client, "_autogpt_server_request_dispatcher", dispatcher)

    def reader_loop(instance: Any) -> None:
        try:
            while True:
                message = instance._read_message()
                if "method" in message and "id" in message:
                    dispatcher.dispatch(message)
                    continue
                if "method" in message and "id" not in message:
                    method = message["method"]
                    if isinstance(method, str):
                        instance._router.route_notification(
                            instance._coerce_notification(
                                method,
                                message.get("params"),
                            )
                        )
                    continue
                instance._router.route_response(message)
        except BaseException as exc:
            instance._router.fail_all(exc)

    sync_client._reader_loop = MethodType(reader_loop, sync_client)


def _shutdown_concurrent_server_request_dispatcher(
    client: AsyncCodex,
    timeout_seconds: float,
) -> bool:
    try:
        dispatcher = getattr(
            client._client._sync,
            "_autogpt_server_request_dispatcher",
        )
    except AttributeError:
        return True
    return dispatcher.close(timeout_seconds)


async def _close_codex_client(
    client: AsyncCodex,
    timeout_seconds: float,
) -> None:
    process = _runtime_process(client)
    deadline = time.monotonic() + max(timeout_seconds, 0)
    closing: asyncio.Task[None] | None = None
    try:
        await asyncio.to_thread(
            _shutdown_concurrent_server_request_dispatcher,
            client,
            timeout_seconds,
        )
        closing = asyncio.create_task(client.close())
        remaining = max(deadline - time.monotonic(), 0)
        if remaining == 0:
            await asyncio.sleep(0)
            done = {closing} if closing.done() else set()
        else:
            done, _ = await asyncio.wait({closing}, timeout=remaining)
        if closing in done:
            await closing
            return
        closing.cancel()
        _detach_task(closing)
        _force_stop_process(process)
    except asyncio.CancelledError:
        if closing is not None:
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
