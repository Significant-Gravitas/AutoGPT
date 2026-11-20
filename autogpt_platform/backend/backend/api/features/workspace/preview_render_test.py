import asyncio
import io
import os
import resource
import runpy
import subprocess
import sys
import textwrap
import time
import types
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from backend.api.features.workspace import preview_render

CHILD_MEMORY_MARGIN = 256 * 1024 * 1024

requires_proc_status = pytest.mark.skipif(
    not os.path.exists("/proc/self/status"),
    reason="Peak address space is read from /proc/self/status (Linux only)",
)

# Stub renderers. The child process imports them from this module by name, so
# they stay at module level and this module imports nothing heavy.


def echo_renderer(content: bytes, width: int) -> bytes:
    return content + str(width).encode()


def hanging_renderer(_content: bytes, _width: int) -> bytes:
    time.sleep(120)
    return b"never"


def allocating_renderer(content: bytes, _width: int) -> bytes:
    # The content names how many bytes to allocate.
    block = bytearray(int(content))
    return bytes(block[:1]) or b"x"


def too_large_renderer(_content: bytes, _width: int) -> bytes:
    raise preview_render.PreviewTooLarge()


def failing_renderer(_content: bytes, _width: int) -> bytes:
    raise ValueError("cannot decode")


def crashing_renderer(_content: bytes, _width: int) -> bytes:
    os.abort()


def out_of_memory_renderer(_content: bytes, _width: int) -> bytes:
    raise MemoryError()


def env_renderer(content: bytes, _width: int) -> bytes:
    # The content names a variable; answer with its value in the child.
    return os.environ.get(content.decode(), "<unset>").encode()


def peak_memory_renderer(_content: bytes, _width: int) -> bytes:
    # Answer with the child's peak address space, the figure RLIMIT_AS caps.
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmPeak:"):
                return str(int(line.split()[1]) * 1024).encode()
    raise RuntimeError("VmPeak not reported")


async def memory_cap_above_child_baseline() -> int:
    """A memory cap ``CHILD_MEMORY_MARGIN`` above an idle render child's peak.

    The child's baseline (interpreter, PIL, pypdfium2) differs between hosts,
    so a cap measured from it holds the same headroom on all of them.
    """
    baseline = await preview_render.render_in_child(peak_memory_renderer, b"", 16)
    return int(baseline) + CHILD_MEMORY_MARGIN


def interval_renderer(_content: bytes, _width: int) -> bytes:
    # Report when the render ran, so the parent can count overlapping renders.
    start = time.time()
    time.sleep(0.5)
    return f"{start},{time.time()}".encode()


@pytest.fixture
def started_children(mocker) -> list[asyncio.subprocess.Process]:
    started: list[asyncio.subprocess.Process] = []
    start = preview_render._start_child

    async def record(*args, **kwargs):
        process = await start(*args, **kwargs)
        started.append(process)
        return process

    mocker.patch.object(preview_render, "_start_child", record)
    return started


async def wait_for_children(
    started_children: list[asyncio.subprocess.Process],
    renders: list[asyncio.Task],
) -> None:
    """Wait until every render in ``renders`` has started its child."""
    async with asyncio.timeout(60):
        while len(started_children) < len(renders):
            for render in renders:
                if render.done():
                    render.result()  # raises why the render ended early
                    raise AssertionError("A render finished before it was observed")
            await asyncio.sleep(0.05)


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def test_render_returns_the_child_output():
    output = await preview_render.render_in_child(echo_renderer, b"bytes-", 64)

    assert output == b"bytes-64"


async def test_render_child_receives_only_allowlisted_variables(monkeypatch):
    monkeypatch.setenv("PREVIEW_RENDER_PARENT_SETTING", "parent-only")
    monkeypatch.setenv("LANG", "C.UTF-8")

    hidden = await preview_render.render_in_child(
        env_renderer, b"PREVIEW_RENDER_PARENT_SETTING", 16
    )
    passed = await preview_render.render_in_child(env_renderer, b"LANG", 16)

    assert hidden == b"<unset>"
    assert passed == b"C.UTF-8"


def _encoded(image_format: str) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (300, 200), (10, 20, 30)).save(out, format=image_format)
    return out.getvalue()


@pytest.mark.parametrize(
    "renderer, image_format",
    [
        (preview_render.image_thumbnail, "PNG"),
        (preview_render.image_thumbnail, "JPEG"),
        (preview_render.pdf_thumbnail, "PDF"),
    ],
)
async def test_render_child_renders_each_format(renderer, image_format):
    output = await preview_render.render_in_child(renderer, _encoded(image_format), 128)

    with Image.open(io.BytesIO(output)) as thumbnail:
        assert thumbnail.format == "WEBP"
        assert 0 < thumbnail.width <= 128


async def test_render_runs_in_a_separate_process(started_children):
    await preview_render.render_in_child(echo_renderer, b"x", 16)

    (child,) = started_children
    assert child.pid != os.getpid()


async def test_render_past_the_timeout_is_killed(mocker, started_children):
    mocker.patch.object(preview_render, "RENDER_TIMEOUT_SECONDS", 1)
    started = time.monotonic()

    with pytest.raises(preview_render.RenderFailed):
        await preview_render.render_in_child(hanging_renderer, b"x", 16)

    elapsed = time.monotonic() - started
    (child,) = started_children
    assert 1 <= elapsed < 30  # ended at the timeout, not when the render would
    assert child.returncode is not None
    assert not _is_running(child.pid)


@requires_proc_status
async def test_render_over_the_memory_cap_is_too_large(mocker):
    cap = await memory_cap_above_child_baseline()
    mocker.patch.object(preview_render, "RENDER_MAX_MEMORY_BYTES", cap)
    parent_peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    with pytest.raises(preview_render.PreviewTooLarge):
        await preview_render.render_in_child(
            allocating_renderer, str(2 * cap).encode(), 16
        )

    # ru_maxrss is in KiB on Linux; the allocation never touched this process.
    parent_peak_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert (parent_peak_after - parent_peak_before) * 1024 < cap


@requires_proc_status
async def test_render_within_the_memory_cap_succeeds(mocker):
    cap = await memory_cap_above_child_baseline()
    mocker.patch.object(preview_render, "RENDER_MAX_MEMORY_BYTES", cap)

    output = await preview_render.render_in_child(
        allocating_renderer, str(16 * 1024 * 1024).encode(), 16
    )

    assert output == b"\x00"


def test_render_under_a_lower_hard_limit_warns_once_and_does_not_run():
    # A child cannot raise its address-space limit above the hard limit it
    # inherits, so under a hard limit below the cap it must not render. Lowering
    # a hard limit cannot be undone, so the parent runs in its own process.
    script = textwrap.dedent(
        """
        import asyncio, resource
        from backend.api.features.workspace import preview_render
        from backend.api.features.workspace.preview_render_test import echo_renderer

        hard_limit = preview_render.RENDER_MAX_MEMORY_BYTES - 1
        resource.setrlimit(resource.RLIMIT_AS, (hard_limit, hard_limit))

        async def render():
            try:
                await preview_render.render_in_child(echo_renderer, b"x", 16)
            except preview_render.RenderFailed:
                return "failed"
            return "rendered"

        print(asyncio.run(render()), asyncio.run(render()))
        """
    )

    parent = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )

    assert parent.stdout.split() == ["failed", "failed"]
    assert parent.stderr.count("previews will not render") == 1
    assert "exited with code" not in parent.stderr


def test_render_child_that_cannot_set_the_cap_warns_once():
    # The parent's check passes and is cached; the limit then drops below the
    # cap, so only the child finds out it cannot apply it.
    script = textwrap.dedent(
        """
        import asyncio, resource
        from backend.api.features.workspace import preview_render
        from backend.api.features.workspace.preview_render_test import echo_renderer

        preview_render._check_memory_cap()
        hard_limit = preview_render.RENDER_MAX_MEMORY_BYTES - 1
        resource.setrlimit(resource.RLIMIT_AS, (hard_limit, hard_limit))

        async def render():
            try:
                await preview_render.render_in_child(echo_renderer, b"x", 16)
            except preview_render.RenderFailed:
                return "failed"
            return "rendered"

        print(asyncio.run(render()), asyncio.run(render()))
        """
    )

    parent = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )

    assert parent.stdout.split() == ["failed", "failed"]
    assert parent.stderr.count("could not set its memory cap") == 1
    assert "exited with code" not in parent.stderr


@pytest.mark.parametrize(
    "renderer, error",
    [
        (too_large_renderer, preview_render.PreviewTooLarge),
        (failing_renderer, preview_render.RenderFailed),
        (crashing_renderer, preview_render.RenderFailed),
    ],
)
async def test_render_outcomes_map_to_errors(renderer, error):
    with pytest.raises(error):
        await preview_render.render_in_child(renderer, b"x", 16)


async def test_render_output_above_the_limit_fails(mocker):
    mocker.patch.object(preview_render, "MAX_THUMBNAIL_BYTES", 1024)

    with pytest.raises(preview_render.RenderFailed):
        await preview_render.render_in_child(echo_renderer, b"x" * 4096, 16)


async def test_render_ends_when_its_output_is_left_unread(mocker, started_children):
    # The parent stops reading while the child is still writing, so the output
    # backs up in the pipe and the parent's read buffer before the timeout.
    async def stalled_read(_stream: asyncio.StreamReader, _limit: int) -> bytes:
        await asyncio.Event().wait()
        return b""

    mocker.patch.object(preview_render, "_read_bounded", stalled_read)
    mocker.patch.object(preview_render, "RENDER_TIMEOUT_SECONDS", 1)

    async with asyncio.timeout(60):
        with pytest.raises(preview_render.RenderFailed):
            await preview_render.render_in_child(
                echo_renderer, b"x" * (8 * 1024 * 1024), 16
            )

    (child,) = started_children
    assert child.returncode is not None
    assert not _is_running(child.pid)


async def test_cancelled_render_kills_its_child(started_children):
    task = asyncio.create_task(
        preview_render.render_in_child(hanging_renderer, b"x", 16)
    )
    await wait_for_children(started_children, [task])

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    (child,) = started_children
    assert child.returncode is not None
    assert not _is_running(child.pid)


def test_hanging_renders_leave_the_default_executor_free(mocker, started_children):
    # Renders that hang must not hold the event loop's default thread pool,
    # which the API's token checks and other ``asyncio.to_thread`` calls share.
    # Run on a loop whose default pool is no bigger than the renders in flight.
    # The renders outlast the probe and are cancelled after it.
    mocker.patch.object(preview_render, "RENDER_TIMEOUT_SECONDS", 600)
    renders = 4

    async def scenario() -> tuple[str, bool]:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=renders))
        in_flight = [
            asyncio.create_task(
                preview_render.render_in_child(hanging_renderer, b"x", 16)
            )
            for _ in range(renders)
        ]
        try:
            await wait_for_children(started_children, in_flight)
            answer = await asyncio.wait_for(asyncio.to_thread(lambda: "answered"), 60)
            return answer, any(task.done() for task in in_flight)
        finally:
            for task in in_flight:
                task.cancel()
            await asyncio.gather(*in_flight, return_exceptions=True)

    loop = asyncio.new_event_loop()
    try:
        with ThreadPoolExecutor(max_workers=1) as runner:
            answer, any_render_done = runner.submit(
                loop.run_until_complete, scenario()
            ).result(timeout=300)
    finally:
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()

    assert answer == "answered"
    assert not any_render_done
    assert all(child.returncode is not None for child in started_children)


# ---------------------------------------------------------------------------
# In-process checks of the code the child runs, and of the parent's helpers,
# called directly rather than through a child process.
# ---------------------------------------------------------------------------

THIS_MODULE = "backend.api.features.workspace.preview_render_test"


@pytest.fixture
def fresh_memory_cap_checks():
    preview_render._check_memory_cap.cache_clear()
    preview_render._warn_child_could_not_cap_memory.cache_clear()
    yield
    preview_render._check_memory_cap.cache_clear()
    preview_render._warn_child_could_not_cap_memory.cache_clear()


@pytest.fixture
def child_io(monkeypatch):
    def run(argv: list[str], content: bytes) -> tuple[int, bytes]:
        stdout = io.BytesIO()
        monkeypatch.setattr(
            sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(content))
        )
        monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(buffer=stdout))
        return preview_render._child_main(argv), stdout.getvalue()

    return run


def _zip_with(members: dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return out.getvalue()


def _cmyk_jpeg() -> bytes:
    out = io.BytesIO()
    Image.new("CMYK", (300, 200), (10, 20, 30, 40)).save(out, format="JPEG")
    return out.getvalue()


def test_child_env_keeps_only_allowlisted_variables(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("PREVIEW_RENDER_PARENT_SETTING", "parent-only")
    monkeypatch.delenv("TMPDIR", raising=False)

    env = preview_render._child_env()

    assert env["PATH"] == "/usr/bin"
    assert "PREVIEW_RENDER_PARENT_SETTING" not in env
    assert "TMPDIR" not in env
    assert set(env) <= set(preview_render.CHILD_ENV_ALLOWLIST)


async def test_render_that_cannot_start_fails(mocker):
    mocker.patch.object(
        preview_render, "_start_child", side_effect=OSError("no interpreter")
    )
    warning = mocker.patch.object(preview_render.logger, "warning")

    with pytest.raises(preview_render.RenderFailed):
        await preview_render.render_in_child(echo_renderer, b"x", 16)

    assert "could not start" in warning.call_args.args[0]


@pytest.mark.parametrize("error", [BrokenPipeError, ConnectionResetError])
async def test_feed_to_a_closed_child_is_quiet(mocker, error):
    stdin = mocker.MagicMock()
    stdin.write.side_effect = error()

    await preview_render._feed(stdin, b"content")

    stdin.close.assert_not_called()


@pytest.mark.parametrize(
    "returncode, stdout, error",
    [
        (preview_render._EXIT_OK, b"", preview_render.RenderFailed),
        (preview_render._EXIT_TOO_LARGE, b"", preview_render.PreviewTooLarge),
        (preview_render._EXIT_NO_THUMBNAIL, b"", preview_render.NoEmbeddedThumbnail),
        (1, b"partial", preview_render.RenderFailed),
        (-9, b"", preview_render.RenderFailed),
        (None, b"", preview_render.RenderFailed),
    ],
)
def test_result_maps_exit_codes(returncode, stdout, error):
    with pytest.raises(error):
        preview_render._result(returncode, stdout)


def test_result_returns_output_of_a_clean_exit():
    assert preview_render._result(preview_render._EXIT_OK, b"webp") == b"webp"


@pytest.mark.parametrize("parent_warned, child_warning_calls", [(True, 0), (False, 1)])
def test_result_of_a_child_without_a_memory_cap(
    mocker, parent_warned, child_warning_calls
):
    mocker.patch.object(preview_render, "_check_memory_cap", return_value=parent_warned)
    child_warning = mocker.patch.object(
        preview_render, "_warn_child_could_not_cap_memory"
    )
    warning = mocker.patch.object(preview_render.logger, "warning")

    with pytest.raises(preview_render.RenderFailed):
        preview_render._result(preview_render._EXIT_NO_MEMORY_CAP, b"")

    assert child_warning.call_count == child_warning_calls
    warning.assert_not_called()


def test_memory_cap_check_passes_without_a_hard_limit(mocker, fresh_memory_cap_checks):
    mocker.patch.object(
        preview_render.resource,
        "getrlimit",
        return_value=(resource.RLIM_INFINITY, resource.RLIM_INFINITY),
    )
    warning = mocker.patch.object(preview_render.logger, "warning")

    assert preview_render._check_memory_cap() is False
    warning.assert_not_called()


def test_memory_cap_check_warns_under_a_lower_hard_limit(
    mocker, fresh_memory_cap_checks
):
    hard_limit = preview_render.RENDER_MAX_MEMORY_BYTES - 1
    mocker.patch.object(
        preview_render.resource, "getrlimit", return_value=(hard_limit, hard_limit)
    )
    warning = mocker.patch.object(preview_render.logger, "warning")

    assert preview_render._check_memory_cap() is True
    assert preview_render._check_memory_cap() is True

    warning.assert_called_once()
    assert "previews will not render" in warning.call_args.args[0]


def test_memory_cap_check_passes_above_the_cap(mocker, fresh_memory_cap_checks):
    hard_limit = preview_render.RENDER_MAX_MEMORY_BYTES * 2
    mocker.patch.object(
        preview_render.resource, "getrlimit", return_value=(hard_limit, hard_limit)
    )

    assert preview_render._check_memory_cap() is False


@pytest.mark.parametrize("error", [ValueError, OSError])
def test_memory_cap_check_warns_when_the_limit_is_unreadable(
    mocker, fresh_memory_cap_checks, error
):
    mocker.patch.object(
        preview_render.resource, "getrlimit", side_effect=error("unsupported")
    )
    warning = mocker.patch.object(preview_render.logger, "warning")

    assert preview_render._check_memory_cap() is True
    assert "RLIMIT_AS is not available" in warning.call_args.args[0]


def test_memory_cap_check_on_windows_renders_without_a_cap(
    mocker, monkeypatch, fresh_memory_cap_checks
):
    monkeypatch.setattr(preview_render.sys, "platform", "win32")
    warning = mocker.patch.object(preview_render.logger, "warning")

    assert preview_render._check_memory_cap() is True
    assert "without a memory cap" in warning.call_args.args[0]


def test_child_memory_cap_warning_is_logged_once(mocker, fresh_memory_cap_checks):
    warning = mocker.patch.object(preview_render.logger, "warning")

    preview_render._warn_child_could_not_cap_memory()
    preview_render._warn_child_could_not_cap_memory()

    warning.assert_called_once()
    assert "could not set its memory cap" in warning.call_args.args[0]


def test_limit_memory_sets_the_address_space_cap(mocker):
    setrlimit = mocker.patch.object(preview_render.resource, "setrlimit")

    assert preview_render._limit_memory(1024) is True
    setrlimit.assert_called_once_with(resource.RLIMIT_AS, (1024, 1024))


@pytest.mark.parametrize("error", [ValueError, OSError])
def test_limit_memory_reports_a_cap_it_cannot_set(mocker, error):
    mocker.patch.object(
        preview_render.resource, "setrlimit", side_effect=error("not permitted")
    )

    assert preview_render._limit_memory(1024) is False


def test_limit_memory_on_windows_is_skipped(mocker, monkeypatch):
    monkeypatch.setattr(preview_render.sys, "platform", "win32")
    setrlimit = mocker.patch.object(preview_render.resource, "setrlimit")

    assert preview_render._limit_memory(1024) is True
    setrlimit.assert_not_called()


def test_child_main_writes_the_render_output(mocker, child_io):
    limit = mocker.patch.object(preview_render, "_limit_memory", return_value=True)

    code, output = child_io([f"{THIS_MODULE}:echo_renderer", "32", "4096"], b"in-")

    assert (code, output) == (preview_render._EXIT_OK, b"in-32")
    limit.assert_called_once_with(4096)


@pytest.mark.parametrize(
    "target, content, exit_code",
    [
        (f"{THIS_MODULE}:too_large_renderer", b"", preview_render._EXIT_TOO_LARGE),
        (f"{THIS_MODULE}:out_of_memory_renderer", b"", preview_render._EXIT_TOO_LARGE),
        (
            f"{preview_render._CHILD_MODULE}:office_thumbnail",
            _zip_with({"ppt/presentation.xml": b"<p/>"}),
            preview_render._EXIT_NO_THUMBNAIL,
        ),
    ],
)
def test_child_main_exit_codes(mocker, child_io, target, content, exit_code):
    mocker.patch.object(preview_render, "_limit_memory", return_value=True)

    code, output = child_io([target, "32", "4096"], content)

    assert (code, output) == (exit_code, b"")


def test_child_main_without_a_memory_cap_does_not_render(mocker, child_io):
    mocker.patch.object(preview_render, "_limit_memory", return_value=False)
    renderer = mocker.patch(f"{THIS_MODULE}.echo_renderer")

    code, output = child_io([f"{THIS_MODULE}:echo_renderer", "32", "4096"], b"x")

    assert (code, output) == (preview_render._EXIT_NO_MEMORY_CAP, b"")
    renderer.assert_not_called()


def test_child_main_rejects_a_malformed_command_line():
    with pytest.raises(ValueError):
        preview_render._child_main([f"{THIS_MODULE}:echo_renderer", "32"])


def test_module_run_as_a_script_exits_with_the_child_result(mocker, monkeypatch):
    child_main = mocker.patch.object(
        preview_render, "_child_main", return_value=preview_render._EXIT_NO_THUMBNAIL
    )
    monkeypatch.setattr(sys, "argv", ["preview_render", "target", "32", "4096"])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(preview_render.__file__, run_name="__main__")

    assert exit_info.value.code == preview_render._EXIT_NO_THUMBNAIL
    child_main.assert_called_once_with(["target", "32", "4096"])


@pytest.fixture
def pixel_limit(monkeypatch):
    # The image renderer sets Pillow's global pixel limit; restore it afterwards.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", Image.MAX_IMAGE_PIXELS)


@pytest.mark.parametrize("encoded", [_encoded("PNG"), _cmyk_jpeg()])
def test_image_thumbnail_is_a_webp_within_the_width(pixel_limit, encoded):
    output = preview_render.image_thumbnail(encoded, 64)

    with Image.open(io.BytesIO(output)) as thumbnail:
        assert thumbnail.format == "WEBP"
        assert thumbnail.mode in ("RGB", "RGBA", "L")
        assert thumbnail.width == 64


def test_image_thumbnail_declines_a_source_over_the_budget(pixel_limit, monkeypatch):
    # 300x200 is above a 50 000 pixel budget but below Pillow's own 2x limit.
    monkeypatch.setattr(preview_render, "MAX_RENDER_PIXELS", 50_000)

    with pytest.raises(preview_render.PreviewTooLarge):
        preview_render.image_thumbnail(_encoded("PNG"), 64)


def test_image_thumbnail_declines_a_decompression_bomb(pixel_limit, monkeypatch):
    # 300x200 is above Pillow's 2x limit for a 10 000 pixel budget.
    monkeypatch.setattr(preview_render, "MAX_RENDER_PIXELS", 10_000)

    with pytest.raises(preview_render.PreviewTooLarge):
        preview_render.image_thumbnail(_encoded("PNG"), 64)


def test_pdf_thumbnail_is_a_webp_within_the_width():
    output = preview_render.pdf_thumbnail(_encoded("PDF"), 64)

    with Image.open(io.BytesIO(output)) as thumbnail:
        assert thumbnail.format == "WEBP"
        assert 0 < thumbnail.width <= 64


def test_pdf_thumbnail_declines_a_page_over_the_budget(monkeypatch):
    monkeypatch.setattr(preview_render, "MAX_RENDER_PIXELS", 1)

    with pytest.raises(preview_render.PreviewTooLarge):
        preview_render.pdf_thumbnail(_encoded("PDF"), 64)


@pytest.mark.parametrize(
    "member", ["docProps/thumbnail.png", "DOCPROPS/Thumbnail.JPEG"]
)
def test_office_thumbnail_renders_the_embedded_image(pixel_limit, member):
    content = _zip_with({member: _encoded("PNG"), "ppt/presentation.xml": b"<p/>"})

    output = preview_render.office_thumbnail(content, 64)

    with Image.open(io.BytesIO(output)) as thumbnail:
        assert thumbnail.format == "WEBP"
        assert thumbnail.width == 64


def test_office_thumbnail_without_an_embedded_image_has_none():
    content = _zip_with({"ppt/presentation.xml": b"<p/>"})

    with pytest.raises(preview_render.NoEmbeddedThumbnail):
        preview_render.office_thumbnail(content, 64)


def test_render_budget_admits_a_source_at_the_limit():
    preview_render._ensure_within_render_budget((preview_render.MAX_RENDER_PIXELS, 1))

    with pytest.raises(preview_render.PreviewTooLarge):
        preview_render._ensure_within_render_budget(
            (preview_render.MAX_RENDER_PIXELS + 1, 1)
        )


@pytest.mark.parametrize("page_width, page_height", [(0, 100), (100, 0), (-1, -1)])
def test_render_scale_of_an_empty_page_is_one(page_width, page_height):
    assert preview_render._render_scale(page_width, page_height, 400) == 1.0


def test_render_scale_reaches_the_target_width_of_a_small_page():
    assert preview_render._render_scale(200, 300, 400) == 2.0
