"""Exercise the bundled Windows CLI against a fake provider, without account access."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[4]
MARKER = "AUTOGPT_POWERSHELL_PROBE"
FILE_CONTENT = "AUTOGPT_MCP_FILE_PROBE"


def source_contract(repo: Path) -> dict[str, Any]:
    sdk = repo / "autogpt_platform/backend/backend/copilot/sdk"
    scope = {"Any": Any, "json": json, "re": re, "logger": logging.getLogger(__name__)}
    names = {
        "SDK_DISALLOWED_TOOLS",
        "BLOCKED_TOOLS",
        "DANGEROUS_PATTERNS",
        "WORKSPACE_SCOPED_TOOLS",
        "_SDK_BUILTIN_FILE_TOOLS",
        "_SDK_BUILTIN_ALWAYS",
        "_SDK_BUILTIN_TOOLS",
    }
    tree = ast.parse((sdk / "tool_adapter.py").read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        )
    ]
    exec(  # noqa: S102 - evaluate only selected definitions from repository source
        compile(ast.Module(body=nodes, type_ignores=[]), "tool_adapter.py", "exec"),
        scope,
    )
    tree = ast.parse((sdk / "security_hooks.py").read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_deny", "_validate_tool_access"}
    ]
    exec(  # noqa: S102 - avoid importing the backend's service configuration
        compile(ast.Module(body=nodes, type_ignores=[]), "security_hooks.py", "exec"),
        scope,
    )
    return scope


def probe(repo: Path, scratch: Path) -> dict[str, Any]:
    import claude_agent_sdk
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        ClaudeSDKClient,
        HookMatcher,
        create_sdk_mcp_server,
        tool,
    )
    from claude_agent_sdk._cli_version import __cli_version__
    from claude_agent_sdk._version import __version__

    cli = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude.exe"
    home, workspace, temp = (scratch / name for name in ("home", "workspace", "temp"))
    for directory in (home, workspace, temp):
        directory.mkdir(parents=True)
    (workspace / "probe.txt").write_text(FILE_CONTENT, encoding="utf-8")
    system = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in {"SYSTEMROOT", "WINDIR", "COMSPEC", "SYSTEMDRIVE", "PATHEXT"}
    }
    os.environ.clear()
    os.environ.update(system)
    os.environ.update(
        {
            "PATH": str(Path(system.get("SYSTEMROOT", r"C:\Windows")) / "System32"),
            "HOME": str(home),
            "USERPROFILE": str(home),
            "APPDATA": str(home / "AppData/Roaming"),
            "LOCALAPPDATA": str(home / "AppData/Local"),
            "TEMP": str(temp),
            "TMP": str(temp),
            "CLAUDE_CONFIG_DIR": str(home / ".claude"),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "OTEL_SDK_DISABLED": "true",
            "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK": "1",
            "HTTPS_PROXY": "http://127.0.0.1:1",
            "HTTP_PROXY": "http://127.0.0.1:1",
            "NO_PROXY": "127.0.0.1,localhost",
            # Git is installed at fixed paths on developer hosts. Force the
            # PowerShell surface that this CLI enables automatically without Git.
            "CLAUDE_CODE_USE_POWERSHELL_TOOL": "1",
        }
    )
    contract = source_contract(repo)
    report: dict[str, Any] = {
        "sdk_version": __version__,
        "cli_version": __cli_version__,
        "cli_sha256": hashlib.sha256(cli.read_bytes()).hexdigest(),
        "requests": [],
        "hooks": [],
        "mcp_reads": [],
        "stderr": [],
    }

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            results = [
                part
                for message in body.get("messages", [])
                for part in (
                    message["content"]
                    if isinstance(message.get("content"), list)
                    else []
                )
                if part.get("type") == "tool_result"
            ]
            if "count_tokens" in self.path:
                self.respond("application/json", b'{"input_tokens":100}')
                return
            report["requests"].append(
                {
                    "tools": [item["name"] for item in body.get("tools", [])],
                    "results": results,
                }
            )
            completed = {result["tool_use_id"] for result in results}
            if "shell-probe" not in completed:
                block = {
                    "type": "tool_use",
                    "id": "shell-probe",
                    "name": "PowerShell",
                    "input": {},
                }
                delta = {
                    "type": "input_json_delta",
                    "partial_json": json.dumps({"command": f"Write-Output {MARKER}"}),
                }
            elif "file-probe" not in completed:
                block = {
                    "type": "tool_use",
                    "id": "file-probe",
                    "name": "mcp__copilot__read_file",
                    "input": {},
                }
                delta = {
                    "type": "input_json_delta",
                    "partial_json": '{"path":"probe.txt"}',
                }
            else:
                block = {"type": "text", "text": ""}
                delta = {"type": "text_delta", "text": "PROBE_COMPLETE"}
            message = {
                "id": "message-probe",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 100, "output_tokens": 0},
            }
            events = [
                ("message_start", {"message": message}),
                ("content_block_start", {"index": 0, "content_block": block}),
                ("content_block_delta", {"index": 0, "delta": delta}),
                ("content_block_stop", {"index": 0}),
                (
                    "message_delta",
                    {
                        "delta": {
                            "stop_reason": (
                                "tool_use"
                                if block["type"] == "tool_use"
                                else "end_turn"
                            ),
                            "stop_sequence": None,
                        },
                        "usage": {"output_tokens": 20},
                    },
                ),
                ("message_stop", {}),
            ]
            data = "".join(
                f"event: {event}\ndata: {json.dumps({'type': event, **value})}\n\n"
                for event, value in events
            ).encode()
            self.respond("text/event-stream", data)

        def respond(self, content_type: str, data: bytes):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    @tool("read_file", "Read the probe workspace file", {"path": str})
    async def read_file(arguments):
        assert arguments == {"path": "probe.txt"}
        report["mcp_reads"].append(arguments["path"])
        return {
            "content": [{"type": "text", "text": (workspace / "probe.txt").read_text()}]
        }

    async def guard(input_data, tool_use_id, context):
        name = input_data["tool_name"]
        decision = (
            {}
            if name.startswith("mcp__copilot__")
            else contract["_validate_tool_access"](
                name, input_data["tool_input"], str(workspace)
            )
        )
        report["hooks"].append({"name": name, "decision": decision})
        return decision

    async def run():
        options = ClaudeAgentOptions(
            cli_path=str(cli),
            cwd=str(workspace),
            setting_sources=[],
            extra_args={"strict-mcp-config": None},
            model="claude-sonnet-4-6",
            max_turns=4,
            allowed_tools=[*contract["_SDK_BUILTIN_TOOLS"], "mcp__copilot__read_file"],
            disallowed_tools=contract["SDK_DISALLOWED_TOOLS"],
            mcp_servers={
                "copilot": create_sdk_mcp_server(name="copilot", tools=[read_file])
            },
            hooks={"PreToolUse": [HookMatcher(matcher="*", hooks=[guard])]},
            stderr=report["stderr"].append,
        )
        async with asyncio.timeout(40):
            async with ClaudeSDKClient(options=options) as client:
                await client.query("Exercise the probe tools, then finish.")
                async for message in client.receive_response():
                    if (
                        type(message).__name__ == "SystemMessage"
                        and message.subtype == "init"
                    ):
                        report["powershell_path"] = message.data.get("powershell_path")
                    elif type(message).__name__ == "ResultMessage":
                        report["result"] = message.result
                        report["is_error"] = message.is_error

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    os.environ.update(
        {
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{server.server_port}",
            "ANTHROPIC_AUTH_TOKEN": "fake-loopback-probe",
            "ANTHROPIC_API_KEY": "",
            "CLAUDE_CODE_OAUTH_TOKEN": "",
            "CLAUDE_CODE_REFRESH_TOKEN": "",
        }
    )
    try:
        asyncio.run(run())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    return report


def test_powershell_is_denied_by_the_security_hook():
    contract = source_contract(REPO)
    decision = contract["_validate_tool_access"](
        "PowerShell", {"command": f"Write-Output {MARKER}"}
    )
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows CLI")
def test_windows_cli_blocks_powershell_and_keeps_mcp_file_tools(tmp_path: Path):
    sdk = pytest.importorskip("claude_agent_sdk")
    cli = Path(sdk.__file__).parent / "_bundled/claude.exe"
    if not cli.is_file():
        pytest.skip("the installed SDK has no bundled native Windows CLI")
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--probe",
            str(REPO),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=55,
        check=True,
    )
    report = json.loads(result.stdout)
    spec = importlib.util.spec_from_file_location(
        "desktop_probe_artifacts", REPO / "autogpt_platform/desktop/build/artifacts.py"
    )
    assert spec and spec.loader
    artifacts = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = artifacts
    spec.loader.exec_module(artifacts)
    assert report["cli_sha256"] == artifacts.ARTIFACTS["win32-x64"]["claude-cli"].sha256
    assert report["cli_version"] == artifacts.CLAUDE_CLI_VERSION
    assert report["result"] == "PROBE_COMPLETE" and report["is_error"] is False
    assert report[
        "powershell_path"
    ], "PowerShell detection must remain enabled for no-Git startup"
    assert report["mcp_reads"] == ["probe.txt"]
    assert all("PowerShell" not in request["tools"] for request in report["requests"])
    results = {
        item["tool_use_id"]: item
        for request in report["requests"]
        for item in request["results"]
    }
    assert results["shell-probe"]["is_error"] is True
    assert MARKER not in json.dumps(results["shell-probe"]["content"])
    assert FILE_CONTENT in json.dumps(results["file-probe"]["content"])


if __name__ == "__main__":
    assert sys.argv[1] == "--probe"
    print(
        json.dumps(
            probe(Path(sys.argv[2]), Path(sys.argv[3])), ensure_ascii=True, indent=2
        )
    )
