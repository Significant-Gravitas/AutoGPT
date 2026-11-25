"""HELLO offers no hardware capability until the shim can drive hardware.

With `enable_hardware`, HELLO advertised `hardware_serial`, `hardware_usb`
and `hardware_gpio`, one for each library installed. There is no HARDWARE_*
message and no handler: a platform that was granted one and sent a hardware
request got no reply and waited out its timeout.
"""

from __future__ import annotations

import asyncio
import importlib.machinery
import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

import autogpt_local_executor.cli as cli
import autogpt_local_executor.daemon as daemon_module
from autogpt_local_executor import platform_info
from autogpt_local_executor.audit import AuditWriter
from autogpt_local_executor.config import ShimConfig, load_config
from autogpt_local_executor.daemon import ShimDaemon

ROOT = Path(__file__).resolve().parent.parent
HARDWARE_LIBRARIES = ("serial", "usb", "RPi.GPIO")


@pytest.fixture
def every_hardware_library_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """pyserial, pyusb and RPi.GPIO, as far as a look for them can tell."""
    for name in HARDWARE_LIBRARIES:
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
        monkeypatch.setitem(sys.modules, name, module)
    for name in HARDWARE_LIBRARIES:
        assert importlib.util.find_spec(name) is not None


def _config(tmp_path: Path, **settings: object) -> ShimConfig:
    return ShimConfig(
        **{
            "allowed_root": tmp_path / "workspace",
            "audit_log_path": tmp_path / "audit.log",
            "platform_url": "http://localhost:9999",
            "machine_id": "test-machine",
            **settings,
        }
    )


def _hardware(capabilities: list[str]) -> list[str]:
    return [capability for capability in capabilities if capability.startswith("hardware")]


@pytest.mark.parametrize("session_id", [None, "chat-1"], ids=["machine", "chat"])
async def test_hello_offers_no_hardware_with_enable_hardware_on(
    tmp_path: Path, every_hardware_library_installed: None, session_id: str | None
) -> None:
    config = _config(tmp_path, enable_hardware=True, session_id=session_id)
    daemon = ShimDaemon(
        config,
        token_store=object(),
        audit=AuditWriter(path=config.audit_log_path, audit_key=b"a" * 32),
    )

    hello = await daemon._build_hello()

    assert _hardware(hello.payload.capabilities) == []
    assert hello.payload.hardware_devices == []
    assert "files" in hello.payload.capabilities


def test_no_hardware_capability_is_detected(every_hardware_library_installed: None) -> None:
    assert _hardware(platform_info.detect_capabilities(enable_hardware=True)) == []
    assert platform_info.detect_capabilities(enable_hardware=True) == ["files"]


def test_enable_hardware_still_loads_and_says_it_does_nothing_yet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config that turns it on isn't an error: it has no effect, and the
    setting's own description says so."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("enable_hardware = true\n", encoding="utf-8")
    assert load_config(config_path=config_file).enable_hardware is True

    monkeypatch.setenv("AUTOGPT_SHIM_ENABLE_HARDWARE", "true")
    assert load_config().enable_hardware is True

    description = ShimConfig.model_fields["enable_hardware"].description or ""
    assert "no effect" in description


async def test_start_says_that_enable_hardware_does_nothing_yet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Runs:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def run(self) -> None:
            return None

    monkeypatch.setattr(daemon_module, "ShimDaemon", Runs)

    assert await cli._cmd_start(_config(tmp_path, enable_hardware=True)) == 0
    assert "enable_hardware has no effect" in capsys.readouterr().out

    assert await cli._cmd_start(_config(tmp_path)) == 0
    assert "enable_hardware" not in capsys.readouterr().out


def _hello_example() -> str:
    """PROTOCOL.md's HELLO example, as written."""
    protocol = (ROOT / "docs" / "PROTOCOL.md").read_text(encoding="utf-8")
    start = protocol.index("#### `HELLO` (shim → platform, on connect)")
    return protocol[start : protocol.index("```\n", protocol.index("```json", start))]


async def test_the_documented_hello_names_only_what_a_hello_can_carry(tmp_path: Path) -> None:
    """The example listed `local_llm` and the three hardware capabilities,
    with a model list and two devices, none of which a HELLO carries."""
    example = _hello_example()
    listed = re.search(r'"capabilities": \[(.*?)\]', example, re.S)
    assert listed is not None
    documented = re.findall(r'^\s*"([a-z_]+)"', listed.group(1), re.M)

    # Everything a HELLO can carry today: each switch on, on each connection.
    offered: set[str] = set()
    for session_id in (None, "chat-1"):
        config = _config(
            tmp_path,
            session_id=session_id,
            enable_shell=True,
            enable_hardware=True,
            audit_log_path=tmp_path / f"{session_id}.log",
        )
        daemon = ShimDaemon(
            config,
            token_store=object(),
            audit=AuditWriter(path=config.audit_log_path, audit_key=b"a" * 32),
        )
        offered.update((await asyncio.wait_for(daemon._build_hello(), 60)).payload.capabilities)
    # Computer use needs the driver, which this test doesn't start.
    offered.add("computer_use")

    assert documented
    assert set(documented) <= offered, set(documented) - offered
    assert '"local_llm_models": []' in example
    assert '"hardware_devices": []' in example
