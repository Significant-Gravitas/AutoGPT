import importlib
import sys
from pathlib import Path
from types import SimpleNamespace


BUILD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUILD))
build_runtime = importlib.import_module("build_runtime")


def test_linux_valkey_keeps_upstream_license(tmp_path, monkeypatch):
    build = build_runtime.Build(tmp_path / "out", tmp_path / "cache")
    staging = build.cache / "valkey-dist"
    (staging / "bin").mkdir(parents=True)
    (staging / "share").mkdir()
    binary = b"synthetic valkey binary"
    license_bytes = b"synthetic upstream copyright and license\r\n"
    (staging / "bin" / "valkey-server").write_bytes(binary)
    (staging / "share" / "LICENSE").write_bytes(license_bytes)
    monkeypatch.setattr(build_runtime, "WINDOWS", False)
    monkeypatch.setattr(build_runtime, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(build, "fetch", lambda name: tmp_path / name)
    monkeypatch.setattr(build_runtime, "extract", lambda *args, **kwargs: None)

    build.step_valkey()

    assert (build.out / "valkey" / "valkey-server").read_bytes() == binary
    assert (build.out / "valkey" / "LICENSE").read_bytes() == license_bytes
