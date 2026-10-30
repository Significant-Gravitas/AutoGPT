"""Offline tokenizer gate regressions with synthetic encoders and data."""

import functools
import hashlib
import importlib
import os
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

BUILD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUILD))
bundle_gate = importlib.import_module("bundle_gate")
tokenizer_assets = importlib.import_module("tokenizer_assets")
Artifact = importlib.import_module("artifacts").Artifact


@pytest.fixture
def tokenizer(tmp_path, monkeypatch):
    cache = tmp_path / "assets" / "tiktoken"
    cache.mkdir(parents=True)
    pinned = []
    for index, original in enumerate(tokenizer_assets.ASSETS):
        data = f"synthetic tokenizer {index}".encode()
        artifact = Artifact(original.url, hashlib.sha256(data).hexdigest())
        pinned.append(artifact)
        (cache / tokenizer_assets.cache_key(artifact)).write_bytes(data)
    monkeypatch.setattr(tokenizer_assets, "ASSETS", tuple(pinned))
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(cache))
    monkeypatch.setenv("DATA_GYM_CACHE_DIR", str(tmp_path / "ambient-warm-cache"))
    state = SimpleNamespace(
        cache=cache,
        names=list(tokenizer_assets.ENCODINGS),
        created=[],
        downloads=[],
        broken=None,
        fallback=False,
        discoveries=0,
    )
    module = ModuleType("tiktoken")
    loader = ModuleType("tiktoken.load")
    registry = ModuleType("tiktoken.registry")
    module.__version__ = tokenizer_assets.VERSION
    module.load, module.registry = loader, registry
    registry._lock = threading.RLock()

    def read_file(url):
        state.downloads.append(url)
        return b"network should never be used"

    loader.read_file = read_file

    @functools.lru_cache
    def available():
        state.discoveries += 1
        return tuple(state.names)

    registry._available_plugin_modules = available

    class Encoding:
        def __init__(self, name):
            self.name = name

        def encode(self, text):
            return list(text.encode())

        def decode(self, tokens):
            if self.name == state.broken:
                return "broken"
            return bytes(tokens).decode()

    def names():
        if registry.ENCODING_CONSTRUCTORS is None:
            registry.ENCODING_CONSTRUCTORS = dict.fromkeys(available())
        return list(registry.ENCODING_CONSTRUCTORS)

    def get_encoding(name):
        if name not in registry.ENCODINGS:
            if state.fallback:
                loader.read_file(pinned[0].url)
            state.created.append(name)
            registry.ENCODINGS[name] = Encoding(name)
        return registry.ENCODINGS[name]

    module.list_encoding_names, module.get_encoding = names, get_encoding
    state.warm_encodings = {name: Encoding(name) for name in state.names}
    state.warm_constructors = dict.fromkeys(available())
    registry.ENCODINGS = state.warm_encodings
    registry.ENCODING_CONSTRUCTORS = state.warm_constructors
    state.module, state.read_file = module, read_file
    for name, value in (
        ("tiktoken", module),
        ("tiktoken.load", loader),
        ("tiktoken.registry", registry),
    ):
        monkeypatch.setitem(sys.modules, name, value)
    return state


def assert_restored(tokenizer):
    assert tokenizer.module.registry.ENCODINGS is tokenizer.warm_encodings
    assert (
        tokenizer.module.registry.ENCODING_CONSTRUCTORS is tokenizer.warm_constructors
    )
    assert tokenizer.module.load.read_file is tokenizer.read_file
    assert os.environ["TIKTOKEN_CACHE_DIR"] == str(tokenizer.cache)
    assert not tokenizer.downloads


def test_gate_constructs_all_encoders_again_without_changing_cache(tmp_path, tokenizer):
    before = {path.name: path.read_bytes() for path in tokenizer.cache.iterdir()}
    assert bundle_gate.offline_tokenizers(tmp_path) == []
    assert tokenizer.created == list(tokenizer_assets.ENCODINGS)
    assert tokenizer.discoveries == 2
    assert {
        path.name: path.read_bytes() for path in tokenizer.cache.iterdir()
    } == before
    assert_restored(tokenizer)


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
@pytest.mark.parametrize("index", range(6))
def test_every_missing_or_corrupt_asset_fails_despite_warm_encoders(
    tmp_path, tokenizer, damage, index
):
    artifact = tokenizer_assets.ASSETS[index]
    path = tokenizer.cache / tokenizer_assets.cache_key(artifact)
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"damaged")
    problems = bundle_gate.offline_tokenizers(tmp_path)
    assert len(problems) == 1
    assert artifact.filename in problems[0]
    assert not tokenizer.created
    assert_restored(tokenizer)


@pytest.mark.parametrize("selection", [None, "", "ambient"])
def test_gate_requires_runtime_to_select_the_managed_path(
    tmp_path, tokenizer, monkeypatch, selection
):
    if selection is None:
        monkeypatch.delenv("TIKTOKEN_CACHE_DIR")
    else:
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", selection)
    assert (
        "runtime settings did not select" in bundle_gate.offline_tokenizers(tmp_path)[0]
    )
    assert not tokenizer.created


def test_download_fallback_is_blocked_even_after_registry_warmup(tmp_path, tokenizer):
    tokenizer.fallback = True
    assert "attempted a download" in bundle_gate.offline_tokenizers(tmp_path)[0]
    assert_restored(tokenizer)


@pytest.mark.parametrize("drift", ["version", "added", "removed"])
def test_dependency_drift_fails_even_with_warm_constructor_discovery(
    tmp_path, tokenizer, drift
):
    if drift == "version":
        tokenizer.module.__version__ = "0.15.0"
    elif drift == "added":
        tokenizer.names.append("unsupported")
    else:
        tokenizer.names.pop()
    problems = bundle_gate.offline_tokenizers(tmp_path)
    assert len(problems) == 1
    assert (
        "version differs" if drift == "version" else "constructor drift"
    ) in problems[0]
    assert_restored(tokenizer)


def test_encode_decode_failure_stops_the_gate(tmp_path, tokenizer):
    tokenizer.broken = "o200k_harmony"
    assert (
        "encode/decode failed: o200k_harmony"
        in bundle_gate.offline_tokenizers(tmp_path)[0]
    )
    assert_restored(tokenizer)


@pytest.mark.parametrize("valid", [False, True])
def test_backend_gate_requires_cache_before_existing_checks(
    tmp_path, monkeypatch, tokenizer, valid
):
    if not valid:
        next(tokenizer.cache.iterdir()).unlink()
    monkeypatch.setattr(bundle_gate, "prevent_package_installs", lambda: None)
    monkeypatch.setattr(bundle_gate, "as_a_service", lambda *args: None)
    monkeypatch.setattr(
        bundle_gate.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path / "scratch")
    )
    called = []
    monkeypatch.setattr(
        bundle_gate,
        "unimportable_backend_modules",
        lambda: called.append("backend") or [],
    )
    monkeypatch.setattr(
        bundle_gate, "blocks_that_do_not_load", lambda: called.append("blocks") or []
    )
    monkeypatch.setattr(
        bundle_gate, "data_that_is_not_reached", lambda: called.append("data") or []
    )
    monkeypatch.setattr(bundle_gate, "zipped", lambda runtime: [])
    monkeypatch.setattr(
        bundle_gate,
        "code_named_for_another_file",
        lambda packages: called.append("code") or [],
    )

    def finish(code):
        raise SystemExit(code)

    monkeypatch.setattr(bundle_gate.os, "_exit", finish)
    with pytest.raises(SystemExit) as result:
        bundle_gate.backend_checks(tmp_path)
    assert result.value.code == (0 if valid else 1)
    assert called == (["backend", "blocks", "data", "code"] if valid else [])
