"""Tokenizer build inputs, without dependencies or network downloads."""

import hashlib
import importlib
import io
import sys
from pathlib import Path

import pytest

BUILD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUILD))
build_runtime = importlib.import_module("build_runtime")
tokenizer_assets = importlib.import_module("tokenizer_assets")
fetch = importlib.import_module("fetch")
Artifact = importlib.import_module("artifacts").Artifact


@pytest.fixture
def assets(monkeypatch):
    contents = {
        asset.url: f"synthetic {i}".encode()
        for i, asset in enumerate(tokenizer_assets.ASSETS)
    }
    pinned = tuple(
        Artifact(url, hashlib.sha256(data).hexdigest())
        for url, data in contents.items()
    )
    monkeypatch.setattr(tokenizer_assets, "ASSETS", pinned)
    requests = []

    def download(request, timeout):
        assert timeout == 120
        requests.append(request.full_url)
        return io.BytesIO(contents[request.full_url])

    monkeypatch.setattr(fetch.urllib.request, "urlopen", download)
    return contents, requests


def lock_at(path, version=tokenizer_assets.VERSION):
    path.write_text(
        f'[[package]]\nname = "tiktoken"\nversion = "{version}"\n', encoding="utf-8"
    )
    return path


def test_tokenizer_assets_are_built_before_pruning_and_verification():
    assert "tokenizers" in build_runtime.STEPS
    assert build_runtime.STEPS.index("tokenizers") < build_runtime.STEPS.index("prune")
    assert build_runtime.STEPS.index("tokenizers") < build_runtime.STEPS.index("verify")


def test_current_lock_and_manifest_cover_the_locked_release():
    tokenizer_assets.check_lock(build_runtime.LOCK)
    assert len(tokenizer_assets.ASSETS) == 6
    assert len({asset.url for asset in tokenizer_assets.ASSETS}) == 6
    assert set(tokenizer_assets.ENCODINGS) == {
        "gpt2",
        "r50k_base",
        "p50k_base",
        "p50k_edit",
        "cl100k_base",
        "o200k_base",
        "o200k_harmony",
    }


def test_build_step_installs_into_its_output(monkeypatch, tmp_path):
    monkeypatch.setattr(build_runtime, "platform_key", lambda: "win32-x64")
    build = build_runtime.Build(tmp_path / "runtime", tmp_path / "downloads")
    calls = []
    monkeypatch.setattr(tokenizer_assets, "install", lambda *args: calls.append(args))
    build.step_tokenizers()
    assert calls == [(build.out, build.cache, build_runtime.LOCK)]


def test_verified_downloads_use_tiktoken_names_and_repair_output_from_cache(
    tmp_path, assets
):
    contents, requests = assets
    runtime, cache = tmp_path / "runtime", tmp_path / "downloads"
    lock = lock_at(tmp_path / "poetry.lock")
    tokenizer_assets.install(runtime, cache, lock)
    output = runtime / "assets" / "tiktoken"
    expected = {
        hashlib.sha1(url.encode()).hexdigest(): data for url, data in contents.items()
    }
    assert {path.name: path.read_bytes() for path in output.iterdir()} == expected
    assert requests == list(contents)
    (output / next(iter(expected))).write_bytes(b"damaged installed copy")
    tokenizer_assets.install(runtime, cache, lock)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == expected
    assert requests == list(contents)


@pytest.mark.parametrize("warm", [False, True])
def test_sha256_mismatch_never_reaches_the_bundle(tmp_path, assets, warm):
    contents, requests = assets
    artifact = tokenizer_assets.ASSETS[0]
    cached = tmp_path / "downloads" / "tiktoken" / tokenizer_assets.cache_key(artifact)
    if warm:
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"corrupt cached asset")
    else:
        contents[artifact.url] = b"corrupt download"
    with pytest.raises(RuntimeError, match="sha256.*does not match"):
        tokenizer_assets.install(
            tmp_path / "runtime",
            tmp_path / "downloads",
            lock_at(tmp_path / "poetry.lock"),
        )
    assert not cached.exists()
    assert not (tmp_path / "runtime" / "assets" / "tiktoken" / cached.name).exists()
    assert len(requests) == (0 if warm else 1)


@pytest.mark.parametrize("version", ["0.15.0", "0.13.0", ""])
def test_lock_drift_stops_before_any_download(tmp_path, assets, version):
    _, requests = assets
    with pytest.raises(RuntimeError, match="review the encoding constructors"):
        tokenizer_assets.install(
            tmp_path / "runtime",
            tmp_path / "downloads",
            lock_at(tmp_path / "poetry.lock", version),
        )
    assert not requests
    assert not (tmp_path / "runtime").exists()
