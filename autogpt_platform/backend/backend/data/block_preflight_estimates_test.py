import json
from pathlib import Path

import pytest

from backend.data import block_preflight_estimates as bpe


@pytest.fixture(autouse=True)
def reset_cache():
    bpe.reset_cache()
    yield
    bpe.reset_cache()


def test_missing_file_returns_zero(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(bpe, "_ESTIMATES_PATH", tmp_path / "missing.json")
    assert bpe.get_preflight_estimate("any-id") == 0


def test_malformed_json_falls_back_to_zero(monkeypatch, tmp_path: Path):
    bad = tmp_path / "estimates.json"
    bad.write_text("{ not valid json")
    monkeypatch.setattr(bpe, "_ESTIMATES_PATH", bad)
    assert bpe.get_preflight_estimate("any-id") == 0


def test_returns_estimate_for_known_block(monkeypatch, tmp_path: Path):
    f = tmp_path / "estimates.json"
    f.write_text(
        json.dumps(
            {
                "version": 1,
                "estimates": {
                    "block-1": {
                        "block_name": "FooBlock",
                        "cost_type": "SECOND",
                        "samples": 50,
                        "mean_credits": 7,
                    }
                },
            }
        )
    )
    monkeypatch.setattr(bpe, "_ESTIMATES_PATH", f)
    assert bpe.get_preflight_estimate("block-1") == 7
    assert bpe.get_preflight_estimate("block-2") == 0


def test_cache_refreshes_on_file_mtime_change(monkeypatch, tmp_path: Path):
    """Hot-swapping the JSON on a running pod must be picked up on the next
    call, without a process restart — keyed on file mtime."""
    import os
    import time

    f = tmp_path / "estimates.json"
    f.write_text(
        json.dumps(
            {
                "version": 1,
                "estimates": {
                    "block-1": {
                        "block_name": "FooBlock",
                        "cost_type": "SECOND",
                        "samples": 50,
                        "mean_credits": 7,
                    }
                },
            }
        )
    )
    monkeypatch.setattr(bpe, "_ESTIMATES_PATH", f)
    assert bpe.get_preflight_estimate("block-1") == 7

    # Rewrite with a newer mtime — cache must refresh on the next call.
    new_payload = {
        "version": 1,
        "estimates": {
            "block-1": {
                "block_name": "FooBlock",
                "cost_type": "SECOND",
                "samples": 50,
                "mean_credits": 999,
            }
        },
    }
    f.write_text(json.dumps(new_payload))
    # Bump mtime to guarantee the stat-based cache key invalidates even on
    # filesystems with second-resolution mtime.
    later = time.time() + 1
    os.utime(f, (later, later))

    assert bpe.get_preflight_estimate("block-1") == 999


def test_negative_or_non_numeric_mean_clamps_to_zero(monkeypatch, tmp_path: Path):
    """Corrupt entries must never produce negative billing pre-flight."""
    f = tmp_path / "estimates.json"
    f.write_text(
        json.dumps(
            {
                "version": 1,
                "estimates": {
                    "negative": {
                        "block_name": "Bad",
                        "cost_type": "SECOND",
                        "samples": 10,
                        "mean_credits": -42,
                    },
                    "non_numeric": {
                        "block_name": "Bad",
                        "cost_type": "SECOND",
                        "samples": 10,
                        "mean_credits": "oops",
                    },
                    "fractional": {
                        "block_name": "Ok",
                        "cost_type": "SECOND",
                        "samples": 10,
                        "mean_credits": 2.6,
                    },
                },
            }
        )
    )
    monkeypatch.setattr(bpe, "_ESTIMATES_PATH", f)
    assert bpe.get_preflight_estimate("negative") == 0
    assert bpe.get_preflight_estimate("non_numeric") == 0
    # Round, don't truncate — 2.6 → 3.
    assert bpe.get_preflight_estimate("fractional") == 3


def test_non_object_root_disables_estimates(monkeypatch, tmp_path: Path):
    f = tmp_path / "estimates.json"
    f.write_text("[1, 2, 3]")
    monkeypatch.setattr(bpe, "_ESTIMATES_PATH", f)
    assert bpe.get_preflight_estimate("anything") == 0


def test_corrupt_json_does_not_poison_cache_when_fixed_in_place(
    monkeypatch, tmp_path: Path
):
    """Atomic-rename / same-mtime fixes must recover. Bumping `_cache_mtime_ns`
    on a failed parse would leave an empty cache pinned until the file's
    mtime advances — on low-resolution filesystems or atomic-rename flows
    that can pin to the corrupt timestamp indefinitely.
    """
    import os

    f = tmp_path / "estimates.json"
    f.write_text("{ not valid json")
    monkeypatch.setattr(bpe, "_ESTIMATES_PATH", f)
    assert bpe.get_preflight_estimate("block-1") == 0
    poisoned_mtime = f.stat().st_mtime

    # Fix the file but pin the mtime to the corrupt-version timestamp —
    # simulates a low-resolution FS or atomic-rename flow that doesn't
    # bump the timestamp.
    f.write_text(
        json.dumps(
            {
                "version": 1,
                "estimates": {
                    "block-1": {
                        "block_name": "FooBlock",
                        "cost_type": "SECOND",
                        "samples": 50,
                        "mean_credits": 11,
                    }
                },
            }
        )
    )
    os.utime(f, (poisoned_mtime, poisoned_mtime))

    assert bpe.get_preflight_estimate("block-1") == 11


def test_non_finite_mean_clamps_to_zero(monkeypatch, tmp_path: Path):
    """Python's json.loads accepts non-spec NaN/Infinity; a hot-swapped or
    corrupted JSON containing those would crash `int(round(...))`. Verify
    they clamp to 0 instead — the docstring promises no JSON content can
    crash a billing call site.
    """
    f = tmp_path / "estimates.json"
    # Write the file directly with NaN/Infinity literals — Python's json.loads
    # accepts these even though spec JSON forbids them.
    f.write_text(
        '{"version": 1, "estimates": {'
        '"nan_block": {"block_name": "Bad", "cost_type": "SECOND",'
        ' "samples": 10, "mean_credits": NaN},'
        '"inf_block": {"block_name": "Bad", "cost_type": "SECOND",'
        ' "samples": 10, "mean_credits": Infinity},'
        '"neg_inf_block": {"block_name": "Bad", "cost_type": "SECOND",'
        ' "samples": 10, "mean_credits": -Infinity}'
        "}}"
    )
    monkeypatch.setattr(bpe, "_ESTIMATES_PATH", f)
    assert bpe.get_preflight_estimate("nan_block") == 0
    assert bpe.get_preflight_estimate("inf_block") == 0
    assert bpe.get_preflight_estimate("neg_inf_block") == 0
