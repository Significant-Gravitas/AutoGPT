"""The rate refresh keeps the rates checked by hand."""

import io
import json

from scripts import refresh_anthropic_rates as refresh


def _upstream(**overrides: float) -> dict:
    """LiteLLM's file with the checked rates, plus *overrides* on Sonnet 5.5."""
    rates = {model: dict(fields) for model, fields in refresh.CHECKED_RATES.items()}
    rates["claude-sonnet-5-5"].update(overrides)
    rates["gpt-5"] = {"input_cost_per_token": 1e-06}
    return rates


def test_vendored_rates_match_the_checked_rates():
    vendored = json.loads(refresh._OUT_PATH.read_text(encoding="utf-8"))
    assert refresh.checked_rate_changes(vendored["rates"]) == []


def test_a_changed_checked_rate_is_reported():
    changes = refresh.checked_rate_changes(_upstream(cache_read_input_token_cost=2e-07))
    assert changes == ["claude-sonnet-5-5 cache_read_input_token_cost: 1e-07 -> 2e-07"]


def test_a_missing_checked_model_is_reported():
    rates = _upstream()
    del rates["claude-opus-5-5"]
    assert len(refresh.checked_rate_changes(rates)) == 2


def _serve(monkeypatch, tmp_path, upstream: dict):
    out = tmp_path / "anthropic_rates.json"
    monkeypatch.setattr(refresh, "_OUT_PATH", out)
    monkeypatch.setattr(
        refresh.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(json.dumps(upstream).encode()),
    )
    return out


def test_refresh_stops_without_writing_when_a_checked_rate_changes(
    monkeypatch, tmp_path
):
    out = _serve(monkeypatch, tmp_path, _upstream(cache_read_input_token_cost=2e-07))
    assert refresh.main() == 1
    assert not out.exists()


def test_refresh_writes_claude_rates_when_the_checked_rates_hold(monkeypatch, tmp_path):
    out = _serve(monkeypatch, tmp_path, _upstream())
    assert refresh.main() == 0
    written = json.loads(out.read_text(encoding="utf-8"))["rates"]
    assert written["claude-sonnet-5-5"]["cache_read_input_token_cost"] == 1e-07
    assert "gpt-5" not in written
