"""Tests for shared attribute-resolution helpers."""

from types import SimpleNamespace

from backend.copilot.graphiti._format import (
    extract_episode_body,
    extract_episode_timestamp,
    extract_fact,
    extract_temporal_validity,
)


def test_extract_fact_prefers_fact_attribute() -> None:
    edge = SimpleNamespace(fact="user likes python", name="preference")
    assert extract_fact(edge) == "user likes python"


def test_extract_fact_falls_back_to_name() -> None:
    edge = SimpleNamespace(name="preference")
    assert extract_fact(edge) == "preference"


def test_extract_fact_handles_none_fact() -> None:
    edge = SimpleNamespace(fact=None, name="fallback")
    assert extract_fact(edge) == "fallback"


def test_extract_fact_handles_missing_both() -> None:
    edge = SimpleNamespace()
    assert extract_fact(edge) == ""


def test_extract_temporal_validity_with_values() -> None:
    edge = SimpleNamespace(valid_at="2025-01-01", invalid_at="2025-12-31")
    assert extract_temporal_validity(edge) == ("2025-01-01", "2025-12-31")


def test_extract_temporal_validity_defaults() -> None:
    edge = SimpleNamespace()
    assert extract_temporal_validity(edge) == ("unknown", "present")


def test_extract_temporal_validity_none_values() -> None:
    edge = SimpleNamespace(valid_at=None, invalid_at=None)
    assert extract_temporal_validity(edge) == ("unknown", "present")


def test_extract_episode_body_prefers_content() -> None:
    ep = SimpleNamespace(content="hello world", body="alt", episode_body="alt2")
    assert extract_episode_body(ep) == "hello world"


def test_extract_episode_body_falls_back_to_body() -> None:
    ep = SimpleNamespace(body="fallback body")
    assert extract_episode_body(ep) == "fallback body"


def test_extract_episode_body_falls_back_to_episode_body() -> None:
    ep = SimpleNamespace(episode_body="last resort")
    assert extract_episode_body(ep) == "last resort"


def test_extract_episode_body_handles_none_all() -> None:
    ep = SimpleNamespace(content=None, body=None, episode_body=None)
    assert extract_episode_body(ep) == ""


def test_extract_episode_body_truncates() -> None:
    ep = SimpleNamespace(content="x" * 1000)
    assert len(extract_episode_body(ep)) == 500


def test_extract_episode_body_custom_max_len() -> None:
    ep = SimpleNamespace(content="x" * 100)
    assert len(extract_episode_body(ep, max_len=10)) == 10


def test_extract_episode_timestamp_with_value() -> None:
    ep = SimpleNamespace(created_at="2025-01-01T00:00:00Z")
    assert extract_episode_timestamp(ep) == "2025-01-01T00:00:00Z"


def test_extract_episode_timestamp_missing() -> None:
    ep = SimpleNamespace()
    assert extract_episode_timestamp(ep) == ""


def test_extract_episode_timestamp_none() -> None:
    ep = SimpleNamespace(created_at=None)
    assert extract_episode_timestamp(ep) == ""
