"""The desktop browser's protocol: what a call may hold, what it runs as, and
the receipts its results become. The batch shapes are agent-browser
0.37.1's, recorded from a real E2B desktop."""

import json

import pytest

from backend.copilot import desktop_browser as driver
from backend.copilot.desktop_browser import (
    OBSERVE,
    Action,
    CallRefused,
    call_seconds,
    commands_for,
    observation_from,
    parse_batch,
    parse_call,
    receipts_for,
)

_LIVE = {"token": "tok", "refs": ["e1", "e2", "e12"]}


def _ok(result: dict | None = None) -> dict:
    return {"command": [], "success": True, "error": None, "result": result or {}}


def _fail(error: str, **extra) -> dict:
    return {"command": [], "success": False, "error": error, "result": None, **extra}


class TestTheCall:
    def test_a_well_formed_call_is_parsed_and_refs_normalised(self):
        actions = parse_call(
            [
                {"action": "fill", "ref": "e2", "text": "Ada"},
                {"action": "press", "key": "Enter"},
                {"action": "wait", "text": "Thanks", "timeout_ms": 5000},
            ],
            "tok",
            _LIVE,
        )
        assert [a.name for a in actions] == ["fill", "press", "wait"]
        assert actions[0].ref == "@e2"

    @pytest.mark.parametrize(
        "actions, why",
        [
            ([], "non-empty"),
            ([{"action": "eval", "js": "1"}], "unknown action"),
            (
                [{"action": "click", "ref": "@e1", "button": "middle"}],
                "unknown parameters",
            ),
            ([{"action": "goto", "url": "file:///etc/passwd"}], "http or https"),
            ([{"action": "goto", "url": "javascript:alert(1)"}], "http or https"),
            ([{"action": "fill", "ref": "@e1"}], "text is required"),
            ([{"action": "fill", "ref": "#login", "text": "x"}], "a ref like @e3"),
            ([{"action": "press", "key": "Enter; rm -rf /"}], "key name"),
            ([{"action": "wait"}], "exactly one"),
            ([{"action": "wait", "text": "a", "time_ms": 5}], "exactly one"),
            ([{"action": "wait", "time_ms": 31_000}], "1 to 30000"),
            ([{"action": "wait", "time_ms": 5, "timeout_ms": 5}], "timeout_ms is for"),
            ([{"action": "type", "ref": "@e1", "text": "x" * 10_001}], "longer than"),
            ([{"action": "look", "full_page": "yes"}], "true or false"),
            ([{"action": "look"}, {"action": "look"}], "one look"),
            ([{"action": "snapshot"}, {"action": "state"}], "call of its own"),
            ([{"action": "state"}] * 21, "at most 20"),
            (
                [{"action": "click", "ref": "@e1", "kind": "right"}],
                "one of left, double",
            ),
        ],
    )
    def test_anything_outside_the_protocol_is_refused(self, actions, why):
        with pytest.raises(CallRefused, match=why):
            parse_call(actions, "tok", _LIVE)

    def test_a_ref_needs_the_newest_snapshots_scope(self):
        call = [{"action": "click", "ref": "@e1"}]
        with pytest.raises(CallRefused, match="no snapshot yet"):
            parse_call(call, None, None)
        with pytest.raises(CallRefused, match="newest snapshot's"):
            parse_call(call, None, _LIVE)
        with pytest.raises(CallRefused, match="newest snapshot's"):
            parse_call(call, "an-older-token", _LIVE)
        with pytest.raises(CallRefused, match="newest snapshot's"):
            parse_call(call, "tök-ünïcode", _LIVE)
        with pytest.raises(CallRefused, match="not in the newest snapshot"):
            parse_call([{"action": "click", "ref": "@e7"}], "tok", _LIVE)

    def test_no_ref_after_a_navigation_in_the_same_call(self):
        """Its page is gone; bdrive's post-navigation rule."""
        with pytest.raises(CallRefused, match="after a navigation"):
            parse_call(
                [
                    {"action": "goto", "url": "https://a.example"},
                    {"action": "click", "ref": "@e1"},
                ],
                "tok",
                _LIVE,
            )
        parse_call(
            [
                {"action": "goto", "url": "https://a.example"},
                {"action": "wait", "text": "Welcome"},
                {"action": "look"},
                {"action": "get_text"},
            ],
            None,
            None,
        )


class TestTranslation:
    def _one(self, raw: dict) -> list[list[str]]:
        (action,) = parse_call([raw], "tok", _LIVE)
        return commands_for(action, "/tmp/look.png")

    def test_actions_map_to_agent_browser_commands(self):
        assert self._one({"action": "goto", "url": "https://a.example/x"}) == [
            ["open", "https://a.example/x"]
        ]
        assert self._one({"action": "click", "ref": "@e1", "kind": "double"}) == [
            ["dblclick", "@e1"]
        ]
        assert self._one({"action": "press", "key": "Enter", "ref": "@e2"}) == [
            ["focus", "@e2"],
            ["press", "Enter"],
        ]
        assert self._one({"action": "look", "full_page": True}) == [
            ["screenshot", "--full", "/tmp/look.png"]
        ]
        assert self._one({"action": "get_text"}) == [["get", "text", "body"]]
        assert self._one({"action": "cookies_clear"}) == [["cookies", "clear"]]

    def test_open_never_sends_a_bare_open(self):
        """A bare ``open`` relaunches the attached browser and loses the
        session's tab (seen in E2B)."""
        assert self._one({"action": "open"}) == [["get", "url"], ["get", "title"]]

    def test_waiting_for_text_quotes_it_as_data(self):
        (cmd,) = self._one(
            {"action": "wait", "text": 'a") || steal("', "timeout_ms": 900}
        )
        assert cmd[:2] == ["wait", "--fn"]
        assert (
            cmd[2]
            == "document.body.innerText.includes(" + json.dumps('a") || steal("') + ")"
        )
        assert cmd[3:] == ["--timeout", "900"]
        (gone,) = self._one({"action": "wait", "text_gone": "Loading"})
        assert gone[2].startswith("!document.body.innerText.includes(")

    def test_the_call_gets_time_for_each_action_and_the_snapshot(self):
        actions = [
            Action(name="wait", params={"time_ms": 20_000}),
            Action(name="wait", params={"text": "x", "timeout_ms": 60_000}),
        ]
        assert call_seconds(actions) == (10_000 + 20_000 + 15_000 + 60_000) // 1000 + 1
        assert (
            call_seconds([Action(name="state", params={})] * 20)
            == driver.MAX_CALL_SECONDS
        )


class TestReceipts:
    def _receipts(self, raws: list[dict], entries: list[dict]):
        actions = parse_call(raws, "tok", _LIVE)
        groups = [commands_for(a, "/tmp/l.png") for a in actions]
        return receipts_for(actions, groups, entries)

    def test_all_completed_with_their_results(self):
        receipts = self._receipts(
            [{"action": "goto", "url": "https://a.example"}, {"action": "get_text"}],
            [_ok({"url": "https://a.example/"}), _ok({"text": "Hello"})],
        )
        assert [r.as_dict() for r in receipts] == [
            {
                "action": "goto",
                "status": "completed",
                "result": {"url": "https://a.example/"},
            },
            {"action": "get_text", "status": "completed", "result": {"text": "Hello"}},
        ]

    def test_a_stale_ref_did_not_start_and_stops_the_rest(self):
        receipts = self._receipts(
            [{"action": "click", "ref": "@e1"}, {"action": "state"}],
            [_fail("Unknown ref: e1")],
        )
        assert [(r.status, r.reason) for r in receipts] == [
            ("not_started", "stale_ref"),
            ("not_started", "earlier_failure"),
        ]

    def test_a_timeout_may_have_run(self):
        (receipt,) = self._receipts(
            [{"action": "wait", "text": "Done"}],
            [_fail("Wait timed out after 15000ms")],
        )
        assert (receipt.status, receipt.reason) == ("unknown", "timeout")

    def test_a_failure_after_part_of_an_action_ran_is_unknown(self):
        """press with a ref focuses first: a failed press after a focus may
        have changed the page."""
        (receipt,) = self._receipts(
            [{"action": "press", "key": "Enter", "ref": "@e2"}],
            [_ok(), _fail("Unknown ref: e2")],
        )
        assert receipt.status == "unknown"

    def test_missing_entries_are_unknown_not_completed(self):
        (receipt,) = self._receipts([{"action": "reload"}], [])
        assert receipt.status == "unknown"


class TestObservation:
    def test_the_terminal_snapshot_gives_url_title_refs(self):
        obs = observation_from(
            [
                _ok({"url": "https://a.example/"}),
                _ok({"title": "A"}),
                _ok(
                    {"snapshot": '- button "Go" [ref=e1]', "refs": {"e1": {}, "e2": {}}}
                ),
            ]
        )
        assert obs is not None
        assert (obs.url, obs.title, obs.refs) == (
            "https://a.example/",
            "A",
            ["e1", "e2"],
        )

    def test_a_huge_snapshot_is_cut_and_says_so(self):
        obs = observation_from(
            [
                _ok({"url": "u"}),
                _ok({"title": "t"}),
                _ok({"snapshot": "x" * 50_000, "refs": {}}),
            ]
        )
        assert obs is not None and obs.snapshot.endswith("the page has more]")
        assert len(obs.snapshot) < driver.MAX_SNAPSHOT_CHARS + 100

    def test_a_failed_observation_is_none(self):
        assert observation_from([_fail("tab_gone")]) is None
        assert len(OBSERVE) == 3


def test_batch_output_parses_lists_and_error_objects():
    assert parse_batch('[{"success": true}]') == [{"success": True}]
    assert parse_batch(
        '{"error": "All CDP discovery methods failed", "success": false}'
    ) == [{"error": "All CDP discovery methods failed", "success": False}]
    assert parse_batch("not json")[0]["success"] is False
    assert parse_batch("") == []


class _LockRedis:
    def __init__(self):
        self.data: dict[str, str] = {}

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.data:
            return None
        self.data[key] = value
        return True

    async def eval(self, script, numkeys, key, token):
        if self.data.get(key) == token:
            del self.data[key]


@pytest.mark.asyncio
async def test_calls_on_one_box_take_turns(monkeypatch):
    """A second call waits for the first, so it is checked against the
    snapshot the first one leaves, never one it replaced."""
    import asyncio
    from unittest.mock import AsyncMock

    redis = _LockRedis()
    monkeypatch.setattr(driver, "get_redis_async", AsyncMock(return_value=redis))
    monkeypatch.setattr(driver, "_LOCK_POLL_SECONDS", 0.01)
    order: list[str] = []

    async def call(name: str):
        async with driver.call_lock("sb-1", 60):
            order.append(f"{name} in")
            await asyncio.sleep(0.05)
            order.append(f"{name} out")

    await asyncio.gather(call("a"), call("b"))
    assert order in (
        ["a in", "a out", "b in", "b out"],
        ["b in", "b out", "a in", "a out"],
    )
    assert redis.data == {}


@pytest.mark.asyncio
async def test_a_box_held_too_long_is_reported_busy(monkeypatch):
    from unittest.mock import AsyncMock

    redis = _LockRedis()
    redis.data["copilot:desktop-browser:lock:sb-1"] = "someone"
    monkeypatch.setattr(driver, "get_redis_async", AsyncMock(return_value=redis))
    monkeypatch.setattr(driver, "_LOCK_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(driver, "_LOCK_POLL_SECONDS", 0.01)
    with pytest.raises(driver.BrowserBusy):
        async with driver.call_lock("sb-1", 60):
            pass
    assert redis.data == {"copilot:desktop-browser:lock:sb-1": "someone"}
