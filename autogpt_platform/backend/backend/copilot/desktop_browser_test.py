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
        # Each wait's own time plus 2 s, and 5 s for the batch itself.
        assert call_seconds(actions) == (20_000 + 2_000 + 60_000 + 2_000) // 1000 + 5
        # Plain interactions are cheap: twenty of them fit.
        assert call_seconds([Action(name="state", params={})] * 20) == 105


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

    @pytest.mark.parametrize(
        "snapshot_result",
        [["not", "a", "dict"], {"snapshot": "x", "refs": ["e1"]}],
    )
    def test_malformed_page_data_reads_as_an_unread_page(self, snapshot_result):
        """Then the receipts still go back and the older scopes are voided."""
        entries = [_ok({"url": "u"}), _ok({"title": "t"}), _ok()]
        entries[2]["result"] = snapshot_result
        assert observation_from(entries) is None


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

    renewed = 0

    async def eval(self, script, numkeys, key, token, *args):
        if self.data.get(key) != token:
            return 0
        if "expire" in script:
            self.renewed += 1
        else:
            del self.data[key]
        return 1


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


@pytest.mark.asyncio
async def test_a_long_call_keeps_its_lock_and_a_dead_one_frees_it_soon(monkeypatch):
    """The lock is renewed while the call runs, and its own lifetime is
    short, so a worker that died holding it frees the box within a minute."""
    import asyncio
    from unittest.mock import AsyncMock

    redis = _LockRedis()
    monkeypatch.setattr(driver, "get_redis_async", AsyncMock(return_value=redis))
    monkeypatch.setattr(driver, "_LOCK_RENEW_SECONDS", 0.01)
    async with driver.call_lock("sb-1", driver.LOCK_HOLD_SECONDS):
        await asyncio.sleep(0.06)
        assert "copilot:desktop-browser:lock:sb-1" in redis.data
    assert redis.renewed >= 2
    assert redis.data == {}
    assert driver.LOCK_HOLD_SECONDS <= 60


# ---------------------------------------------------------------- review fixes


class TestBudget:
    def test_a_call_that_cannot_finish_in_time_is_refused(self):
        with pytest.raises(CallRefused, match="split it"):
            parse_call([{"action": "wait", "time_ms": 30_000}] * 7, None, None)

    def test_a_navigation_is_budgeted_past_agent_browsers_load_wait(self):
        goto = Action(name="goto", params={"url": "https://example.com"})
        assert call_seconds([goto]) == driver.NAVIGATION_MS // 1000 + 5
        assert driver.NAVIGATION_MS >= 30_000

    def test_a_long_type_is_budgeted_per_chunk(self):
        typed = Action(name="type", params={"ref": "@e1", "text": "x" * 1_500})
        assert (
            call_seconds([typed]) == (3 * driver.DEFAULT_TIMEOUT_MS + 1_000) // 1000 + 5
        )


class TestTypeAndDialog:
    def test_type_goes_in_chunks_none_of_them_a_flag(self):
        assert driver.type_chunks("a" * 1_200) == ["a" * 500, "a" * 500, "a" * 200]
        tail = "a" * 500 + "--clear"
        assert driver.type_chunks(tail) == [tail]
        typed = commands_for(
            Action(name="type", params={"ref": "@e1", "text": "b" * 501}), "/x"
        )
        assert typed == [["type", "@e1", "b" * 500], ["type", "@e1", "b"]]

    @pytest.mark.parametrize("text", ["--clear", "--delay"])
    def test_type_refuses_text_agent_browser_reads_as_a_flag(self, text):
        with pytest.raises(CallRefused, match="use fill"):
            parse_call([{"action": "type", "ref": "e1", "text": text}], "tok", _LIVE)

    def test_type_is_capped_and_fill_is_not(self):
        long = "x" * (driver.MAX_TYPE_TEXT + 1)
        with pytest.raises(CallRefused):
            parse_call([{"action": "type", "ref": "e1", "text": long}], "tok", _LIVE)
        parse_call([{"action": "fill", "ref": "e1", "text": long}], "tok", _LIVE)

    def test_dialog_answers(self):
        accept = parse_call(
            [{"action": "dialog", "answer": "accept", "text": "yes"}], None, None
        )
        assert commands_for(accept[0], "/x") == [["dialog", "accept", "yes"]]
        dismiss = parse_call([{"action": "dialog", "answer": "dismiss"}], None, None)
        assert commands_for(dismiss[0], "/x") == [["dialog", "dismiss"]]
        with pytest.raises(CallRefused, match="only for accept"):
            parse_call(
                [{"action": "dialog", "answer": "dismiss", "text": "x"}], None, None
            )


class TestThePageCheck:
    """Each ref action checks the page is the document its snapshot stamped."""

    _CLICK = Action(name="click", params={"ref": "@e1"})

    def test_only_ref_actions_are_checked_and_only_once_a_stamp_exists(self):
        commands = commands_for(self._CLICK, "/x")
        checked = driver.guarded(self._CLICK, commands, "stamp")
        assert driver.is_guard(checked[0]) and checked[1:] == commands
        assert '"stamp"' in checked[0][2]
        assert driver.guarded(self._CLICK, commands, None) == commands
        goto = Action(name="goto", params={"url": "https://example.com"})
        assert driver.guarded(goto, [["open", "https://example.com"]], "stamp") == [
            ["open", "https://example.com"]
        ]

    def test_a_changed_page_stops_the_call_before_the_ref_action(self):
        """A click that navigated left the next ref on a new document."""
        second = Action(name="click", params={"ref": "@e2"})
        groups = [
            driver.guarded(self._CLICK, commands_for(self._CLICK, "/x"), "stamp"),
            driver.guarded(second, commands_for(second, "/x"), "stamp"),
        ]
        entries = [_ok(), _ok(), _fail("Timeout 1000ms exceeded waiting for function")]
        first, then = receipts_for([self._CLICK, second], groups, entries)
        assert first.status == "completed"
        assert (then.status, then.reason) == ("not_started", "page_changed")

    def test_a_miss_after_a_passed_check_did_not_start(self):
        groups = [driver.guarded(self._CLICK, commands_for(self._CLICK, "/x"), "stamp")]
        (receipt,) = receipts_for(
            [self._CLICK], groups, [_ok(), _fail("Unknown ref: @e1")]
        )
        assert (receipt.status, receipt.reason) == ("not_started", "stale_ref")

    def test_the_closing_read_settles_after_a_click_and_stamps_last(self):
        commands = driver.observe_commands([self._CLICK], "stamp")
        assert commands[0][:3] == ["wait", "--load", "networkidle"]
        assert commands[1:4] == OBSERVE
        assert commands[-1][0] == "eval" and '"stamp"' in commands[-1][1]
        reads = driver.observe_commands([Action(name="state", params={})], "stamp")
        assert reads[:3] == OBSERVE

    def test_a_settle_that_timed_out_does_not_spoil_the_read(self):
        entries = [
            _fail("Timeout waiting for networkidle"),
            _ok({"url": "https://a.example/"}),
            _ok({"title": "A"}),
            _ok({"snapshot": "- button [ref=e1]", "refs": {"e1": {}}}),
            _ok(),
        ]
        observed = observation_from(entries, settled=True)
        assert observed is not None and observed.refs == ["e1"]


class TestReceiptAnchors:
    """agent-browser 0.37.1's own strings, the receipts depend on them."""

    @pytest.mark.parametrize(
        "error, reason",
        [
            ("Unknown ref: @e9", "stale_ref"),
            ("Element not found: @e9", "not_found"),
            ("Element not found in the selected frame: iframe#f", "not_found"),
            ("No element found: getByRole('button')", "not_found"),
            ("No element found by css 'x'", "not_found"),
        ],
    )
    def test_a_locator_miss_did_not_start(self, error, reason):
        click = Action(name="click", params={"ref": "@e9"})
        (receipt,) = receipts_for([click], [commands_for(click, "/x")], [_fail(error)])
        assert (receipt.status, receipt.reason) == ("not_started", reason)

    @pytest.mark.parametrize(
        "error",
        [
            "Session with given id not found",
            "Frame was detached: node not found",
            "Element is not visible",
        ],
    )
    def test_anything_else_may_have_run(self, error):
        click = Action(name="click", params={"ref": "@e9"})
        (receipt,) = receipts_for([click], [commands_for(click, "/x")], [_fail(error)])
        assert receipt.status == "unknown"

    def test_a_lookup_error_on_an_action_with_no_ref_may_have_run(self):
        """A navigation's failure never reads as safe to repeat, whatever it
        says."""
        goto = Action(name="goto", params={"url": "https://a.example"})
        (receipt,) = receipts_for(
            [goto], [commands_for(goto, "/x")], [_fail("Element not found: x")]
        )
        assert receipt.status == "unknown"


class TestPageNotes:
    def test_dialog_warnings_and_why_the_read_failed_not_action_errors(self):
        acted = [_fail("Unknown ref: @e1"), _ok({"warning": "confirm: Leave?"})]
        observed = [_fail("A JavaScript confirm dialog is blocking the page")]
        assert driver.page_notes(acted, observed) == [
            "confirm: Leave?",
            "A JavaScript confirm dialog is blocking the page",
        ]


class TestBatchSize:
    def test_text_travels_unescaped_and_an_oversized_batch_is_refused(self):
        japanese = "日本語" * 3_000
        payload = driver.batch_payload([["fill", "@e1", japanese]])
        assert japanese in payload
        with pytest.raises(CallRefused, match="too long"):
            driver.batch_payload([["fill", "@e1", "x" * driver.MAX_BATCH_BYTES]])


# ---------------------------------------------------------------- the box


class _Box:
    """Records what runs in the box; *answers* maps a command fragment to its
    stdout, or to an exception to raise."""

    def __init__(self, answers: dict[str, object]):
        self.sandbox_id = "sb-1"
        self.answers = answers
        self.ran: list[tuple[str, dict]] = []

    async def run(self, sandbox, command, **kwargs):
        self.ran.append((command, kwargs))
        for fragment, answer in self.answers.items():
            if fragment in command:
                if isinstance(answer, BaseException):
                    raise answer
                return type("R", (), {"stdout": answer})()
        return type("R", (), {"stdout": ""})()


class _Failed(Exception):
    def __init__(self, stdout: str):
        super().__init__("exit 1")
        self.stdout = stdout


@pytest.fixture
def box(monkeypatch):
    def make(answers):
        fake = _Box(answers)
        monkeypatch.setattr(driver, "run_internal", fake.run)
        return fake

    return make


class TestTheBox:
    @pytest.mark.asyncio
    async def test_a_running_chrome_reuses_the_recorded_session(self, box):
        fake = box({"cat /tmp/otto-browser.session": "otto-0123456789ab"})
        assert await driver.ensure_browser(fake) == "otto-0123456789ab"
        commands = [c for c, _ in fake.ran]
        assert not any("google-chrome" in c for c in commands)
        assert not any("rm -f /tmp/otto-browser.session" in c for c in commands)

    @pytest.mark.asyncio
    async def test_a_closed_chrome_is_started_with_a_fresh_session_and_tab(self, box):
        fake = box(
            {
                "json/version >/dev/null\n": _Failed(""),
                "curl -sf --max-time 2 http://127.0.0.1:9222/json/version >/dev/null": None,
                "cat /tmp/otto-browser.session": "otto-0123456789ab",
            }
        )
        # The first probe fails, Chrome is started, and the wait loop passes.
        calls = {"n": 0}
        original = fake.run

        async def run(sandbox, command, **kwargs):
            if command.startswith("curl -sf"):
                calls["n"] += 1
                if calls["n"] == 1:
                    fake.ran.append((command, kwargs))
                    raise _Failed("")
            return await original(sandbox, command, **kwargs)

        driver.run_internal = run  # type: ignore[assignment]
        assert await driver.ensure_browser(fake) == "otto-0123456789ab"
        commands = [c for c, _ in fake.ran]
        assert any(c.startswith("google-chrome") for c in commands)
        assert any(c.startswith("rm -f /tmp/otto-browser.session") for c in commands)
        tab = next(kw for c, kw in fake.ran if "tab new" in c)
        assert tab["envs"] == driver._AB_ENVS

    @pytest.mark.asyncio
    async def test_chrome_that_never_listens_is_unavailable(self, box):
        fake = box({"curl -sf": _Failed(""), "for i in": _Failed("")})
        with pytest.raises(driver.BrowserUnavailable, match="did not start"):
            await driver.ensure_browser(fake)

    @pytest.mark.asyncio
    async def test_a_failed_install_says_what_npm_said(self, box):
        fake = box(
            {
                "agent-browser --version": _Failed(""),
                "npm install": _Failed("npm ERR! network request failed\n"),
            }
        )
        with pytest.raises(driver.BrowserUnavailable, match="network request failed"):
            await driver.ensure_browser(fake)

    @pytest.mark.asyncio
    async def test_a_session_that_cannot_be_recorded_is_unavailable(self, box):
        fake = box({"cat /tmp/otto-browser.session": "garbage"})
        with pytest.raises(driver.BrowserUnavailable, match="session"):
            await driver._session(fake, fresh=False)

    @pytest.mark.asyncio
    async def test_the_batch_dies_with_its_deadline_in_the_box(self, box):
        fake = box({"batch": json.dumps([_ok()])})
        await driver.run_batch(
            fake, "otto-0123456789ab", [["get", "url"]], bail=True, timeout=40
        )
        command, kwargs = fake.ran[-1]
        assert "timeout -k 2 40 agent-browser" in command
        assert kwargs["timeout"] == 45
        assert kwargs["envs"]["OTTO_BATCH"] == '[["get", "url"]]'
        assert {k: kwargs["envs"][k] for k in driver._AB_ENVS} == driver._AB_ENVS

    @pytest.mark.asyncio
    async def test_a_failed_batch_still_gives_its_entries(self, box):
        fake = box({"batch": _Failed(json.dumps([_ok(), _fail("Unknown ref: @e1")]))})
        entries = await driver.run_batch(
            fake, "otto-0123456789ab", [["a"], ["b"]], bail=True, timeout=10
        )
        assert [e["success"] for e in entries] == [True, False]

    @pytest.mark.asyncio
    async def test_a_lost_tab_is_bound_anew_and_the_batch_run_again(self, box):
        answers = iter(
            [json.dumps([_fail("tab gone", code="tab_gone")]), json.dumps([_ok()])]
        )
        fake = box({})

        async def run(sandbox, command, **kwargs):
            fake.ran.append((command, kwargs))
            if "batch" in command:
                return type("R", (), {"stdout": next(answers)})()
            return type("R", (), {"stdout": ""})()

        driver.run_internal = run  # type: ignore[assignment]
        entries = await driver.run_batch(
            fake, "otto-0123456789ab", [["get", "url"]], bail=True, timeout=10
        )
        assert entries == [_ok()]
        commands = [c for c, _ in fake.ran]
        assert sum("batch" in c for c in commands) == 2
        assert any("tab new" in c for c in commands)

    @pytest.mark.asyncio
    async def test_an_open_dialog_is_read_from_agent_browser_not_the_page(self, box):
        status = {"hasDialog": True, "type": "confirm", "message": "Leave?"}
        fake = box({"batch": json.dumps([_ok(status)])})
        note = await driver.open_dialog(fake, "otto-0123456789ab")
        assert note is not None and "confirm" in note and '"Leave?"' in note
        assert '[["dialog", "status"]]' == fake.ran[-1][1]["envs"]["OTTO_BATCH"]
        fake.answers["batch"] = json.dumps([_ok({"hasDialog": False})])
        assert await driver.open_dialog(fake, "otto-0123456789ab") is None


def test_a_page_that_was_not_stamped_leaves_its_refs_unchecked():
    core = [
        _ok({"url": "https://a.example/"}),
        _ok({"title": "A"}),
        _ok({"snapshot": "- button [ref=e1]", "refs": {"e1": {}}}),
    ]
    stamped = observation_from([*core, _ok()])
    unstamped = observation_from([*core, _fail("Execution context was destroyed")])
    assert stamped is not None and stamped.stamped
    assert unstamped is not None and not unstamped.stamped and unstamped.refs == ["e1"]
