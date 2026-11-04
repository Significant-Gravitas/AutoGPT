"""The task contract: one shape for every task, whatever the work.

The three test examples ("Pick a project tool", "Hire 5 experts", "Launch
pricing page") are only inputs to the same generic calls; nothing here or in
the code knows about them."""

from datetime import UTC, datetime

import pytest

from backend.api.features.tasks import contract as c
from backend.api.features.tasks.models import (
    Actor,
    Attachment,
    CardApproval,
    ContractState,
    PendingInput,
    Plan,
    PlanStatus,
    PlanStep,
    Task,
    TaskPriority,
    TaskStatus,
)

_NOW = datetime(2026, 11, 4, tzinfo=UTC)


def _patch(contract: c.Contract, **fields) -> c.Contract:
    updated, _ = c.apply_patch(contract, c.TaskPatch.model_validate(fields))
    return updated


def _start(goal: str, done_when=None, plan=None) -> c.Contract:
    return _patch(c.Contract(goal=goal), done_when=done_when, plan=plan)


def _task(status: TaskStatus = TaskStatus.IN_PROGRESS, **extra) -> Task:
    return Task(
        id="task-1",
        number=1,
        key="T-1",
        title="Pick a project tool",
        status=status,
        priority=TaskPriority.NONE,
        created_at=_NOW,
        updated_at=_NOW,
        **extra,
    )


# --- update_task: one patch, any subset ------------------------------------


def test_a_quick_task_needs_one_check_and_no_plan():
    contract = _start("Rename the Q3 report", done_when=["The file is renamed"])

    assert [x.id for x in contract.done_when] == ["d1"]
    assert contract.plan == []


def test_sizes_are_guidance_not_rules():
    many = [f"Step {i}" for i in range(12)]

    contract = _start("Hire 5 experts", plan=many)

    assert len(contract.plan) == 12


def test_runaway_lists_are_capped():
    contract = _start("x", plan=[f"Step {i}" for i in range(50)])

    assert len(contract.plan) == c.MAX_ITEMS


def test_one_patch_moves_steps_and_criteria_and_finishes():
    contract = _start(
        "Pick a project tool",
        done_when=["Three tools compared", "One recommended"],
        plan=["Research", "Compare", "Recommend"],
    )
    contract, evidence, _ = c.add_evidence(
        contract, "quote", "Comparison table", "Linear wins on speed"
    )

    contract = _patch(
        contract,
        steps=[{"id": "s1", "status": "done"}, {"id": "s2", "status": "done"}],
        criteria=[
            {"id": "d1", "status": "met", "evidence_ids": [evidence.id]},
            {"id": "d2", "status": "met"},
        ],
        result={"summary": "Use Linear.\nIt is fastest for a team of 8."},
    )

    assert [s.status for s in contract.plan][:2] == [c.StepStatus.DONE] * 2
    assert c.all_met(contract)
    assert contract.result is not None
    assert contract.result.primary_evidence_id == evidence.id


def test_a_patch_reports_each_field_once():
    contract = _start("x", plan=["a", "b"])

    _, patches = c.apply_patch(
        contract,
        c.TaskPatch(
            steps=[
                c.StepChange(id="s1", status=c.StepStatus.RUNNING),
                c.StepChange(id="s1", status=c.StepStatus.DONE),
            ]
        ),
    )

    assert [p.field for p in patches] == ["plan"]
    assert patches[0].value[0]["status"] == "done"


def test_an_unknown_id_is_refused_with_the_known_ones():
    contract = _start("x", plan=["a"])

    with pytest.raises(c.ContractError, match="Known ids: s1"):
        _patch(contract, steps=[{"id": "s9", "status": "done"}])


def test_a_long_result_is_kept_to_two_lines():
    contract = _patch(_start("x"), result={"summary": "one\ntwo\nthree\nfour"})

    assert contract.result is not None
    assert contract.result.summary == "one\ntwo three four"


def test_finishing_clears_the_ask_and_the_blocker():
    contract, _, _ = c.open_ask(_start("x"), "Which plan?")
    contract = _patch(contract, blocker={"what_stopped_it": "No access"})

    contract = _patch(contract, result={"summary": "Done anyway."})

    assert contract.ask is None
    assert contract.blocker is None


def test_revising_the_plan_keeps_finished_steps_and_moves_the_ask():
    contract = _start("Launch pricing page", plan=["Draft copy", "Design", "Ship"])
    contract = _patch(contract, steps=[{"id": "s1", "status": "done"}])
    contract, _, _ = c.open_ask(contract, "Approve the page?", blocking_step_id="s3")

    contract = _patch(contract, plan=["Research", "Draft copy", "Ship"])

    assert [(s.id, s.title, s.status) for s in contract.plan] == [
        ("s1", "Research", c.StepStatus.PENDING),
        ("s2", "Draft copy", c.StepStatus.DONE),
        ("s3", "Ship", c.StepStatus.PENDING),
    ]
    assert contract.ask is not None
    assert contract.ask.blocking_step_id == "s3"


def test_revising_the_checks_keeps_their_proof():
    contract = _start("x", done_when=["A", "B"])
    contract, _, _ = c.add_evidence(
        contract, "link", "Doc", "The doc", criterion_ids=["d2"]
    )

    contract = _patch(contract, done_when=["B", "C"])

    assert contract.done_when[0].evidence_ids == ["e1"]
    assert contract.evidence[0].criterion_ids == ["d1"]


# --- evidence and asks ------------------------------------------------------


def test_evidence_kinds_are_open():
    contract, item, _ = c.add_evidence(_start("x"), "Loom Video", "Demo", "")

    assert item.kind == "loom video"
    assert item.summary == "Demo"
    assert contract.evidence == [item]


def test_evidence_must_point_at_real_checks():
    with pytest.raises(c.ContractError):
        c.add_evidence(_start("x"), "link", "Doc", "", criterion_ids=["d1"])


def test_a_second_question_joins_the_open_one():
    contract, first, _ = c.open_ask(
        _start("Hire 5 experts"), "Which budget?", options=["$50", "$100"]
    )

    contract, merged, _ = c.open_ask(
        contract, "Which time zone?", options=["$100", "UTC"]
    )

    assert merged.id == first.id
    assert merged.question == "Which budget?\nWhich time zone?"
    assert merged.options == ["$50", "$100", "UTC"]


# --- state ------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,blocker,ask,pending,state",
    [
        (TaskStatus.IN_PROGRESS, False, False, False, ContractState.WORKING),
        (TaskStatus.IN_PROGRESS, False, True, False, ContractState.NEEDS_YOU),
        (TaskStatus.BLOCKED, False, False, True, ContractState.NEEDS_YOU),
        (TaskStatus.BLOCKED, True, False, False, ContractState.STUCK),
        (TaskStatus.BLOCKED, False, False, False, ContractState.STUCK),
        (TaskStatus.IN_REVIEW, False, False, False, ContractState.NEEDS_YOU),
        (TaskStatus.DONE, True, False, False, ContractState.DONE),
        (TaskStatus.CANCELLED, False, True, False, ContractState.STOPPED),
    ],
)
def test_state_is_derived(status, blocker, ask, pending, state):
    contract = c.Contract(
        goal="x",
        blocker=c.make_blocker(None, "No access") if blocker else None,
        ask=c.Ask(id="a", question="?") if ask else None,
    )
    waiting = (
        PendingInput(id="p", question="?", asked_by=Actor.otto(), asked_at=_NOW)
        if pending
        else None
    )

    assert c.derive_state(status, contract, waiting) == state


# --- errors become blockers -------------------------------------------------


@pytest.mark.parametrize(
    "message,cause",
    [
        ("Hiring isn't allowed in a chat started by an automation.", "permission"),
        ("HTTP 403 Forbidden", "permission"),
        ("You have reached your monthly limit", "limit"),
        ("Google credentials are not connected", "missing_input"),
        ("Request timed out", "external_error"),
        ("Something odd", "unknown"),
    ],
)
def test_an_error_message_suggests_a_cause(message, cause):
    assert c.classify_error(message) == cause


def test_a_blocker_fills_in_what_was_left_out():
    contract = _start("Hire 5 experts", plan=["Pick roles", "Hire"])
    contract = _patch(contract, steps=[{"id": "s1", "status": "done"}])

    blocker = c.make_blocker(
        contract, "Hiring isn't allowed here.", tool_name="hire_expert"
    )

    assert blocker.cause == "permission"
    assert blocker.what_stopped_it == "Hiring isn't allowed here. (hire_expert)"
    assert blocker.done_so_far == "1 of 2 steps done (Pick roles)."
    assert blocker.short == "Hiring isn't allowed here."
    assert blocker.next_actions == ["allow", "do_in_chat", "take_over"]


def test_a_cause_and_actions_outside_the_suggestions_are_kept():
    contract = _patch(
        _start("x"),
        blocker={
            "what_stopped_it": "The vendor wants a call",
            "cause": "Needs a human call",
            "next_actions": ["book_a_call"],
        },
    )

    assert contract.blocker is not None
    assert contract.blocker.cause == "needs_a_human_call"
    assert contract.blocker.next_actions == ["book_a_call"]


def test_sentences_written_as_next_actions_fall_back_to_the_causes_actions():
    contract = _patch(
        _start("x"),
        blocker={
            "what_stopped_it": "Slack isn't connected yet.",
            "cause": "permission",
            "next_actions": [
                "Connect Slack using the sign-in card",
                "Then I read the last 7 days of #support",
            ],
        },
    )

    assert contract.blocker is not None
    assert contract.blocker.next_actions == ["allow", "do_in_chat", "take_over"]


def test_a_tool_error_is_read_from_its_output():
    assert c.tool_error('{"type": "error", "message": "No access"}') == "No access"
    assert c.tool_error({"type": "expert_change_proposed"}) is None
    assert c.tool_error("plain text") is None


def test_a_failure_retried_successfully_is_not_a_blocker():
    calls = [("web_search", "timed out"), ("web_search", None)]

    assert c.unrecovered_failure(calls) is None


def test_the_last_unrecovered_failure_is_the_one_reported():
    calls = [
        ("hire_expert", "not allowed"),
        ("update_task", "No step with id s9"),
        ("web_search", None),
    ]

    assert c.unrecovered_failure(calls, frozenset({"update_task"})) == (
        "hire_expert",
        "not allowed",
    )


def test_only_a_failure_retrying_cant_fix_stops_a_finished_turn():
    assert c.turn_blocker_reason(False, ("hire_expert", "not allowed")) == (
        "not allowed",
        "hire_expert",
    )
    assert c.turn_blocker_reason(False, ("web_search", "timed out")) is None
    assert c.turn_blocker_reason(True, None) == (
        "The work stopped with an error.",
        None,
    )


def test_a_logged_effect_becomes_evidence_and_a_delete_does_not():
    found = c.evidence_from_event(
        "FILE", "file.written", "pricing.html", "file-1", {"path": "/site/pricing"}
    )

    assert found == {
        "kind": "file",
        "title": "pricing.html",
        "summary": "pricing.html: /site/pricing",
        "ref": "file-1",
    }
    assert c.evidence_from_event("FILE", "file.deleted", "x", "f", {}) is None


# --- tasks from before the contract ------------------------------------------


def test_a_task_without_a_contract_gets_one_from_what_it_has():
    task = _task(TaskStatus.DONE, outcome_summary="Picked Linear.")
    plan = Plan(
        id="plan-1",
        status=PlanStatus.CONFIRMED,
        goal="Pick a project tool",
        steps=[PlanStep(title="Research"), PlanStep(title="Recommend")],
        created_at=_NOW,
    )
    attachments = [
        Attachment(
            id="att-1",
            kind="url",
            title="Linear pricing",
            url="https://linear.app/pricing",
            created_at=_NOW,
            added_by=Actor.otto(),
        )
    ]

    contract = c.fallback_contract(task, attachments, plan)

    assert contract.goal == "Pick a project tool"
    assert [s.status for s in contract.plan] == [c.StepStatus.DONE] * 2
    assert contract.evidence[0].kind == "link"
    assert contract.result == c.Result(
        summary="Picked Linear.", primary_evidence_id="e1"
    )


def test_a_stuck_task_without_a_contract_explains_itself():
    contract = c.fallback_contract(
        _task(TaskStatus.BLOCKED), [], None, "Otto's work turn stopped with an error."
    )

    assert contract.blocker is not None
    assert contract.blocker.what_stopped_it == "Otto's work turn stopped with an error."
    assert contract.blocker.next_actions


def test_a_contract_that_no_longer_parses_falls_back():
    assert c.parse_contract({"goal": "x", "plan": [{"bad": 1}]}) is None
    assert c.parse_contract(None) is None


# --- parallel rows, steps that follow the work, the line about now ----------


def test_parallel_work_is_rows_inside_one_step():
    contract = _start("Hire 5 experts", plan=["Pick roles", "Hire each"])

    contract = _patch(
        contract,
        steps=[
            {
                "id": "s2",
                "rows": [
                    {"title": "Designer", "status": "done"},
                    {"title": "Writer", "status": "running"},
                    {"title": "Analyst"},
                ],
            }
        ],
    )
    contract = _patch(
        contract, steps=[{"id": "s2", "rows": [{"title": "Writer", "status": "done"}]}]
    )

    step = contract.plan[1]
    assert [(r.title, r.status) for r in step.rows] == [
        ("Designer", c.StepStatus.DONE),
        ("Writer", c.StepStatus.DONE),
        ("Analyst", c.StepStatus.PENDING),
    ]
    assert step.status == c.StepStatus.RUNNING
    assert c.now_line("running", contract) == "Hire each · 2 of 3"


def test_a_step_settles_when_its_rows_do():
    contract = _start("x", plan=["Read pages"])

    contract = _patch(
        contract,
        steps=[
            {
                "id": "s1",
                "rows": [
                    {"title": "a.com", "status": "done"},
                    {"title": "b.com", "status": "failed"},
                ],
            }
        ],
    )

    assert contract.plan[0].status == c.StepStatus.FAILED


def test_starting_a_step_finishes_the_one_before():
    contract = _start("x", plan=["Research", "Compare"])
    contract = _patch(contract, steps=[{"id": "s1", "status": "running"}])

    contract = _patch(contract, steps=[{"id": "s2", "status": "running"}])

    assert [s.status for s in contract.plan] == [
        c.StepStatus.DONE,
        c.StepStatus.RUNNING,
    ]
    assert c.now_line("running", contract) == "Compare · step 2 of 2"


def test_a_blocker_fails_the_running_step_and_keeps_the_rest():
    contract = _start("x", plan=["Read pages"])
    contract = _patch(
        contract,
        steps=[
            {
                "id": "s1",
                "rows": [
                    {"title": "a.com", "status": "done"},
                    {"title": "b.com", "status": "running"},
                ],
            }
        ],
    )

    contract = _patch(contract, blocker={"what_stopped_it": "b.com timed out"})

    step = contract.plan[0]
    assert step.status == c.StepStatus.FAILED
    assert [r.status for r in step.rows] == [c.StepStatus.DONE, c.StepStatus.FAILED]
    assert contract.blocker is not None
    assert contract.blocker.done_so_far == "0 of 1 steps done; Read pages: 1 of 2 done."
    assert contract.blocker.next_actions == ["retry_failed_part", "skip", "take_over"]


def test_finishing_settles_running_work_as_done():
    contract = _patch(
        _start("x", plan=["Only"]), steps=[{"id": "s1", "status": "running"}]
    )

    contract = _patch(contract, result={"summary": "Done."})

    assert contract.plan[0].status == c.StepStatus.DONE


def test_skipped_checks_count_as_closed():
    contract = _start("x", done_when=["A", "B"])

    contract = _patch(
        contract,
        criteria=[{"id": "d1", "status": "met"}, {"id": "d2", "status": "skipped"}],
    )

    assert c.all_met(contract)
    assert c.now_line("done", contract) == "Done"
    finished = _patch(contract, result={"summary": "Use Linear.\nIt is fast."})
    assert c.now_line("done", finished) == "Use Linear."


def test_an_ask_with_options_always_has_a_default():
    _, ask, _ = c.open_ask(_start("x"), "Publish?", options=["Now", "Monday"])

    assert ask.default_option == "Now"


def test_a_goal_change_after_planning_is_remembered():
    contract = _start("Launch pricing page", plan=["Draft", "Ship"])

    contract = _patch(contract, goal="Launch pricing page in German too")

    assert contract.previous_goal == "Launch pricing page"
    assert _start("First goal").previous_goal is None


def test_a_long_goal_is_kept_whole():
    goal = (
        "For this test use the public repo pmndrs/zustand instead, read-only via "
        "the public GitHub API. Triage its 10 newest open issues and draft "
        "release notes from PRs merged in the last 14 days, grouped by feature, "
        "fix and chore, with a one-line summary on top."
    )

    assert _patch(_start("Triage issues"), goal=goal).goal == goal


def test_a_line_over_its_limit_is_cut_between_words():
    assert c.one_line("release notes from PRs merged", 24) == "release notes from PRs…"


# --- the detailed status and the lanes it folds into ------------------------


@pytest.mark.parametrize(
    "signals,raw,state",
    [
        ({"running": True, "quiet_minutes": 1}, "running", ContractState.WORKING),
        ({"quiet_minutes": 2}, "idle", ContractState.WORKING),
        ({"quiet_minutes": 30}, "no_signal", ContractState.STUCK),
        ({"running": True, "quiet_minutes": 30}, "no_signal", ContractState.STUCK),
        (
            {"quiet_minutes": 30, "open_subtasks": 2},
            "waiting_on_subtasks",
            ContractState.WORKING,
        ),
    ],
)
def test_a_working_task_that_goes_quiet_is_stuck(signals, raw, state):
    assert c.raw_status(TaskStatus.IN_PROGRESS, None, None, **signals) == raw
    assert c.state_of(raw) == state


def test_a_new_raw_status_counts_as_working_until_mapped():
    assert c.state_of("waiting_on_vendor") == ContractState.WORKING


def test_a_quiet_task_shows_a_blocker_with_a_retry():
    blocker = c.card_blocker(None, "no_signal", TaskStatus.IN_PROGRESS)

    assert blocker is not None
    assert blocker.cause == "no_signal"
    assert blocker.next_actions[0] == "retry_failed_part"
    assert blocker.next_actions == ["retry_failed_part"]
    assert c.now_line("no_signal", None) == "No progress for 15 minutes."


@pytest.mark.parametrize(
    "raw,line",
    [
        ("asked", "Needs your answer"),
        ("awaiting_accept", "Needs your OK"),
        ("queued", "Starting soon"),
        ("cancelled", "Cancelled"),
        ("blocked", "The work stopped before it finished."),
    ],
)
def test_the_line_about_now(raw, line):
    assert c.now_line(raw, None) == line


def test_a_long_reason_gets_a_label_never_half_a_sentence():
    reason = (
        "The vendor portal refused the login because the account needs a second "
        "factor that only the account owner can approve."
    )

    assert c.short_reason(reason, "permission") == "Needs your permission"
    assert c.short_reason("Site timed out. Retried twice.", "external_error") == (
        "Site timed out."
    )


# --- a result that says the work wasn't done fails the task ----------------


def test_a_result_saying_the_file_was_missing_is_a_blocker_not_done():
    contract = _start("Summarize the Q3 report PDF", done_when=["Summary written"])

    contract = _patch(
        contract,
        criteria=[{"id": "d1", "status": "met"}],
        result={"summary": "I couldn't find the **Q3 report PDF**; none is attached."},
    )

    assert contract.result is None
    assert contract.blocker is not None
    assert contract.blocker.cause == "missing_input"
    assert contract.blocker.short == (
        "I couldn't find the Q3 report PDF; none is attached."
    )
    assert contract.done_when[0].status == c.CriterionStatus.UNMET
    assert c.now_line("blocked", contract) == contract.blocker.short


def test_a_blocker_and_a_result_in_one_call_keep_the_blocker():
    contract = _patch(
        _start("x"),
        blocker={"what_stopped_it": "The export wasn't found."},
        result={"summary": "Done."},
    )

    assert contract.result is None
    assert contract.blocker is not None


def test_a_check_with_proof_stays_met_when_the_work_fails():
    contract, evidence, _ = c.add_evidence(
        _start("x", done_when=["A", "B"]), "file", "a.md", "A", criterion_ids=["d1"]
    )
    contract = _patch(
        contract,
        criteria=[
            {"id": "d1", "status": "met", "evidence_ids": [evidence.id]},
            {"id": "d2", "status": "met"},
        ],
    )

    reopened, _ = c.reopen_as_blocked(contract, c.make_blocker(contract, "No access"))

    assert [x.status for x in reopened.done_when] == [
        c.CriterionStatus.MET,
        c.CriterionStatus.UNMET,
    ]


def test_a_plain_result_still_finishes():
    contract = _patch(_start("x"), result={"summary": "Use Linear; no issues found."})

    assert contract.result is not None
    assert contract.blocker is None


@pytest.mark.parametrize(
    "statuses,review,done",
    [
        ([], False, True),
        (["pending"], False, True),
        (["met", "skipped"], False, True),
        (["met", "unmet"], False, False),
        (["met"], True, False),
    ],
)
def test_only_an_unmet_check_or_a_review_holds_a_result(statuses, review, done):
    contract = _start("x", done_when=[f"c{i}" for i in range(len(statuses))])
    contract = _patch(
        contract,
        criteria=[
            {"id": f"d{i + 1}", "status": s}
            for i, s in enumerate(statuses)
            if s != "pending"
        ],
    )

    assert c.finishes(contract, review) is done


def test_short_action_labels_become_keys():
    assert c.action_keys(["Retry", "Upload file", "skip", "Book a quick call now"]) == [
        "retry_failed_part",
        "upload_file",
        "skip",
    ]


def test_open_approvals_need_the_user():
    contract = c.Contract(
        goal="Hire",
        approvals=[CardApproval(confirmation_id="x", name="Sam")],
    )

    raw = c.raw_status(TaskStatus.BLOCKED, contract, None)

    assert raw == "asked"
    assert c.state_of(raw) == ContractState.NEEDS_YOU
    assert c.now_line(raw, contract) == "Needs your approval"
