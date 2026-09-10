"""The runner without a model: the prompt set per expert, the summary and
its comparison against the stored baseline, and the drift line that says
whether a paid run would measure anything new."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from backend.copilot.config import ChatConfig

from .assembly import load_baseline, load_fixtures, load_rubric, roster_experts
from .generation import Turn
from .models import (
    DimensionJudgement,
    Judgement,
    ReferencePrompt,
    ScoredResponse,
    Usage,
)
from .runner import (
    Job,
    RunOptions,
    cache_prefix,
    compare,
    drift,
    generate,
    plan_jobs,
    prompt_scores,
    run,
    separation,
    summarize,
    summarize_expert,
    wrong_spec_rows,
)
from .scorer import response_score

_RUNNER = "backend.copilot.eval.style.runner"


def _judgement(score: int) -> Judgement:
    rubric = load_rubric()
    return Judgement(
        scores={
            d.key: DimensionJudgement(score=score, evidence="q")
            for d in rubric.dimensions
        }
    )


def test_response_score_spans_the_scale():
    rubric = load_rubric()
    assert response_score(_judgement(1), rubric) == 0.0
    assert response_score(_judgement(5), rubric) == 100.0
    assert response_score(_judgement(3), rubric) == 50.0


def test_plan_assembles_each_experts_own_prompt_set():
    experts = roster_experts()
    jobs = plan_jobs(experts, load_fixtures(), RunOptions(repeats=2, control=3))
    for expert in experts:
        own = [j for j in jobs if j.expert.name == expert.name and j.arm == "expert"]
        assert len(own) == 60
        assert all(j.prompt.id.startswith(expert.name.lower()) for j in own)
        controls = [
            j for j in jobs if j.expert.name == expert.name and j.arm == "no_suffix"
        ]
        assert len(controls) == 3
        assert all(j.prompt.kind != "briefing_lede" for j in controls)


def test_the_control_arm_warms_one_prompt_prefix_for_the_whole_roster():
    """Its system prompt and tools are plain AutoPilot's whatever expert's
    prompts it runs; a per-expert key would pay the cache write three times."""
    jobs = plan_jobs(roster_experts(), load_fixtures(), RunOptions(control=3))
    controls = {cache_prefix(j) for j in jobs if j.arm == "no_suffix"}
    experts = {cache_prefix(j) for j in jobs if j.arm == "expert"}
    assert len(controls) == 1
    assert len(experts) == 6, "one per expert, chat and lede apart"


def test_plan_filters_kinds():
    jobs = plan_jobs(
        roster_experts(["Max"]), load_fixtures(), RunOptions(kinds=["failure"])
    )
    assert {j.prompt.kind for j in jobs} == {"failure"}
    assert len(jobs) == 6


def _row(
    expert: str,
    score: float | None,
    *,
    arm="expert",
    repeat=0,
    error=None,
    prompt_id: str | None = None,
    generation: Usage | None = None,
) -> ScoredResponse:
    return ScoredResponse(
        expert=expert,
        spec_expert=expert,
        arm=arm,
        kind="reply_draft",
        prompt_id=prompt_id or f"{expert.lower()}-reply-01",
        repeat=repeat,
        model="m",
        response="hi",
        score=score,
        error=error,
        generation=generation,
    )


def test_the_comparison_pairs_this_run_against_the_baseline_prompt_by_prompt():
    """Mean against mean would mostly report which prompts happened to be
    scored; the baseline stores every prompt's score so the pairing is real."""
    baseline = load_baseline()
    stored = next(b for b in baseline.experts if b.expert == "Max")
    prompts = sorted(stored.by_prompt)[:4]
    rows = [_row("Max", stored.by_prompt[pid] + 6.0, prompt_id=pid) for pid in prompts]
    comparison = compare(summarize_expert("Max", rows), rows, baseline)
    assert comparison.shared_prompts == 4
    assert comparison.paired_delta == 6.0
    assert comparison.paired_sem == 0.0
    assert comparison.baseline_mean == stored.scores.mean


def test_repeats_of_one_prompt_are_one_comparison_not_several():
    """Both sides average a prompt's repeats. Comparing each repeat against a
    single stored value counts one prompt's noise three times and understates
    the standard error."""
    baseline = load_baseline()
    stored = next(b for b in baseline.experts if b.expert == "Max")
    prompt = sorted(stored.by_prompt)[0]
    rows = [
        _row("Max", stored.by_prompt[prompt] + delta, prompt_id=prompt, repeat=i)
        for i, delta in enumerate((0.0, 6.0, 12.0))
    ]
    assert prompt_scores(rows, "Max") == {prompt: stored.by_prompt[prompt] + 6.0}
    comparison = compare(summarize_expert("Max", rows), rows, baseline)
    assert comparison.shared_prompts == 1
    assert comparison.paired_delta == 6.0
    assert comparison.paired_sem == 0.0


def test_run_options_refuse_a_hang_and_a_partial_baseline():
    """Concurrency 0 builds a semaphore nothing can acquire, and a filtered
    --write-baseline stores part of the set under a whole-set fingerprint."""
    for invalid in ({"concurrency": 0}, {"repeats": 0}, {"control": -1}):
        with pytest.raises(ValidationError):
            RunOptions(**invalid)
    with pytest.raises(ValidationError):
        RunOptions(write_baseline=True, experts=["Max"])
    with pytest.raises(ValidationError):
        RunOptions(write_baseline=True, kinds=["failure"])
    assert RunOptions(write_baseline=True).concurrency == 6
    assert RunOptions(experts=["Max"], kinds=["failure"]).repeats == 1


def test_the_comparison_says_so_when_a_prompt_set_has_no_baseline():
    baseline = load_baseline()
    rows = [_row("Max", 80.0, prompt_id="max-invented-01")]
    comparison = compare(summarize_expert("Max", rows), rows, baseline)
    assert comparison.shared_prompts == 0
    assert comparison.paired_delta is None
    assert comparison.baseline_mean is not None

    unknown = [_row("Nobody", 80.0)]
    assert (
        compare(summarize_expert("Nobody", unknown), unknown, baseline).baseline_mean
        is None
    )


def test_summary_reports_distribution_and_repeats():
    rows = [
        _row("Max", 60.0, repeat=0),
        _row("Max", 80.0, repeat=0),
        _row("Max", 100.0, repeat=1),
    ]
    summary = summarize_expert("Max", rows)
    assert summary.scores.n == 3
    assert summary.scores.mean == 80.0
    assert summary.scores.min == 60.0
    assert summary.by_repeat == [70.0, 100.0]
    assert summary.below_60 == 0
    assert summary.by_kind["reply_draft"].n == 3


def test_wrong_spec_rows_pair_each_response_with_every_other_expert():
    experts = roster_experts()
    rows = [_row("Maria", 80.0), _row("Max", None, error="x")]
    crossed = wrong_spec_rows(rows, experts)
    assert {(r.expert, r.spec_expert) for r in crossed} == {
        ("Maria", "Max"),
        ("Maria", "Frankie"),
    }
    assert all(r.arm == "wrong_spec" and r.response == "hi" for r in crossed)


def test_the_generation_cost_is_counted_once_across_the_cross_spec_copies():
    """A wrong-spec row re-judges an existing response; carrying its
    generation usage priced the generation leg three times over."""
    usage = Usage(model="m", input_tokens=100, output_tokens=10, cost_usd=1.0)
    rows = [_row("Maria", 80.0, generation=usage)]
    rows += wrong_spec_rows(rows, roster_experts())
    assert [r.generation for r in rows[1:]] == [None, None]
    result = summarize(
        rows,
        roster_experts(["Maria"]),
        load_baseline(),
        fingerprint_value="f",
        chat_model="m",
        lede_model="m",
        judge_model="m",
    )
    assert result.cost_usd == pytest.approx(1.0)
    assert result.input_tokens == 100


@pytest.mark.asyncio
async def test_a_turn_still_calling_tools_at_the_cap_is_an_error_not_a_zero():
    """Its visible text is a half-written tool narration, so scoring it would
    measure the harness. Run A scored six such rows 0-15 and dragged an
    expert's mean down 11 points."""
    capped = Turn(
        text="Let me check that run.",
        truncated=False,
        usage=Usage(model="m", cost_usd=0.01),
        tool_calls=["run_agent"] * 10,
        rounds=10,
        hit_round_cap=True,
    )
    expert = roster_experts(["Max"])[0]
    prompt = load_fixtures(["Max"])[0].prompts[0]
    generate_turn = AsyncMock(return_value=capped)
    with patch(f"{_RUNNER}.generate_turn", generate_turn):
        row = await generate(
            Job(expert=expert, arm="expert", prompt=prompt, repeat=0),
            MagicMock(),
            ChatConfig(),
            [expert],
            chat_model="m",
            lede_model="m",
        )
    assert row.score is None
    assert row.error is not None and "still calling tools after 10 rounds" in row.error
    assert generate_turn.await_count == 1, "a capped turn is not worth retrying"
    assert summarize_expert("Max", [row]).errors == 1


def test_separation_compares_own_spec_with_the_controls():
    rows = [
        _row("Maria", 85.0, prompt_id="p1"),
        _row("Maria", 75.0, prompt_id="p2"),
        _row("Maria", 40.0, arm="wrong_spec", prompt_id="p1"),
        _row("Maria", 50.0, arm="no_suffix", prompt_id="p2"),
    ]
    sep = separation(rows)
    assert sep is not None
    assert sep.right_spec_mean == 80.0
    assert sep.wrong_spec_mean == 40.0
    assert sep.no_suffix_mean == 50.0
    assert sep.gap == 30.0
    assert sep.paired is not None and sep.paired.n == 1
    assert not sep.separated, "one comparison decides nothing"
    assert separation([_row("Maria", 85.0)]) is None


def _paired_rows(own: list[float], wrong: list[float]) -> list[ScoredResponse]:
    return [
        row
        for i, (o, w) in enumerate(zip(own, wrong))
        for row in (
            _row("Maria", o, prompt_id=f"p{i}"),
            _row("Maria", w, arm="wrong_spec", prompt_id=f"p{i}"),
        )
    ]


def test_pairing_sees_an_advantage_that_the_run_wide_sd_hides():
    """Run A's verdict: +26 points on 174 pairs, called NOT separated because
    six unfinished turns scored near zero in both arms and doubled the SD."""
    own = [85.0] * 20 + [0.0] * 4
    wrong = [60.0] * 20 + [0.0, 0.0, 0.0, 5.0]
    sep = separation(_paired_rows(own, wrong))
    assert sep is not None and sep.paired is not None
    assert sep.gap is not None and sep.gap < sep.right_spec_sd
    assert sep.paired.n == 24
    assert sep.paired.win_rate == pytest.approx(20 / 24, abs=0.001)
    assert sep.separated


def test_a_judge_that_scores_both_specs_alike_is_not_separated():
    own = [80.0, 60.0, 70.0, 90.0] * 6
    wrong = [82.0, 58.0, 71.0, 88.0] * 6
    sep = separation(_paired_rows(own, wrong))
    assert sep is not None and sep.paired is not None
    assert sep.paired.n == 24
    assert not sep.separated


@pytest.fixture
def stubbed_models(tmp_path: Path):
    """Generation and judging without a model, the router pinned to a name."""
    from .assembly import RoutedModel

    routed = RoutedModel(
        mode="thinking",
        slug="anthropic/claude-x",
        transport_slug="anthropic/claude-x",
        source="env",
    )
    usage = Usage(model="m", input_tokens=10, output_tokens=5, cost_usd=0.001)
    turn = Turn(text="text", truncated=False, usage=usage, tool_calls=["memory_search"])
    judge = AsyncMock(return_value=(_judgement(4), usage))
    with (
        patch(f"{_RUNNER}.resolve_chat_model", AsyncMock(return_value=routed)),
        patch(f"{_RUNNER}.chat_client", MagicMock()),
        patch(f"{_RUNNER}.generate_turn", AsyncMock(return_value=turn)),
        patch(f"{_RUNNER}.generate_lede", AsyncMock(return_value=("lede", usage))),
        patch(f"{_RUNNER}.judge_response", judge),
        patch(f"{_RUNNER}.save_baseline") as save_baseline,
    ):
        yield judge, save_baseline


@pytest.mark.asyncio
async def test_run_scores_every_prompt_and_reads_it_against_the_baseline(
    stubbed_models, tmp_path: Path
):
    judge, save_baseline = stubbed_models
    out = tmp_path / "r.json"
    result = await run(RunOptions(experts=["Max"], out=out))
    assert result is not None
    assert result.experts[0].scores.n == 30
    assert result.experts[0].scores.mean == 75.0
    assert result.experts[0].tool_calls == 27
    assert judge.await_count == 30
    assert result.cost_usd == pytest.approx(0.06)
    (comparison,) = result.comparison
    assert comparison.expert == "Max"
    assert comparison.shared_prompts == 28, "the baseline's two unscored prompts"
    save_baseline.assert_not_called()
    written = json.loads(out.read_text())
    assert written["fingerprint"] == result.fingerprint
    assert len(written["responses"]) == 30


@pytest.mark.asyncio
async def test_write_baseline_stores_every_prompt_of_this_run(
    stubbed_models, tmp_path: Path
):
    _, save_baseline = stubbed_models
    result = await run(RunOptions(out=tmp_path / "r.json", write_baseline=True))
    assert result is not None
    written = save_baseline.call_args.args[0]
    assert {e.expert for e in written.experts} == {e.name for e in roster_experts()}
    assert all(len(e.by_prompt) == 30 for e in written.experts)
    assert written.fingerprint == result.fingerprint


def test_drift_names_the_components_that_moved_since_the_baseline():
    """The one line that decides whether a run is worth paying for."""
    baseline = load_baseline()
    assert "unchanged since" in drift(dict(baseline.parts), baseline)
    moved = dict(baseline.parts) | {"rubric": "0" * 64}
    assert "1 component(s) changed" in drift(moved, baseline)
    assert "rubric" in drift(moved, baseline)
    assert "records no fingerprint" in drift(
        dict(baseline.parts), baseline.model_copy(update={"parts": {}})
    )


@pytest.mark.asyncio
async def test_a_dry_run_makes_no_calls(stubbed_models, tmp_path: Path):
    judge, _ = stubbed_models
    assert await run(RunOptions(experts=["Max"], dry_run=True)) is None
    assert judge.await_count == 0


@pytest.mark.asyncio
async def test_the_fingerprint_survives_a_change_of_transport(
    stubbed_models, tmp_path: Path, monkeypatch
):
    """We run through OpenRouter and CI ran direct-Anthropic, which spell the
    same model differently. A fingerprint that moved with the transport would
    report a change nobody made."""
    # Set both credentials so the transport is decided by the flag alone;
    # CI has neither, and without them both arms resolve direct-Anthropic
    # and the test passes without comparing anything.
    monkeypatch.setenv("CHAT_API_KEY", "style-eval-test-key")
    monkeypatch.setenv("CHAT_BASE_URL", "https://openrouter.ai/api/v1")
    fingerprints = []
    for openrouter in ("true", "false"):
        monkeypatch.setenv("CHAT_USE_OPENROUTER", openrouter)
        result = await run(RunOptions(experts=["Max"], out=tmp_path / "r.json"))
        assert result is not None
        fingerprints.append((result.chat_model, result.fingerprint))
    (or_model, or_fp), (direct_model, direct_fp) = fingerprints
    assert or_model != direct_model
    assert or_fp == direct_fp


def test_reference_prompt_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        ReferencePrompt(id="x", kind="failure", prompt="")
    with pytest.raises(ValueError):
        ReferencePrompt(id="x", kind="briefing_lede", prompt="p")
