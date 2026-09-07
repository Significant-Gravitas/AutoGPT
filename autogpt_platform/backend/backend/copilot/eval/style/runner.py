"""Run the gate: generate every reference prompt in the expert's voice, judge
each response, aggregate per expert, and decide against the threshold.

    poetry run expert-style-eval --dry-run          # prompt sizes, no calls
    poetry run expert-style-eval --repeats 3 --cross-spec --control 3
    poetry run expert-style-eval --gate             # exit 1 below threshold

``--gate`` skips the paid legs while the fingerprint in ``gate.json`` still
matches the assembled prompts, models, fixtures and rubric; a passing gate
run writes the new fingerprint back so the next unchanged run is free.
"""

import argparse
import asyncio
import logging
import statistics
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import openai
from pydantic import BaseModel

from backend.api.features.experts.models import Expert
from backend.copilot.briefing.narrative import _MAX_NARRATIVE_CHARS, NarrativeResponse
from backend.copilot.config import ChatConfig
from backend.copilot.dream.llm import structured_completion
from backend.copilot.eval.metrics import mean, percentile
from backend.copilot.model_normalize import normalize_model_for_transport

from .assembly import (
    attach_workflows,
    chat_system_prompt,
    fingerprint,
    lede_prompt,
    load_fixtures,
    load_gate,
    load_rubric,
    resolve_chat_model,
    roster_experts,
    save_gate,
    user_prefix,
)
from .generation import (
    GENERATION_TIMEOUT_SECONDS,
    chat_client,
    expert_tools,
    generate_turn,
)
from .models import (
    Arm,
    Distribution,
    ExpertFixture,
    ExpertSummary,
    GateOutcome,
    PairedAdvantage,
    PromptKind,
    ReferencePrompt,
    Rubric,
    ScoredResponse,
    Separation,
    StyleEvalResult,
    Usage,
)
from .scorer import judge_response, response_score, to_usage

logger = logging.getLogger(__name__)

LEDE_MAX_OUTPUT_TOKENS = 200
# An expert with more errored rows than this cannot be scored honestly.
MAX_ERRORS_PER_EXPERT = 3
# Paired own-vs-wrong-spec comparisons needed before the separation verdict
# means anything, and the share of them the own spec must win.
MIN_PAIRED_COMPARISONS = 20
MIN_PAIRED_WIN_RATE = 0.6


class RoundCapReached(Exception):
    """The turn was still calling tools at the round cap, so it has no
    finished answer to judge."""


class RunOptions(BaseModel):
    experts: list[str] | None = None
    kinds: list[PromptKind] | None = None
    repeats: int = 1
    model: str | None = None
    judge: str | None = None
    concurrency: int = 6
    out: Path = Path("expert_style_results.json")
    gate: bool = False
    force: bool = False
    dry_run: bool = False
    cross_spec: bool = False
    # Prompts per expert to also run as plain AutoPilot, no suffix.
    control: int = 0


class Job(BaseModel):
    expert: Expert
    arm: Arm
    prompt: ReferencePrompt
    repeat: int


async def run(options: RunOptions) -> tuple[StyleEvalResult | None, int]:
    """Returns the result (``None`` on a dry run) and the process exit code."""
    config = ChatConfig()
    fixtures = load_fixtures(options.experts)
    fixture_by_name = {f.expert: f for f in fixtures}
    experts = [
        attach_workflows(e, fixture_by_name[e.name])
        for e in roster_experts(options.experts)
    ]
    rubric, gate = load_rubric(), load_gate()
    routed = await resolve_chat_model(config)
    # The fingerprint reads the ROUTED names, not the transport ones: the
    # OpenRouter and direct-Anthropic spellings of one model differ, so
    # fingerprinting the transport name would never match between a local
    # run and CI, and the skip that pays for the broad path list would never
    # fire where it matters.
    chat_route = options.model or routed.slug
    judge_route = options.judge or gate.judge_model
    chat_model = normalize_model_for_transport(chat_route, config)
    lede_model = normalize_model_for_transport(config.title_model, config)
    judge_model = normalize_model_for_transport(judge_route, config)
    current = fingerprint(
        experts,
        fixtures,
        rubric,
        chat_model=chat_route,
        lede_model=config.title_model,
        judge_model=judge_route,
    )
    print(
        f"chat model {chat_model} ({routed.mode}/standard via {routed.source}"
        f"{', overridden' if options.model else ''}); lede model {lede_model}; "
        f"judge {judge_model}; fingerprint {current}"
    )
    if options.dry_run:
        _print_dry_run(experts, fixtures)
        return None, 0
    if options.gate and not options.force and current == gate.last_gated_fingerprint:
        print(f"gate: prompts and models unchanged since {gate.last_gated_at}; skipped")
        return None, 0

    jobs = plan_jobs(experts, fixtures, options)
    rows = await generate_all(
        jobs, options, config=config, chat_model=chat_model, lede_model=lede_model
    )
    by_name = {e.name: e for e in experts}
    if options.cross_spec:
        rows += wrong_spec_rows(rows, experts)
    await judge_all(rows, by_name, rubric, judge_model=judge_model, options=options)

    result = summarize(
        rows,
        experts,
        rubric,
        fingerprint_value=current,
        chat_model=chat_model,
        lede_model=lede_model,
        judge_model=judge_model,
        threshold=gate.pass_threshold if options.gate else None,
    )
    options.out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print_summary(result, options.out)
    if result.gate is None:
        return result, 0
    if result.gate.passed:
        gate.last_gated_fingerprint = current
        gate.last_gated_at = result.ts
        save_gate(gate)
    return result, 0 if result.gate.passed else 1


def plan_jobs(
    experts: list[Expert], fixtures: list[ExpertFixture], options: RunOptions
) -> list[Job]:
    by_name = {f.expert: f for f in fixtures}
    jobs: list[Job] = []
    for expert in experts:
        prompts = [
            p
            for p in by_name[expert.name].prompts
            if options.kinds is None or p.kind in options.kinds
        ]
        jobs += [
            Job(expert=expert, arm="expert", prompt=p, repeat=r)
            for r in range(options.repeats)
            for p in prompts
        ]
        chat_prompts = [p for p in prompts if p.kind != "briefing_lede"]
        jobs += [
            Job(expert=expert, arm="no_suffix", prompt=p, repeat=0)
            for p in chat_prompts[: options.control]
        ]
    return jobs


async def generate_all(
    jobs: list[Job],
    options: RunOptions,
    *,
    config: ChatConfig,
    chat_model: str,
    lede_model: str,
) -> list[ScoredResponse]:
    semaphore = asyncio.Semaphore(options.concurrency)
    client = chat_client(config)
    roster = [job.expert for job in jobs]
    roster = list({e.name: e for e in roster}.values())

    async def one(job: Job) -> ScoredResponse:
        async with semaphore:
            return await generate(
                job,
                client,
                config,
                roster,
                chat_model=chat_model,
                lede_model=lede_model,
            )

    # One call per distinct prompt prefix lands before the fan-out, so the
    # rest read the prompt cache instead of each paying the write.
    groups: dict[tuple[str, str, bool], list[Job]] = {}
    for job in jobs:
        key = (job.expert.name, job.arm, job.prompt.kind == "briefing_lede")
        groups.setdefault(key, []).append(job)
    firsts = [group[0] for group in groups.values()]
    rest = [job for group in groups.values() for job in group[1:]]
    rows = list(await asyncio.gather(*(one(job) for job in firsts)))
    rows += list(await asyncio.gather(*(one(job) for job in rest)))
    return rows


async def generate(
    job: Job,
    client: openai.AsyncOpenAI,
    config: ChatConfig,
    roster: list[Expert],
    *,
    chat_model: str,
    lede_model: str,
) -> ScoredResponse:
    row = ScoredResponse(
        expert=job.expert.name,
        spec_expert=job.expert.name,
        arm=job.arm,
        kind=job.prompt.kind,
        prompt_id=job.prompt.id,
        repeat=job.repeat,
        model=lede_model if job.prompt.kind == "briefing_lede" else chat_model,
        response="",
    )
    try:
        if job.prompt.kind == "briefing_lede":
            row.response, row.generation = await _with_retry(
                lambda: generate_lede(job, model=lede_model)
            )
        else:
            expert = job.expert if job.arm == "expert" else None
            turn = await _with_retry(
                lambda: generate_turn(
                    client,
                    config,
                    model=chat_model,
                    expert=expert,
                    roster=roster,
                    user_message=user_prefix(expert, roster) + job.prompt.prompt,
                )
            )
            row.response, row.truncated = turn.text, turn.truncated
            row.generation, row.tool_calls = turn.usage, turn.tool_calls
            row.rounds, row.hit_round_cap = turn.rounds, turn.hit_round_cap
            if turn.hit_round_cap:
                raise RoundCapReached(
                    f"still calling tools after {turn.rounds} rounds "
                    f"({', '.join(turn.tool_calls)})"
                )
    except RoundCapReached as exc:
        row.error = f"generation: {exc}"
        logger.warning(f"[style-eval] {job.prompt.id} {row.error}")
    except Exception as exc:
        row.error = f"generation: {type(exc).__name__}: {exc}"
        logger.warning(f"[style-eval] {job.prompt.id} {row.error}")
    return row


async def generate_lede(job: Job, *, model: str) -> tuple[str, Usage]:
    """The briefing lede the way ``narrative.compose_narrative`` makes it."""
    if job.prompt.facts is None:
        raise ValueError(f"{job.prompt.id} has no facts")
    system, facts = lede_prompt(job.expert, job.prompt.facts)
    completion = await structured_completion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": facts},
        ],
        response_model=NarrativeResponse,
        max_output_tokens=LEDE_MAX_OUTPUT_TOKENS,
        timeout_seconds=GENERATION_TIMEOUT_SECONDS,
    )
    narrative = " ".join(completion.value.narrative.split())[:_MAX_NARRATIVE_CHARS]
    usage = completion.usage
    return narrative.rstrip(), to_usage(
        model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_creation_tokens=usage.cache_creation_tokens,
        cost_usd=usage.cost_usd,
    )


def wrong_spec_rows(
    rows: list[ScoredResponse], experts: list[Expert]
) -> list[ScoredResponse]:
    """Each expert-arm response, queued to be judged against every other
    expert's spec. Same text, different spec: the judge must score it lower."""
    return [
        row.model_copy(
            update={
                "arm": "wrong_spec",
                "spec_expert": other.name,
                # The response is generated once; only the judging is new.
                "generation": None,
            }
        )
        for row in rows
        if row.arm == "expert" and row.error is None
        for other in experts
        if other.name != row.expert
    ]


async def judge_all(
    rows: list[ScoredResponse],
    experts: dict[str, Expert],
    rubric: Rubric,
    *,
    judge_model: str,
    options: RunOptions,
) -> None:
    semaphore = asyncio.Semaphore(options.concurrency)
    prompts = {p.id: p for f in load_fixtures(list(experts)) for p in f.prompts}

    async def one(row: ScoredResponse) -> None:
        if row.error is not None:
            return
        prompt = prompts[row.prompt_id]
        prompt_text = prompt.prompt or (
            prompt.facts.model_dump_json() if prompt.facts else ""
        )
        async with semaphore:
            try:
                judgement, usage = await _with_retry(
                    lambda: judge_response(
                        experts[row.spec_expert],
                        rubric,
                        kind=row.kind,
                        prompt=prompt_text,
                        response=row.response,
                        model=judge_model,
                    )
                )
                row.judgement, row.judging = judgement, usage
                row.score = response_score(judgement, rubric)
            except Exception as exc:
                row.error = f"judge: {type(exc).__name__}: {exc}"
                logger.warning(f"[style-eval] {row.prompt_id} {row.error}")

    await asyncio.gather(*(one(row) for row in rows))


def summarize(
    rows: list[ScoredResponse],
    experts: list[Expert],
    rubric: Rubric,
    *,
    fingerprint_value: str,
    chat_model: str,
    lede_model: str,
    judge_model: str,
    threshold: float | None,
) -> StyleEvalResult:
    summaries = [summarize_expert(e.name, rows) for e in experts]
    usages = [u for r in rows for u in (r.generation, r.judging) if u is not None]
    costs = [u.cost_usd for u in usages]
    return StyleEvalResult(
        run_id=f"style-{uuid.uuid4().hex[:8]}",
        ts=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        fingerprint=fingerprint_value,
        chat_model=chat_model,
        lede_model=lede_model,
        judge_model=judge_model,
        experts=summaries,
        separation=separation(rows),
        gate=None if threshold is None else evaluate_gate(summaries, threshold),
        cost_usd=round(sum(c for c in costs if c is not None), 4),
        cost_known=all(c is not None for c in costs),
        input_tokens=sum(
            u.input_tokens + u.cache_read_tokens + u.cache_creation_tokens
            for u in usages
        ),
        output_tokens=sum(u.output_tokens for u in usages),
        responses=rows,
    )


def summarize_expert(name: str, rows: list[ScoredResponse]) -> ExpertSummary:
    own = [r for r in rows if r.expert == name and r.arm == "expert"]
    scored = [r for r in own if r.score is not None]
    repeats = sorted({r.repeat for r in scored})
    return ExpertSummary(
        expert=name,
        scores=distribution([r.score for r in scored if r.score is not None]),
        by_kind={
            kind: distribution(
                [r.score for r in scored if r.kind == kind and r.score is not None]
            )
            for kind in sorted({r.kind for r in scored})
        },
        by_repeat=[
            round(
                mean(
                    [r.score for r in scored if r.repeat == i and r.score is not None]
                ),
                2,
            )
            for i in repeats
        ],
        below_60=sum(1 for r in scored if r.score is not None and r.score < 60),
        errors=sum(1 for r in own if r.error is not None),
        tool_calls=sum(len(r.tool_calls) for r in own),
        round_cap_hits=sum(1 for r in own if r.hit_round_cap),
    )


def distribution(values: list[float]) -> Distribution:
    return Distribution(
        n=len(values),
        mean=round(mean(values), 2),
        sd=round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
        min=round(min(values), 2) if values else 0.0,
        p25=round(percentile(values, 25), 2),
        median=round(percentile(values, 50), 2),
        p75=round(percentile(values, 75), 2),
        max=round(max(values), 2) if values else 0.0,
    )


def separation(rows: list[ScoredResponse]) -> Separation | None:
    """Does the judge score a response higher against its own spec than
    against a wrong one or with no persona at all? Without that gap the
    expert score measures nothing."""
    right = [r.score for r in rows if r.arm == "expert" and r.score is not None]
    wrong = [r.score for r in rows if r.arm == "wrong_spec" and r.score is not None]
    none = [r.score for r in rows if r.arm == "no_suffix" and r.score is not None]
    if not right or not (wrong or none):
        return None
    controls = [mean(c) for c in (wrong, none) if c]
    paired = paired_advantage(rows)
    return Separation(
        right_spec_mean=round(mean(right), 2),
        right_spec_sd=round(statistics.stdev(right), 2) if len(right) > 1 else 0.0,
        wrong_spec_mean=round(mean(wrong), 2) if wrong else None,
        no_suffix_mean=round(mean(none), 2) if none else None,
        gap=round(mean(right) - max(controls), 2),
        paired=paired,
        separated=None if paired is None else is_separated(paired),
    )


def paired_advantage(rows: list[ScoredResponse]) -> PairedAdvantage | None:
    """Own-spec minus the mean wrong-spec score of the SAME response. An
    unpaired mean-against-SD test called run A unseparated on a +26-point
    advantage, because a handful of unfinished turns scored near zero in
    both arms and inflated the SD."""
    own = {
        (r.expert, r.prompt_id, r.repeat): r.score
        for r in rows
        if r.arm == "expert" and r.score is not None
    }
    wrong: dict[tuple[str, str, int], list[float]] = {}
    for row in rows:
        if row.arm == "wrong_spec" and row.score is not None:
            key = (row.expert, row.prompt_id, row.repeat)
            if key in own:
                wrong.setdefault(key, []).append(row.score)
    deltas = [own[key] - mean(scores) for key, scores in wrong.items()]
    if not deltas:
        return None
    sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    wins = sum(1 for d in deltas if d > 0)
    ties = sum(1 for d in deltas if d == 0)
    return PairedAdvantage(
        n=len(deltas),
        mean=round(mean(deltas), 2),
        sem=round(sd / len(deltas) ** 0.5, 2),
        wins=wins,
        ties=ties,
        win_rate=round(wins / len(deltas), 3),
    )


def is_separated(paired: PairedAdvantage) -> bool:
    """Enough comparisons, an advantage whose 95% interval clears zero, and
    a majority of responses won on their own spec."""
    return (
        paired.n >= MIN_PAIRED_COMPARISONS
        and paired.mean - 2 * paired.sem > 0
        and paired.win_rate >= MIN_PAIRED_WIN_RATE
    )


def evaluate_gate(summaries: list[ExpertSummary], threshold: float) -> GateOutcome:
    failing = [
        s.expert
        for s in summaries
        if s.scores.n == 0
        or s.scores.mean < threshold
        or s.errors > MAX_ERRORS_PER_EXPERT
    ]
    reason = "; ".join(
        f"{s.expert}: mean {s.scores.mean} over {s.scores.n}, {s.errors} errors"
        for s in summaries
        if s.expert in failing
    )
    return GateOutcome(
        threshold=threshold, passed=not failing, failing=failing, reason=reason
    )


def print_summary(result: StyleEvalResult, out: Path) -> None:
    for s in result.experts:
        kinds = ", ".join(f"{k} {d.mean}" for k, d in s.by_kind.items())
        print(
            f"{s.expert:<8} mean {s.scores.mean:6.2f}  sd {s.scores.sd:5.2f}  "
            f"min {s.scores.min:6.2f}  n {s.scores.n:3d}  below60 {s.below_60}  "
            f"errors {s.errors}  tool calls {s.tool_calls}  "
            f"cap hits {s.round_cap_hits}  [{kinds}]"
        )
    if result.separation:
        sep = result.separation
        paired = (
            f"paired +{sep.paired.mean} ± {sep.paired.sem} over {sep.paired.n}, "
            f"own wins {sep.paired.wins} ({sep.paired.win_rate:.0%}, "
            f"{sep.paired.ties} ties), "
            if sep.paired
            else "unpaired, "
        )
        print(
            f"separation: own-spec {sep.right_spec_mean} (sd {sep.right_spec_sd}), "
            f"wrong-spec {sep.wrong_spec_mean}, no-suffix {sep.no_suffix_mean}, "
            f"gap {sep.gap}, {paired}"
            f"-> {'separated' if sep.separated else 'NOT separated'}"
        )
    known = "" if result.cost_known else " (some rows unpriced)"
    print(
        f"cost ${result.cost_usd:.4f}{known}; tokens in {result.input_tokens} "
        f"out {result.output_tokens}; results {out}"
    )
    if result.gate:
        verdict = "PASS" if result.gate.passed else f"FAIL ({result.gate.reason})"
        print(f"gate: threshold {result.gate.threshold} -> {verdict}")


def _print_dry_run(experts: list[Expert], fixtures: list[ExpertFixture]) -> None:
    by_name = {f.expert: f for f in fixtures}
    for expert in experts:
        prompts = by_name[expert.name].prompts
        ledes = sum(1 for p in prompts if p.kind == "briefing_lede")
        system, facts = lede_prompt(expert, next(p.facts for p in prompts if p.facts))
        print(
            f"{expert.name:<8} chat system prompt {len(chat_system_prompt(expert))} chars, "
            f"{len(expert_tools(expert))} tools, first-turn prefix "
            f"{len(user_prefix(expert, experts))} chars; "
            f"{len(prompts) - ledes} chat prompts + {ledes} ledes "
            f"(lede system {len(system)} chars, facts {len(facts)} chars)"
        )


async def _with_retry(call):
    try:
        return await call()
    except RoundCapReached:
        raise
    except Exception as exc:
        logger.warning(f"[style-eval] retrying after {type(exc).__name__}: {exc}")
        await asyncio.sleep(2)
        return await call()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--experts", nargs="*", help="roster names; default all")
    parser.add_argument("--kinds", nargs="*", help="prompt kinds; default all")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--model", help="override the routed chat model")
    parser.add_argument("--judge", help="override the judge model")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("expert_style_results.json"))
    parser.add_argument(
        "--gate", action="store_true", help="exit 1 below the threshold"
    )
    parser.add_argument(
        "--force", action="store_true", help="run even if the fingerprint is unchanged"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="assemble prompts, make no calls"
    )
    parser.add_argument(
        "--cross-spec",
        action="store_true",
        help="also judge each response against the other experts' specs",
    )
    parser.add_argument(
        "--control",
        type=int,
        default=0,
        help="prompts per expert to also run with no persona",
    )
    args = parser.parse_args()
    options = RunOptions(
        experts=args.experts or None,
        kinds=args.kinds or None,
        repeats=args.repeats,
        model=args.model,
        judge=args.judge,
        concurrency=args.concurrency,
        out=args.out,
        gate=args.gate,
        force=args.force,
        dry_run=args.dry_run,
        cross_spec=args.cross_spec,
        control=args.control,
    )
    _, code = asyncio.run(run(options))
    sys.exit(code)


if __name__ == "__main__":
    main()
