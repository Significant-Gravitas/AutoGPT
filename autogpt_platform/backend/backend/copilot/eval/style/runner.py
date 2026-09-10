"""Score what each expert writes against its own style spec, by hand.

    poetry run expert-style-eval --dry-run          # what changed, no calls
    poetry run expert-style-eval                    # score against baseline.json
    poetry run expert-style-eval --model <slug>     # check a proposed model
    poetry run expert-style-eval --write-baseline   # make this run the baseline

Every run costs money, so start with ``--dry-run``: it prints which of the
prompts, models, fixtures or rubric differ from the ones ``baseline.json``
was measured on, and an unchanged fingerprint means the baseline still
stands. A scored run reports each expert's distribution against the
baseline, prompt by prompt.
"""

import argparse
import asyncio
import logging
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path

import openai
from pydantic import BaseModel, Field, ValidationError, model_validator

from backend.api.features.experts.models import Expert
from backend.copilot.config import ChatConfig
from backend.copilot.eval.metrics import mean, percentile
from backend.copilot.model_normalize import normalize_model_for_transport

from .assembly import (
    DELEGATION_ENABLED,
    attach_workflows,
    chat_system_prompt,
    fingerprint,
    fingerprint_parts,
    load_baseline,
    load_fixtures,
    load_rubric,
    resolve_chat_model,
    roster_experts,
    save_baseline,
    user_prefix,
)
from .generation import chat_client, expert_tools, generate_turn
from .models import (
    Arm,
    Baseline,
    BaselineExpert,
    Distribution,
    ExpertComparison,
    ExpertFixture,
    ExpertSummary,
    PairedAdvantage,
    PromptKind,
    ReferencePrompt,
    Rubric,
    ScoredResponse,
    Separation,
    StyleEvalResult,
)
from .scorer import judge_response, response_score

logger = logging.getLogger(__name__)

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
    repeats: int = Field(default=1, gt=0)
    model: str | None = None
    judge: str | None = None
    # Zero builds a semaphore nothing can acquire; every task waits forever.
    concurrency: int = Field(default=6, gt=0)
    out: Path = Path("expert_style_results.json")
    write_baseline: bool = False
    dry_run: bool = False
    cross_spec: bool = False
    # Prompts per expert to also run as plain AutoPilot, no suffix.
    control: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _baseline_needs_the_whole_set(self) -> "RunOptions":
        """A filtered run scores part of the set and ``--kinds`` is not in the
        fingerprint, so the baseline it wrote would claim to cover everything."""
        if self.write_baseline and (self.experts or self.kinds):
            raise ValueError(
                "--write-baseline takes a full run; drop --experts/--kinds"
            )
        return self


class Job(BaseModel):
    expert: Expert
    arm: Arm
    prompt: ReferencePrompt
    repeat: int


async def run(options: RunOptions) -> StyleEvalResult | None:
    """Returns the run's result, or ``None`` on a dry run."""
    config = ChatConfig()
    fixtures = load_fixtures(options.experts)
    fixture_by_name = {f.expert: f for f in fixtures}
    experts = [
        attach_workflows(e, fixture_by_name[e.name])
        for e in roster_experts(options.experts)
    ]
    rubric, baseline = load_rubric(), load_baseline()
    routed = await resolve_chat_model(config)
    # The fingerprint reads the ROUTED names, not the transport ones, so it
    # does not move when the same run is made through the other provider.
    chat_route = options.model or routed.slug
    judge_route = options.judge or baseline.judge_model
    chat_model = normalize_model_for_transport(chat_route, config)
    judge_model = normalize_model_for_transport(judge_route, config)
    parts = fingerprint_parts(
        experts, fixtures, rubric, chat_model=chat_route, judge_model=judge_route
    )
    print(
        f"chat model {chat_model} ({routed.mode}/standard via {routed.source}"
        f"{', overridden' if options.model else ''}); "
        f"judge {judge_model}; fingerprint {fingerprint(parts)}"
    )
    print(drift(parts, baseline))
    if options.dry_run:
        _print_dry_run(experts, fixtures)
        return None

    jobs = plan_jobs(experts, fixtures, options)
    rows = await generate_all(jobs, options, config=config, chat_model=chat_model)
    by_name = {e.name: e for e in experts}
    if options.cross_spec:
        rows += wrong_spec_rows(rows, experts)
    await judge_all(rows, by_name, rubric, judge_model=judge_model, options=options)

    result = summarize(
        rows,
        experts,
        baseline,
        fingerprint_value=fingerprint(parts),
        chat_model=chat_model,
        judge_model=judge_model,
    )
    options.out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print_summary(result, options.out)
    if options.write_baseline:
        save_baseline(new_baseline(result, rows, parts))
        print(f"baseline: rewritten from this run ({len(result.experts)} experts)")
    return result


def drift(parts: dict[str, str], baseline: Baseline) -> str:
    """What the run is being asked that the baseline was not. This is the
    whole reason to spend money on a run, so it is the first line printed."""
    changed = sorted(
        {k for k, v in parts.items() if baseline.parts.get(k) != v}
        | {k for k in baseline.parts if k not in parts}
    )
    if not baseline.parts:
        return "baseline: records no fingerprint; cannot say what changed"
    if not changed:
        return (
            f"baseline: prompts, models, fixtures and rubric unchanged since "
            f"{baseline.ts}; a run would re-measure the same thing"
        )
    return f"baseline: {len(changed)} component(s) changed since " + (
        f"{baseline.ts}: {', '.join(changed)}"
    )


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
        jobs += [
            Job(expert=expert, arm="no_suffix", prompt=p, repeat=0)
            for p in prompts[: options.control]
        ]
    return jobs


async def generate_all(
    jobs: list[Job], options: RunOptions, *, config: ChatConfig, chat_model: str
) -> list[ScoredResponse]:
    semaphore = asyncio.Semaphore(options.concurrency)
    client = chat_client(config)
    roster = [job.expert for job in jobs]
    roster = list({e.name: e for e in roster}.values())

    async def one(job: Job) -> ScoredResponse:
        async with semaphore:
            return await generate(job, client, config, roster, chat_model=chat_model)

    # One call per distinct prompt prefix lands before the fan-out, so the
    # rest read the prompt cache instead of each paying the write.
    groups: dict[tuple[str, str], list[Job]] = {}
    for job in jobs:
        groups.setdefault(cache_prefix(job), []).append(job)
    firsts = [group[0] for group in groups.values()]
    rest = [job for group in groups.values() for job in group[1:]]
    rows = list(await asyncio.gather(*(one(job) for job in firsts)))
    rows += list(await asyncio.gather(*(one(job) for job in rest)))
    return rows


def cache_prefix(job: Job) -> tuple[str, str]:
    """What the job's cached prompt prefix depends on. Every no-suffix job
    shares one prompt and one tool list, whichever expert's prompts it runs,
    so keying those by expert would pay the cache write three times."""
    return (job.expert.name if job.arm == "expert" else "", job.arm)


async def generate(
    job: Job,
    client: openai.AsyncOpenAI,
    config: ChatConfig,
    roster: list[Expert],
    *,
    chat_model: str,
) -> ScoredResponse:
    row = ScoredResponse(
        expert=job.expert.name,
        spec_expert=job.expert.name,
        arm=job.arm,
        kind=job.prompt.kind,
        prompt_id=job.prompt.id,
        repeat=job.repeat,
        model=chat_model,
        response="",
    )
    try:
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
        prompt_text = prompts[row.prompt_id].prompt
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
    baseline: Baseline,
    *,
    fingerprint_value: str,
    chat_model: str,
    judge_model: str,
) -> StyleEvalResult:
    summaries = [summarize_expert(e.name, rows) for e in experts]
    usages = [u for r in rows for u in (r.generation, r.judging) if u is not None]
    costs = [u.cost_usd for u in usages]
    return StyleEvalResult(
        run_id=f"style-{uuid.uuid4().hex[:8]}",
        ts=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        fingerprint=fingerprint_value,
        chat_model=chat_model,
        judge_model=judge_model,
        experts=summaries,
        separation=separation(rows),
        comparison=[compare(s, rows, baseline) for s in summaries],
        cost_usd=round(sum(c for c in costs if c is not None), 4),
        cost_known=all(c is not None for c in costs),
        input_tokens=sum(
            u.input_tokens + u.cache_read_tokens + u.cache_creation_tokens
            for u in usages
        ),
        output_tokens=sum(u.output_tokens for u in usages),
        responses=rows,
    )


def compare(
    summary: ExpertSummary, rows: list[ScoredResponse], baseline: Baseline
) -> ExpertComparison:
    """This expert against the stored baseline, paired on the prompts both
    ran: an unpaired mean-to-mean difference mostly reports which prompts
    happened to be scored."""
    stored = next((b for b in baseline.experts if b.expert == summary.expert), None)
    if stored is None:
        return ExpertComparison(
            expert=summary.expert, mean=summary.scores.mean, baseline_mean=None
        )
    current = prompt_scores(rows, summary.expert)
    deltas = [
        score - stored.by_prompt[prompt_id]
        for prompt_id, score in current.items()
        if prompt_id in stored.by_prompt
    ]
    if not deltas:
        return ExpertComparison(
            expert=summary.expert,
            mean=summary.scores.mean,
            baseline_mean=stored.scores.mean,
        )
    sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    return ExpertComparison(
        expert=summary.expert,
        mean=summary.scores.mean,
        baseline_mean=stored.scores.mean,
        paired_delta=round(mean(deltas), 2),
        paired_sem=round(sd / len(deltas) ** 0.5, 2),
        shared_prompts=len(deltas),
    )


def new_baseline(
    result: StyleEvalResult, rows: list[ScoredResponse], parts: dict[str, str]
) -> Baseline:
    return Baseline(
        run_id=result.run_id,
        ts=result.ts,
        chat_model=result.chat_model,
        judge_model=result.judge_model,
        delegation_enabled=DELEGATION_ENABLED,
        cost_usd=result.cost_usd,
        fingerprint=result.fingerprint,
        parts=parts,
        separation=result.separation,
        experts=[
            BaselineExpert(
                expert=s.expert,
                scores=s.scores,
                by_kind=s.by_kind,
                by_prompt=prompt_scores(rows, s.expert),
            )
            for s in result.experts
        ],
    )


def prompt_scores(rows: list[ScoredResponse], expert: str) -> dict[str, float]:
    """One score per prompt, averaging repeats. The baseline and the run read
    against it both use this, so a repeated prompt is one pair rather than
    several — counting each repeat separately understates the spread."""
    by_prompt: dict[str, list[float]] = {}
    for row in rows:
        if row.expert == expert and row.arm == "expert" and row.score is not None:
            by_prompt.setdefault(row.prompt_id, []).append(row.score)
    return {prompt: round(mean(scores), 2) for prompt, scores in by_prompt.items()}


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
            f"gap {sep.gap}, {paired}-> {separation_verdict(sep)}"
        )
    for c in result.comparison:
        if c.baseline_mean is None:
            print(f"{c.expert:<8} no baseline")
        elif c.paired_delta is None:
            print(f"{c.expert:<8} mean {c.mean} vs baseline {c.baseline_mean}")
        else:
            print(
                f"{c.expert:<8} {c.paired_delta:+.2f} ± {c.paired_sem} against the "
                f"baseline over {c.shared_prompts} shared prompts "
                f"(mean {c.mean} vs {c.baseline_mean})"
            )
    known = "" if result.cost_known else " (some rows unpriced)"
    print(
        f"cost ${result.cost_usd:.4f}{known}; tokens in {result.input_tokens} "
        f"out {result.output_tokens}; results {out}"
    )


def separation_verdict(sep: Separation) -> str:
    """``separated`` is None when nothing was compared — without --cross-spec
    there is no wrong-spec arm, and reading that as NOT separated reports a
    judge failure that was never measured."""
    if sep.separated is None:
        return "not assessed"
    return "separated" if sep.separated else "NOT separated"


def _print_dry_run(experts: list[Expert], fixtures: list[ExpertFixture]) -> None:
    by_name = {f.expert: f for f in fixtures}
    for expert in experts:
        print(
            f"{expert.name:<8} chat system prompt "
            f"{len(chat_system_prompt(expert))} chars, "
            f"{len(expert_tools(expert))} tools, first-turn prefix "
            f"{len(user_prefix(expert, experts))} chars; "
            f"{len(by_name[expert.name].prompts)} prompts"
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
        "--write-baseline",
        action="store_true",
        help="store this run as the baseline later runs are read against",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="say what changed since the baseline, make no calls",
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
    try:
        options = RunOptions(
            experts=args.experts or None,
            kinds=args.kinds or None,
            repeats=args.repeats,
            model=args.model,
            judge=args.judge,
            concurrency=args.concurrency,
            out=args.out,
            write_baseline=args.write_baseline,
            dry_run=args.dry_run,
            cross_spec=args.cross_spec,
            control=args.control,
        )
    except ValidationError as invalid:
        parser.error(str(invalid))
    asyncio.run(run(options))


if __name__ == "__main__":
    main()
