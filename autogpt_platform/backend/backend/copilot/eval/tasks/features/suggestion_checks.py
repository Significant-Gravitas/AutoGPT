"""Code checks for drafted next tasks: how many, no repeats (exact, and a
looser reworded check), reasons grounded in the account's history, valid
and fitting owners, no generic filler, briefs a worker could start from."""

import re
from collections.abc import Callable

from backend.copilot.briefing.suggestion_prompt import MAX_SUGGESTIONS
from backend.copilot.briefing.suggestions import same_title
from backend.copilot.eval.tasks.models import CheckResult
from backend.data.task import Task

from .models import FeatureChecks
from .suggestion_run import Drafting

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an and or of for to in on at by with from this that your you it is be "
    "as are was were will can into up out new next task tasks draft make get "
    "review check follow about after before week".split()
)
_GENERIC = re.compile(
    r"organi[sz]e (?:your |the )?(?:files|drive|folders)|review (?:your|all|open) "
    r"tasks|clean up|catch up on|plan (?:your|the) week|check in with (?:the )?team|"
    r"tidy up|update your notes",
    re.IGNORECASE,
)
# Two titles sharing this much of the shorter one's content words read as
# the same task reworded.
_REWORD_OVERLAP = 0.6
_MIN_BRIEF_CHARS = 60
_MAX_TITLE_CHARS = 80


def suggestion_checks(checks: FeatureChecks, drafting: Drafting) -> list[CheckResult]:
    new = drafting.created
    taken = (
        [t.title for t in drafting.board]
        + [t.title for t in drafting.live_before]
        + drafting.dismissed
    )
    results = [
        CheckResult(
            name="at_most_three",
            passed=len(drafting.returned) <= MAX_SUGGESTIONS,
            detail=f"{len(drafting.returned)} live",
        )
    ]

    def add(name: str, passed: bool, detail: str = "") -> None:
        results.append(CheckResult(name=name, passed=passed, detail=detail))

    if checks.model_called is not None:
        called = bool(drafting.calls)
        add(
            "model_called",
            called == checks.model_called,
            f"{len(drafting.calls)} calls",
        )
    if checks.min_suggestions is not None:
        add("min_suggestions", len(new) >= checks.min_suggestions, f"{len(new)} new")
    if checks.max_suggestions is not None:
        add("max_suggestions", len(new) <= checks.max_suggestions, f"{len(new)} new")
    if checks.max_new is not None:
        add("max_new", len(new) <= checks.max_new, f"{len(new)} new")
    if checks.no_repeats is not None:
        repeats = _pairs(new, taken, same_title)
        add("no_repeats", not repeats, "; ".join(repeats))
    if checks.no_rewords is not None:
        rewords = _pairs(new, taken, reworded)
        add("no_rewords", not rewords, "; ".join(rewords))
    if checks.reasons_grounded is not None:
        loose = _ungrounded(new, drafting)
        add("reasons_grounded", bool(new) and not loose, "; ".join(loose))
    if checks.owners_valid is not None:
        wrong = [
            t.title for t in new if t.owner_expert_id not in (None, *drafting.experts)
        ]
        add("owners_valid", not wrong, ", ".join(wrong))
    for keyword, owner in checks.owner_for.items():
        add(f"owner_for:{keyword}", *_owner_for(new, keyword, owner))
    if checks.no_generic is not None:
        generic = [t.title for t in new if _GENERIC.search(_all_text(t))]
        add("no_generic", not generic, ", ".join(generic))
    if checks.briefs_actionable is not None:
        weak = [t.title for t in new if not _actionable(t)]
        add("briefs_actionable", not weak, ", ".join(weak))
    if checks.has_done_when is not None:
        missing = [t.title for t in new if not (t.done_when or "").strip()]
        add("has_done_when", not missing, ", ".join(missing) or f"{len(new)} drafted")
    if checks.forbidden:
        hits = [
            f"{t.title}: {word}"
            for t in new
            for word in checks.forbidden
            if word.lower() in _all_text(t).lower()
        ]
        add("forbidden", not hits, "; ".join(hits))
    return results


def reworded(a: str, b: str) -> bool:
    """Most of the shorter title's content words appear in the other."""
    words_a, words_b = _content(a), _content(b)
    if not words_a or not words_b:
        return False
    shared = len(words_a & words_b)
    return shared / min(len(words_a), len(words_b)) >= _REWORD_OVERLAP


def _content(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def _pairs(
    new: list[Task], taken: list[str], same: Callable[[str, str], bool]
) -> list[str]:
    seen = list(taken)
    found: list[str] = []
    for task in new:
        match = next((title for title in seen if same(task.title, title)), None)
        if match is not None:
            found.append(f"'{task.title}' ~ '{match}'")
        seen.append(task.title)
    return found


def _ungrounded(new: list[Task], drafting: Drafting) -> list[str]:
    history = " ".join(
        " ".join(
            filter(None, [t.title, t.goal, t.result, t.failure_reason, t.status_text])
        )
        for t in drafting.board
    )
    known = _content(history)
    return [t.title for t in new if not (_content(t.suggested_reason or "") & known)]


def _owner_for(new: list[Task], keyword: str, owner: str) -> tuple[bool, str]:
    matching = [t for t in new if keyword.lower() in _all_text(t).lower()]
    if not matching:
        return True, "no suggestion on this; not checked"
    wrong = [
        f"{t.title} -> {t.owner_expert_id}"
        for t in matching
        if t.owner_expert_id != owner
    ]
    return not wrong, "; ".join(wrong)


def _actionable(task: Task) -> bool:
    brief = (task.description or "").strip()
    return len(brief) >= _MIN_BRIEF_CHARS and 0 < len(task.title) <= _MAX_TITLE_CHARS


def _all_text(task: Task) -> str:
    return " ".join(
        filter(None, [task.title, task.goal, task.description, task.suggested_reason])
    )
