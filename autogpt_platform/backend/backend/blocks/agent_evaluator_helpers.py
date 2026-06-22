from typing import Any

from pydantic import BaseModel, Field, field_validator

# Per-criterion scores are expressed on a 0-100 scale: easy for the LLM to
# reason about and intuitive when surfaced to users.
MIN_SCORE = 0.0
MAX_SCORE = 100.0
DEFAULT_PASS_THRESHOLD = 70.0

SYSTEM_PROMPT = (
    "You are a rigorous, impartial evaluator of AI agent outputs. "
    "Your job is to grade an agent's output against a set of criteria and to "
    "help the agent's author make it stronger. Be critical and demanding: "
    "reserve high scores for genuinely excellent work, penalize hallucinations, "
    "omissions, and irrelevance, and never inflate scores out of politeness. "
    "Score each criterion on a 0-100 integer scale and justify every score with "
    "specific evidence from the output. Provide concrete, actionable suggestions "
    "the author can apply to improve the agent."
)


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return MIN_SCORE
    return max(MIN_SCORE, min(MAX_SCORE, score))


class EvaluationCriterion(BaseModel):
    """A single dimension the evaluator scores the agent on (the rubric entry)."""

    name: str = Field(description="Short name of the criterion, e.g. 'Accuracy'.")
    description: str = Field(
        default="",
        description="What this criterion measures and what a high score looks like.",
    )
    weight: float = Field(
        default=1.0,
        ge=0,
        description="Relative importance of this criterion in the overall score.",
    )


class CriterionEvaluation(BaseModel):
    """The judge LLM's score for a single rubric criterion."""

    name: str = ""
    score: float = MIN_SCORE
    reasoning: str = ""

    @field_validator("score", mode="before")
    @classmethod
    def _clamp(cls, value: Any) -> float:
        return clamp_score(value)


def parse_criterion_evaluations(raw: list) -> list[CriterionEvaluation]:
    """Parse the LLM's raw criterion payloads into validated models."""
    return [CriterionEvaluation.model_validate(item) for item in raw]


def default_criteria(
    *, has_reference: bool, has_design: bool
) -> list[EvaluationCriterion]:
    """Build a strong default rubric, adapting to the provided context."""
    criteria = [
        EvaluationCriterion(
            name="Goal Achievement",
            description="How well the output accomplishes the stated goal or task.",
        ),
        EvaluationCriterion(
            name="Completeness",
            description="Whether the output fully addresses every part of the request without leaving gaps.",
        ),
        EvaluationCriterion(
            name="Relevance",
            description="Whether the output stays on-topic and avoids irrelevant or extraneous content.",
        ),
        EvaluationCriterion(
            name="Coherence",
            description="Whether the output is clear, well-structured, logically consistent, and free of contradictions.",
        ),
    ]
    if has_reference:
        criteria.append(
            EvaluationCriterion(
                name="Accuracy vs Reference",
                description="How closely the output matches the provided expected/reference answer in facts and substance.",
                weight=2.0,
            )
        )
    if has_design:
        criteria.append(
            EvaluationCriterion(
                name="Design Quality",
                description="Whether the agent's design (block choices, connections, structure) is sound, efficient, and well-suited to the goal.",
            )
        )
    return criteria


def validate_criteria_names(
    evaluations: list[CriterionEvaluation], criteria: list[EvaluationCriterion]
) -> str | None:
    """Reject responses whose criterion names don't match the rubric one-to-one.

    A mismatch would silently skew the weighted score: missing criteria drop
    their weight from the denominator, unexpected/misspelled names fall back to
    weight 1.0, and duplicates get double-counted. Requiring no missing, no
    unexpected, and no duplicate names enforces exactly one score per criterion.
    Returns an error message when invalid, otherwise ``None``.
    """
    expected = {c.name.strip().lower() for c in criteria}
    returned_names = [e.name.strip().lower() for e in evaluations]
    returned = set(returned_names)
    duplicates = sorted(
        {name for name in returned_names if name and returned_names.count(name) > 1}
    )
    missing = expected - returned
    unexpected = returned - expected
    if not duplicates and not missing and not unexpected:
        return None

    problems = []
    if duplicates:
        problems.append(f"duplicate criteria: {duplicates}")
    if missing:
        problems.append(f"missing criteria: {sorted(missing)}")
    if unexpected:
        problems.append(f"unexpected criteria: {sorted(unexpected)}")
    return "Evaluator returned mismatched criterion names — " + "; ".join(problems)


def weighted_overall(
    evaluations: list[CriterionEvaluation], criteria: list[EvaluationCriterion]
) -> float:
    """Compute the weighted average so the score is deterministic, not LLM math."""
    weight_by_name = {c.name.strip().lower(): c.weight for c in criteria}
    total_weight = 0.0
    weighted_sum = 0.0
    for evaluation in evaluations:
        weight = weight_by_name.get(evaluation.name.strip().lower(), 1.0)
        weighted_sum += evaluation.score * weight
        total_weight += weight
    if total_weight == 0:
        return MIN_SCORE
    return round(weighted_sum / total_weight, 2)


def normalize_criteria_scores(
    evaluations: list[CriterionEvaluation], criteria: list[EvaluationCriterion]
) -> list[dict]:
    weight_by_name = {c.name.strip().lower(): c.weight for c in criteria}
    return [
        {
            "name": evaluation.name,
            "score": evaluation.score,
            "weight": weight_by_name.get(evaluation.name.strip().lower(), 1.0),
            "reasoning": evaluation.reasoning,
        }
        for evaluation in evaluations
    ]


def build_user_prompt(
    *,
    goal: str,
    agent_output: str,
    expected_output: str,
    agent_design: str,
    criteria: list[EvaluationCriterion],
) -> str:
    sections = [f"## GOAL / TASK\n{goal or '(not provided)'}"]
    sections.append(f"## AGENT OUTPUT TO EVALUATE\n{agent_output}")
    if expected_output:
        sections.append(
            "## EXPECTED / REFERENCE ANSWER\n"
            "Use this as ground truth when judging accuracy. The output need not "
            f"match it word-for-word, but it must be factually consistent.\n{expected_output}"
        )
    if agent_design:
        sections.append(
            "## AGENT DESIGN\n"
            "A description or JSON of the agent's graph (blocks and connections). "
            f"Judge whether the design is sound and well-suited to the goal.\n{agent_design}"
        )
    rubric = "\n".join(
        f"{i + 1}. {c.name} (weight {c.weight}): {c.description}"
        for i, c in enumerate(criteria)
    )
    sections.append(
        "## CRITERIA\nScore the output on each of the following:\n" + rubric
    )
    return "\n\n".join(sections)


def evaluation_response_format() -> dict[str, str]:
    """The JSON schema description the judge LLM must populate."""
    return {
        "criteria_evaluations": (
            "An array of objects, exactly one per criterion listed, each with: "
            "'name' (string, matching the criterion name exactly), "
            "'score' (integer 0-100), and 'reasoning' (string justifying the score)."
        ),
        "strengths": "Array of strings describing what the agent did well.",
        "weaknesses": "Array of strings describing shortcomings in the output.",
        "suggestions": "Array of concrete, actionable suggestions to improve the agent.",
        "summary": "A concise overall assessment paragraph.",
    }
