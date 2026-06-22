from typing import Any

from pydantic import BaseModel, Field

from backend.blocks._base import (
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.llm import (
    DEFAULT_LLM_MODEL,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    AIBlockBase,
    AICredentials,
    AICredentialsField,
    AIStructuredResponseGeneratorBlock,
    LlmModel,
)
from backend.data.model import APIKeyCredentials, SchemaField

# Per-criterion scores are expressed on a 0-100 scale: easy for the LLM to
# reason about and intuitive when surfaced to users.
MIN_SCORE = 0.0
MAX_SCORE = 100.0
DEFAULT_PASS_THRESHOLD = 70.0


class EvaluationCriterion(BaseModel):
    """A single dimension the evaluator scores the agent on."""

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


def _default_criteria(
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


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return MIN_SCORE
    return max(MIN_SCORE, min(MAX_SCORE, score))


def _weighted_overall(
    criteria_evaluations: list[dict], criteria: list[EvaluationCriterion]
) -> float:
    """Compute the weighted average so the score is deterministic, not LLM math."""
    weight_by_name = {c.name.strip().lower(): c.weight for c in criteria}
    total_weight = 0.0
    weighted_sum = 0.0
    for evaluation in criteria_evaluations:
        name = str(evaluation.get("name", "")).strip().lower()
        weight = weight_by_name.get(name, 1.0)
        weighted_sum += _clamp_score(evaluation.get("score")) * weight
        total_weight += weight
    if total_weight == 0:
        return MIN_SCORE
    return round(weighted_sum / total_weight, 2)


def _normalize_criteria_scores(
    criteria_evaluations: list[dict], criteria: list[EvaluationCriterion]
) -> list[dict]:
    weight_by_name = {c.name.strip().lower(): c.weight for c in criteria}
    return [
        {
            "name": str(evaluation.get("name", "")),
            "score": _clamp_score(evaluation.get("score")),
            "weight": weight_by_name.get(
                str(evaluation.get("name", "")).strip().lower(), 1.0
            ),
            "reasoning": str(evaluation.get("reasoning", "")),
        }
        for evaluation in criteria_evaluations
    ]


_SYSTEM_PROMPT = (
    "You are a rigorous, impartial evaluator of AI agent outputs. "
    "Your job is to grade an agent's output against a set of criteria and to "
    "help the agent's author make it stronger. Be critical and demanding: "
    "reserve high scores for genuinely excellent work, penalize hallucinations, "
    "omissions, and irrelevance, and never inflate scores out of politeness. "
    "Score each criterion on a 0-100 integer scale and justify every score with "
    "specific evidence from the output. Provide concrete, actionable suggestions "
    "the author can apply to improve the agent."
)


def _build_user_prompt(
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


class AIAgentEvaluatorBlock(AIBlockBase):
    """
    A strong LLM-as-judge evaluator for AI agent outputs.

    Grades an agent's output against a configurable rubric, optionally comparing
    it to an expected/reference answer and assessing the agent's design. Returns
    a weighted overall score, per-criterion scores, and actionable suggestions to
    help authors build stronger agents.
    """

    class Input(BlockSchemaInput):
        goal: str = SchemaField(
            description="The goal or task the agent was supposed to accomplish.",
            placeholder="E.g., 'Summarize the article in 3 bullet points'",
            advanced=False,
        )
        agent_output: str = SchemaField(
            description="The actual output produced by the agent that should be evaluated.",
            placeholder="Paste the agent's output here",
            advanced=False,
        )
        expected_output: str = SchemaField(
            default="",
            description="(Optional) The expected/reference answer to compare the output against.",
            advanced=False,
        )
        agent_design: str = SchemaField(
            default="",
            description="(Optional) A description or JSON of the agent's graph structure to assess design quality.",
        )
        criteria: list[EvaluationCriterion] = SchemaField(
            default_factory=list,
            description="(Optional) Custom rubric. If empty, a strong default rubric is used, adapted to the inputs provided.",
            advanced=False,
        )
        pass_threshold: float = SchemaField(
            default=DEFAULT_PASS_THRESHOLD,
            ge=MIN_SCORE,
            le=MAX_SCORE,
            description="Minimum overall score (0-100) required for the output to be considered passing.",
        )
        model: LlmModel = SchemaField(
            title="LLM Model",
            default=DEFAULT_LLM_MODEL,
            description="The language model to use as the evaluator (judge).",
            advanced=False,
        )
        credentials: AICredentials = AICredentialsField()

    class Output(BlockSchemaOutput):
        overall_score: float = SchemaField(
            description="Weighted overall score from 0 to 100."
        )
        passed: bool = SchemaField(
            description="Whether the overall score meets the pass threshold."
        )
        criteria_scores: list[dict] = SchemaField(
            description="Per-criterion results: name, score, weight, and reasoning."
        )
        strengths: list[str] = SchemaField(description="What the agent did well.")
        weaknesses: list[str] = SchemaField(
            description="Shortcomings and problems found in the output."
        )
        suggestions: list[str] = SchemaField(
            description="Concrete, actionable suggestions to make the agent stronger."
        )
        summary: str = SchemaField(
            description="A concise overall assessment of the agent's output."
        )
        error: str = SchemaField(
            description="Error message if the evaluation could not be completed."
        )

    def __init__(self):
        super().__init__(
            id="b9f1c2a4-7e3d-4c8a-9d2b-1f6a5c0e8d34",
            description="Uses an LLM as a strong judge to evaluate an agent's output against a rubric, expected answers, and design, returning scores and improvement suggestions.",
            categories={BlockCategory.AI, BlockCategory.LOGIC},
            input_schema=AIAgentEvaluatorBlock.Input,
            output_schema=AIAgentEvaluatorBlock.Output,
            test_input={
                "goal": "Summarize the benefits of unit testing in one sentence.",
                "agent_output": "Unit testing catches bugs early, documents behavior, and makes refactoring safer.",
                "model": DEFAULT_LLM_MODEL,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("overall_score", 85.0),
                ("passed", True),
                ("criteria_scores", list),
                ("strengths", list),
                ("weaknesses", list),
                ("suggestions", list),
                ("summary", "A solid response that achieves the goal well."),
            ],
            test_mock={
                "llm_call": lambda *args, **kwargs: {
                    "criteria_evaluations": [
                        {
                            "name": "Goal Achievement",
                            "score": 90,
                            "reasoning": "Directly answers the request.",
                        },
                        {
                            "name": "Completeness",
                            "score": 80,
                            "reasoning": "Covers the main benefits.",
                        },
                        {
                            "name": "Relevance",
                            "score": 85,
                            "reasoning": "Stays on topic.",
                        },
                        {
                            "name": "Coherence",
                            "score": 85,
                            "reasoning": "Clear and well-structured.",
                        },
                    ],
                    "strengths": ["Clear and on-topic"],
                    "weaknesses": ["Could include more detail"],
                    "suggestions": ["Add a concrete example"],
                    "summary": "A solid response that achieves the goal well.",
                }
            },
        )

    async def llm_call(
        self,
        input_data: AIStructuredResponseGeneratorBlock.Input,
        credentials: APIKeyCredentials,
    ) -> dict:
        """Wraps the structured-response block so tests can mock the judge call."""
        block = AIStructuredResponseGeneratorBlock()
        response = await block.run_once(input_data, "response", credentials=credentials)
        self.merge_llm_stats(block)
        return response

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        criteria = input_data.criteria or _default_criteria(
            has_reference=bool(input_data.expected_output),
            has_design=bool(input_data.agent_design),
        )

        expected_format = {
            "criteria_evaluations": (
                "An array of objects, one per criterion listed, each with: "
                "'name' (string, matching the criterion name), "
                "'score' (integer 0-100), and 'reasoning' (string justifying the score)."
            ),
            "strengths": "Array of strings describing what the agent did well.",
            "weaknesses": "Array of strings describing shortcomings in the output.",
            "suggestions": "Array of concrete, actionable suggestions to improve the agent.",
            "summary": "A concise overall assessment paragraph.",
        }

        llm_input = AIStructuredResponseGeneratorBlock.Input(
            prompt=_build_user_prompt(
                goal=input_data.goal,
                agent_output=input_data.agent_output,
                expected_output=input_data.expected_output,
                agent_design=input_data.agent_design,
                criteria=criteria,
            ),
            sys_prompt=_SYSTEM_PROMPT,
            expected_format=expected_format,
            model=input_data.model,
            credentials=input_data.credentials,
        )

        evaluation = await self.llm_call(llm_input, credentials)

        criteria_evaluations = evaluation.get("criteria_evaluations")
        if not isinstance(criteria_evaluations, list) or not criteria_evaluations:
            yield "error", "Evaluator did not return any criterion scores."
            return

        overall_score = _weighted_overall(criteria_evaluations, criteria)

        yield "overall_score", overall_score
        yield "passed", overall_score >= input_data.pass_threshold
        yield "criteria_scores", _normalize_criteria_scores(
            criteria_evaluations, criteria
        )
        yield "strengths", [str(s) for s in evaluation.get("strengths", [])]
        yield "weaknesses", [str(w) for w in evaluation.get("weaknesses", [])]
        yield "suggestions", [str(s) for s in evaluation.get("suggestions", [])]
        yield "summary", str(evaluation.get("summary", ""))
