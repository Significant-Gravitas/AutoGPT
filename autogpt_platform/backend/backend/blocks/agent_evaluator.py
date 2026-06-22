from backend.blocks._base import (
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.agent_evaluator_helpers import (
    DEFAULT_PASS_THRESHOLD,
    MAX_SCORE,
    MIN_SCORE,
    SYSTEM_PROMPT,
    EvaluationCriterion,
    build_user_prompt,
    default_criteria,
    evaluation_response_format,
    normalize_criteria_scores,
    validate_criteria_names,
    weighted_overall,
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

_TEST_INPUT = {
    "goal": "Summarize the benefits of unit testing in one sentence.",
    "agent_output": "Unit testing catches bugs early, documents behavior, and makes refactoring safer.",
    "model": DEFAULT_LLM_MODEL,
    "credentials": TEST_CREDENTIALS_INPUT,
}

_TEST_OUTPUT = [
    ("overall_score", 85.0),
    ("passed", True),
    ("criteria_scores", list),
    ("strengths", list),
    ("weaknesses", list),
    ("suggestions", list),
    ("summary", "A solid response that achieves the goal well."),
]

_TEST_MOCK = {
    "llm_call": lambda *args, **kwargs: {
        "criteria_evaluations": [
            {"name": "Goal Achievement", "score": 90, "reasoning": "Answers it."},
            {"name": "Completeness", "score": 80, "reasoning": "Covers benefits."},
            {"name": "Relevance", "score": 85, "reasoning": "Stays on topic."},
            {"name": "Coherence", "score": 85, "reasoning": "Clear structure."},
        ],
        "strengths": ["Clear and on-topic"],
        "weaknesses": ["Could include more detail"],
        "suggestions": ["Add a concrete example"],
        "summary": "A solid response that achieves the goal well.",
    }
}


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
            test_input=_TEST_INPUT,
            test_credentials=TEST_CREDENTIALS,
            test_output=_TEST_OUTPUT,
            test_mock=_TEST_MOCK,
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

    def _build_llm_input(
        self, input_data: Input, criteria: list[EvaluationCriterion]
    ) -> AIStructuredResponseGeneratorBlock.Input:
        return AIStructuredResponseGeneratorBlock.Input(
            prompt=build_user_prompt(
                goal=input_data.goal,
                agent_output=input_data.agent_output,
                expected_output=input_data.expected_output,
                agent_design=input_data.agent_design,
                criteria=criteria,
            ),
            sys_prompt=SYSTEM_PROMPT,
            expected_format=evaluation_response_format(),
            model=input_data.model,
            credentials=input_data.credentials,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        criteria = input_data.criteria or default_criteria(
            has_reference=bool(input_data.expected_output),
            has_design=bool(input_data.agent_design),
        )

        evaluation = await self.llm_call(
            self._build_llm_input(input_data, criteria), credentials
        )

        criteria_evaluations = evaluation.get("criteria_evaluations")
        if not isinstance(criteria_evaluations, list) or not criteria_evaluations:
            yield "error", "Evaluator did not return any criterion scores."
            return

        name_error = validate_criteria_names(criteria_evaluations, criteria)
        if name_error:
            yield "error", name_error
            return

        overall_score = weighted_overall(criteria_evaluations, criteria)

        yield "overall_score", overall_score
        yield "passed", overall_score >= input_data.pass_threshold
        yield "criteria_scores", normalize_criteria_scores(
            criteria_evaluations, criteria
        )
        yield "strengths", [str(s) for s in evaluation.get("strengths", [])]
        yield "weaknesses", [str(w) for w in evaluation.get("weaknesses", [])]
        yield "suggestions", [str(s) for s in evaluation.get("suggestions", [])]
        yield "summary", str(evaluation.get("summary", ""))
