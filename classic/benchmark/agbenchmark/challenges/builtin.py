import glob
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Annotated, Any, ClassVar, Iterator, Literal, Optional

import pytest
from agent_protocol_client import AgentApi, ApiClient
from agent_protocol_client import Configuration as ClientConfig
from agent_protocol_client import Step
from colorama import Fore, Style
from openai import _load_client as get_openai_client
from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    ValidationInfo,
    field_validator,
)

from agbenchmark.agent_api_interface import download_agent_artifacts_into_folder
from agbenchmark.agent_interface import copy_challenge_artifacts_into_workspace
from agbenchmark.config import AgentBenchmarkConfig
from agbenchmark.utils.data_types import Category, DifficultyLevel, EvalResult
from agbenchmark.utils.prompts import (
    END_PROMPT,
    FEW_SHOT_EXAMPLES,
    PROMPT_MAP,
    SCORING_MAP,
)

from .base import BaseChallenge, ChallengeInfo

logger = logging.getLogger(__name__)

# Load optional categories safely
optional_categories_path = Path(__file__).parent / "optional_categories.json"
if optional_categories_path.exists():
    with open(optional_categories_path) as f:
        OPTIONAL_CATEGORIES: list[str] = json.load(f)["optional_categories"]
else:
    OPTIONAL_CATEGORIES = []
    logger.warning("optional_categories.json not found — continuing without it.")


# =====================================================================
# Builtin Challenge Specification (data.json file format)
# =====================================================================

class BuiltinChallengeSpec(BaseModel):
    eval_id: str = ""
    name: str
    task: str
    category: list[Category]
    dependencies: list[str]
    cutoff: int

    class Info(BaseModel):
        difficulty: DifficultyLevel
        description: Annotated[str, StringConstraints(pattern=r"^Tests if the agent can.*")]
        side_effects: list[str] = Field(default_factory=list)

    info: Info

    class Ground(BaseModel):
        answer: str
        should_contain: Optional[list[str]] = None
        should_not_contain: Optional[list[str]] = None
        files: list[str]
        case_sensitive: Optional[bool] = True

        class Eval(BaseModel):
            type: str
            scoring: Optional[Literal["percentage", "scale", "binary"]] = None
            template: Optional[Literal["rubric", "reference", "question", "custom"]] = None
            examples: Optional[str] = None

            @field_validator("scoring", "template")
            def validate_eval_fields(cls, value, info: ValidationInfo):
                field_name = info.field_name
                if info.data.get("type") == "llm":
                    if value is None:
                        raise ValueError(f"{field_name} must be provided when eval type is 'llm'")
                else:
                    if value is not None:
                        raise ValueError(f"{field_name} should only exist when eval type is 'llm'")
                return value

        eval: Eval

    ground: Ground
    metadata: Optional[dict[str, Any]] = None
    spec_file: Optional[Path] = Field(None, exclude=True)


# =====================================================================
# BuiltinChallenge Class — core logic
# =====================================================================

class BuiltinChallenge(BaseChallenge):
    """AGBenchmark built-in challenges (JSON-based test definitions)."""

    _spec: ClassVar[BuiltinChallengeSpec]
    CHALLENGE_LOCATION: ClassVar[str]
    ARTIFACTS_LOCATION: ClassVar[str]
    SOURCE_URI_PREFIX = "__BUILTIN__"

    # -----------------------------------------------------------------
    # Class Constructors
    # -----------------------------------------------------------------
    @classmethod
    def from_challenge_spec(cls, spec: BuiltinChallengeSpec) -> type["BuiltinChallenge"]:
        """Create a dynamic subclass for each JSON spec."""
        if not spec.spec_file:
            raise ValueError("spec.spec_file not defined")

        challenge_info = ChallengeInfo(
            eval_id=spec.eval_id,
            name=spec.name,
            task=spec.task,
            task_artifacts_dir=spec.spec_file.parent,
            category=spec.category,
            difficulty=spec.info.difficulty,
            description=spec.info.description,
            dependencies=spec.dependencies,
            reference_answer=spec.ground.answer,
            source_uri=f"{cls.SOURCE_URI_PREFIX}/{spec.spec_file.relative_to(Path(__file__).parent)}",
        )

        challenge_class_name = f"Test{challenge_info.name}"
        logger.debug(f"Creating {challenge_class_name} from {spec.spec_file}")

        return type(
            challenge_class_name,
            (BuiltinChallenge,),
            {
                "info": challenge_info,
                "_spec": spec,
                "CHALLENGE_LOCATION": str(spec.spec_file),
                "ARTIFACTS_LOCATION": str(spec.spec_file.resolve().parent),
            },
        )

    @classmethod
    def from_challenge_spec_file(cls, spec_file: Path) -> type["BuiltinChallenge"]:
        """Load challenge definition from a data.json file."""
        challenge_spec = BuiltinChallengeSpec.model_validate_json(spec_file.read_text())
        challenge_spec.spec_file = spec_file
        return cls.from_challenge_spec(challenge_spec)

    @classmethod
    def from_source_uri(cls, source_uri: str) -> type["BuiltinChallenge"]:
        """Recreate a BuiltinChallenge from its source URI."""
        if not source_uri.startswith(cls.SOURCE_URI_PREFIX):
            raise ValueError(f"Invalid source_uri for BuiltinChallenge: {source_uri}")

        path = source_uri.split("/", 1)[1]
        spec_file = Path(__file__).parent / path
        return cls.from_challenge_spec_file(spec_file)

    # -----------------------------------------------------------------
    # Core Evaluation Logic
    # -----------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_method(self, config: AgentBenchmarkConfig, request: pytest.FixtureRequest, i_attempt: int):
        """Main test entry point executed by pytest."""
        timeout = self._spec.cutoff or 60
        if request.config.getoption("--nc"):
            timeout = 100000
        elif cutoff := request.config.getoption("--cutoff"):
            timeout = int(cutoff)

        task_id, n_steps, steps = "", 0, []
        timed_out = None
        agent_task_cost = None

        try:
            async for step in self.run_challenge(config, timeout, mock=bool(request.config.getoption("--mock"))):
                if not task_id:
                    task_id = step.task_id
                n_steps += 1
                steps.append(step.model_copy())
                if step.additional_output:
                    agent_task_cost = step.additional_output.get(
                        "task_total_cost",
                        step.additional_output.get("task_cumulative_cost"),
                    )
            timed_out = False
        except TimeoutError:
            timed_out = True

        # Attach run info to the pytest item
        assert isinstance(request.node, pytest.Item)
        request.node.user_properties += [
            ("steps", steps),
            ("n_steps", n_steps),
            ("timed_out", timed_out),
            ("agent_task_cost", agent_task_cost),
        ]

        # Evaluate the agent’s output
        agent_client_config = ClientConfig(host=config.host)
        async with ApiClient(agent_client_config) as api_client:
            api_instance = AgentApi(api_client)
            eval_results = await self.evaluate_task_state(api_instance, task_id)

        if not eval_results:
            raise TimeoutError("Timed out, no results to evaluate") if timed_out else ValueError("No results to evaluate")

        request.node.user_properties += [
            ("answers", [r.result for r in eval_results] if request.config.getoption("--keep-answers") else None),
            ("scores", [r.score for r in eval_results]),
        ]

        # Pass if at least one evaluation passes
        assert any(r.passed for r in eval_results), (
            f"No passed evals: {eval_results}" if not timed_out else f"Timed out; no passed evals: {eval_results}"
        )

    # -----------------------------------------------------------------
    # Evaluation helpers
    # -----------------------------------------------------------------
    @classmethod
    async def evaluate_task_state(cls, agent: AgentApi, task_id: str) -> list[EvalResult]:
        with tempfile.TemporaryDirectory() as workspace:
            workspace = Path(workspace)
            await download_agent_artifacts_into_folder(agent, task_id, workspace)
            if cls.info.task_artifacts_dir:
                copy_challenge_artifacts_into_workspace(cls.info.task_artifacts_dir, "custom_python", workspace)
            return list(cls.evaluate_workspace_content(workspace))

    @classmethod
    def evaluate_workspace_content(cls, workspace: Path) -> Iterator[EvalResult]:
        """Check challenge outputs against the ground truth."""
        result_ground = cls._spec.ground
        outputs_for_eval = cls.get_outputs_for_eval(workspace, result_ground)

        # Text-based scoring
        if result_ground.should_contain or result_ground.should_not_contain:
            for source, content in outputs_for_eval:
                score = cls.score_result(content, result_ground)
                if score is not None:
                    yield EvalResult(result=content, result_source=str(source), score=score, passed=score > 0.9)

        # Python test or LLM scoring
        if result_ground.eval.type in ("python", "pytest"):
            for py_file, output in outputs_for_eval:
                yield EvalResult(result=output, result_source=str(py_file),
                                 score=float(not output.startswith("Error:")),
                                 passed=not output.startswith("Error:"))

        if result_ground.eval.type == "llm":
            combined_results = "\n".join(output[1] for output in outputs_for_eval)
            llm_eval = cls.score_result_with_llm(combined_results, result_ground)
            score = llm_eval / 100 if result_ground.eval.scoring == "percentage" else (
                llm_eval / 10 if result_ground.eval.scoring == "scale" else llm_eval
            )
            yield EvalResult(result=combined_results, result_source=", ".join(str(res[0]) for res in outputs_for_eval),
                             score=score, passed=score > 0.9)

    # -----------------------------------------------------------------
    # Helper Functions
    # -----------------------------------------------------------------
    @staticmethod
    def get_outputs_for_eval(workspace: Path, ground: BuiltinChallengeSpec.Ground) -> Iterator[tuple[Path, str]]:
        """Collect and read files to be evaluated."""
        for file_pattern in ground.files:
            matching_files = glob.glob(os.path.join(workspace, "*" + file_pattern)) if file_pattern.startswith(".") else [os.path.join(workspace, file_pattern)]
            for file_path in matching_files:
                file_path = Path(file_path)
                if ground.eval.type == "python":
                    result = subprocess.run([sys.executable, file_path], cwd=workspace, capture_output=True, text=True)
                    output = f"Error: {result.stderr}" if result.returncode != 0 else f"Output: {result.stdout}"
                    yield file_path.relative_to(workspace), output
                else:
                    yield file_path.relative_to(workspace), Path(file_path).read_text()

        # pytest evals
        if ground.eval.type == "pytest":
            result = subprocess.run([sys.executable, "-m", "pytest"], cwd=workspace, capture_output=True, text=True)
            output = "Error: " + (result.stderr.strip() or result.stdout) if result.returncode != 0 else "Output: " + result.stdout
            yield Path("pytest"), output

    @staticmethod
    def score_result(content: str, ground: BuiltinChallengeSpec.Ground) -> float | None:
        """Simple keyword-based scoring."""
        text = content if ground.case_sensitive else content.lower()
        if ground.should_contain:
            return 1.0 if all(w.lower() in text for w in ground.should_contain) else 0.0
        if ground.should_not_contain:
            return 1.0 if all(w.lower() not in text for w in ground.should_not_contain) else 0.0
        return None

    @classmethod
    def score_result_with_llm(cls, content: str, ground: BuiltinChallengeSpec.Ground, *, mock: bool = False) -> float:
        """Use an LLM (GPT-4) to score answers."""
        if mock:
            return 1.0
        scoring = SCORING_MAP[ground.eval.scoring]  # type: ignore
        prompt = PROMPT_MAP[ground.eval.template].format(
            task=cls._spec.task, scoring=scoring, answer=ground.answer, response=content
        )
        if ground.eval.examples:
            prompt += FEW_SHOT_EXAMPLES.format(examples=ground.eval.examples)
        prompt += END_PROMPT
        answer = get_openai_client().chat.completions.create(model="gpt-4", messages=[{"role": "system", "content": prompt}])
        return float(answer.choices[0].message.content)  # type: ignore


# =====================================================================
# Challenge Loading Utilities
# =====================================================================

def load_builtin_challenges() -> Iterator[type[BuiltinChallenge]]:
    """Find and yield all built-in challenges."""
    logger.info("Loading built-in challenges...")
    challenges_path = Path(__file__).parent
    json_files = deque(challenges_path.rglob("data.json"))
    loaded, ignored = 0, 0
    for json_file in json_files:
        if _challenge_should_be_ignored(json_file):
            ignored += 1
            continue
        yield BuiltinChallenge.from_challenge_spec_file(json_file)
        loaded += 1
    logger.info(f"Loaded {loaded} challenges, ignored {ignored}.")


def _challenge_should_be_ignored(json_file_path: Path) -> bool:
    """Skip deprecated or library challenges."""
    return any(x in json_file_path.as_posix() for x in ["challenges/deprecated", "challenges/library"])
