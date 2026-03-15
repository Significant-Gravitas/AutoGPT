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

with open(Path(__file__).parent / "optional_categories.json") as f:
    OPTIONAL_CATEGORIES: list[str] = json.load(f)["optional_categories"]


class BuiltinChallengeSpec(BaseModel):
    eval_id: str = ""
    name: str
    task: str
    category: list[Category]
    dependencies: list[str]
    cutoff: int

    class Info(BaseModel):
        difficulty: DifficultyLevel
        description: Annotated[
            str, StringConstraints(pattern=r"^Tests if the agent can.*")
        ]
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
            template: Optional[
                Literal["rubric", "reference", "question", "custom"]
            ] = None
            examples: Optional[str] = None

            @field_validator("scoring", "template")
            def validate_eval_fields(cls, value, info: ValidationInfo):
                field_name = info.field_name
                if "type" in info.data and info.data["type"] == "llm":
                    if value is None:
                        raise ValueError(
                            f"{field_name} must be provided when eval type is 'llm'"
                        )
                else:
                    if value is not None:
                        raise ValueError(
                            f"{field_name} should only exist when eval type is 'llm'"
                        )
                return value

        eval: Eval

    ground: Ground

    metadata: Optional[dict[str, Any]] = None
    spec_file: Path | None = Field(None, exclude=True)


class BuiltinChallenge(BaseChallenge):
    """
    Base class for AGBenchmark's built-in challenges (challenges/**/*.json).

    All of the logic is present in this class. Individual challenges are created as
    subclasses of `BuiltinChallenge` with challenge-specific values assigned to the
    ClassVars `_spec` etc.

    Dynamically constructing subclasses rather than class instances for the individual
    challenges makes them suitable for collection by Pytest, which will run their
    `test_method` like any regular test item.
    """

    _spec: ClassVar[BuiltinChallengeSpec]
    CHALLENGE_LOCATION: ClassVar[str]
    ARTIFACTS_LOCATION: ClassVar[str]

    SOURCE_URI_PREFIX = "__BUILTIN__"

    @classmethod
    def from_challenge_spec(
        cls, spec: BuiltinChallengeSpec
    ) -> type["BuiltinChallenge"]:
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
            source_uri=(
                f"__BUILTIN__/{spec.spec_file.relative_to(Path(__file__).parent)}"
            ),
        )

        challenge_class_name = f"Test{challenge_info.name}"
        logger.debug(f"Creating {challenge_class_name} from spec: {spec.spec_file}")
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
        challenge_spec = BuiltinChallengeSpec.model_validate_json(spec_file.read_text())
        challenge_spec.spec_file = spec_file
        return cls.from_challenge_spec(challenge_spec)

    @classmethod
    def from_source_uri(cls, source_uri: str) -> type["BuiltinChallenge"]:
        if not source_uri.startswith(cls.SOURCE_URI_PREFIX):
            raise ValueError(f"Invalid source_uri for BuiltinChallenge: {source_uri}")

        path = source_uri.split("/", 1)[1]
        spec_file = Path(__file__).parent / path
        return cls.from_challenge_spec_file(spec_file)

    @pytest.mark.asyncio
    async def test_method(
        self,
        config: AgentBenchmarkConfig,
        request: pytest.FixtureRequest,
        i_attempt: int,
    ) -> None:
        # if os.environ.get("HELICONE_API_KEY"):
        #     from helicone.lock import HeliconeLockManager

        #     HeliconeLockManager.write_custom_property("challenge", self.info.name)

        timeout = self._spec.cutoff or 60

        if request.config.getoption("--nc"):
            timeout = 100000
        elif cutoff := request.config.getoption("--cutoff"):
            timeout = int(cutoff)  # type: ignore

        task_id = ""
        n_steps = 0
        timed_out = None
        agent_task_cost = None
        steps: list[Step] = []
        try:
            async for step in self.run_challenge(
                config, timeout, mock=bool(request.config.getoption("--mock"))
            ):
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

        assert isinstance(request.node, pytest.Item)
        request.node.user_properties.append(("steps", steps))
        request.node.user_properties.append(("n_steps", n_steps))
        request.node.user_properties.append(("timed_out", timed_out))
        request.node.user_properties.append(("agent_task_cost", agent_task_cost))

        agent_client_config = ClientConfig(host=config.host)
        async with ApiClient(agent_client_config) as api_client:
            api_instance = AgentApi(api_client)
            eval_results = await self.evaluate_task_state(api_instance, task_id)

        if not eval_results:
            if timed_out:
                raise TimeoutError("Timed out, no results to evaluate")
            else:
                raise ValueError("No results to evaluate")

        request.node.user_properties.append(
            (
                "answers",
                [r.result for r in eval_results]
                if request.config.getoption("--keep-answers")
                else None,
            )
        )
        request.node.user_properties.append(("scores", [r.score for r in eval_results]))

        # FIXME: this allows partial failure
        assert any(r.passed for r in eval_results), (
            f"No passed evals: {eval_results}"
            if not timed_out
            else f"Timed out; no passed evals: {eval_results}"
        )

    @classmethod
    async def evaluate_task_state(
        cls, agent: AgentApi, task_id: str
    ) -> list[EvalResult]:
        with tempfile.TemporaryDirectory() as workspace:
            workspace = Path(workspace)
            await download_agent_artifacts_into_folder(agent, task_id, workspace)
            if cls.info.task_artifacts_dir:
                copy_challenge_artifacts_into_workspace(
                    cls.info.task_artifacts_dir, "custom_python", workspace
                )

            return list(cls.evaluate_workspace_content(workspace))

    @classmethod
    def evaluate_workspace_content(cls, workspace: Path) -> Iterator[EvalResult]:
        result_ground = cls._spec.ground
        outputs_for_eval = cls.get_outputs_for_eval(workspace, result_ground)

        if result_ground.should_contain or result_ground.should_not_contain:
            for source, content in outputs_for_eval:
                score = cls.score_result(content, result_ground)
                if score is not None:
                    print(f"{Fore.GREEN}Your score is:{Style.RESET_ALL}", score)
                    yield EvalResult(
                        result=content,
                        result_source=str(source),
                        score=score,
                        passed=score > 0.9,  # FIXME: arbitrary threshold
                    )

        if result_ground.eval.type in ("python", "pytest"):
            for py_file, output in outputs_for_eval:
                yield EvalResult(
                    result=output,
                    result_source=str(py_file),
                    score=float(not output.startswith("Error:")),
                    passed=not output.startswith("Error:"),
                )

        if result_ground.eval.type == "llm":
            combined_results = "\n".join(output[1] for output in outputs_for_eval)
            llm_eval = cls.score_result_with_llm(combined_results, result_ground)
            print(f"{Fore.GREEN}Your score is:{Style.RESET_ALL}", llm_eval)
            if result_ground.eval.scoring == "percentage":
                score = llm_eval / 100
            elif result_ground.eval.scoring == "scale":
                score = llm_eval / 10
            else:
                score = llm_eval

            yield EvalResult(
                result=combined_results,
                result_source=", ".join(str(res[0]) for res in outputs_for_eval),
                score=score,
                passed=score > 0.9,  # FIXME: arbitrary threshold
            )

    @staticmethod
    def get_outputs_for_eval(
        workspace: str | Path | dict[str, str], ground: BuiltinChallengeSpec.Ground
    ) -> Iterator[tuple[str | Path, str]]:
        if isinstance(workspace, dict):
            workspace = workspace["output"]

        script_dir = workspace

        for file_pattern in ground.files:
            # Check if it is a file extension
            if file_pattern.startswith("."):
                # Find all files with the given extension in the workspace
                matching_files = glob.glob(os.path.join(script_dir, "*" + file_pattern))
            else:
                # Otherwise, it is a specific file
                matching_files = [os.path.join(script_dir, file_pattern)]

            logger.debug(
                f"Files to evaluate for pattern `{file_pattern}`: {matching_files}"
            )

            for file_path in matching_files:
                relative_file_path = Path(file_path).relative_to(workspace)
                logger.debug(
                    f"Evaluating {relative_file_path} "
                    f"(eval type: {ground.eval.type})..."
                )
                if ground.eval.type == "python":
                    result = subprocess.run(
                        [sys.executable, file_path],
                        cwd=os.path.abspath(workspace),
                        capture_output=True,
                        text=True,
                    )
                    if "error" in result.stderr or result.returncode != 0:
                        yield relative_file_path, f"Error: {result.stderr}\n"
                    else:
                        yield relative_file_path, f"Output: {result.stdout}\n"
                else:
                    with open(file_path, "r") as f:
                        yield relative_file_path, f.read()
        else:
            if ground.eval.type == "pytest":
                result = subprocess.run(
                    [sys.executable, "-m", "pytest"],
                    cwd=os.path.abspath(workspace),
                    capture_output=True,
                    text=True,
                )
                logger.debug(f"EXIT CODE: {result.returncode}")
                logger.debug(f"STDOUT: {result.stdout}")
                logger.debug(f"STDERR: {result.stderr}")
                if "error" in result.stderr or result.returncode != 0:
                    yield "pytest", f"Error: {result.stderr.strip() or result.stdout}\n"
                else:
                    yield "pytest", f"Output: {result.stdout}\n"

    @staticmethod
    def score_result(content: str, ground: BuiltinChallengeSpec.Ground) -> float | None:
        print(f"{Fore.BLUE}Scoring content:{Style.RESET_ALL}", content)
        if ground.should_contain:
            for should_contain_word in ground.should_contain:
                if not ground.case_sensitive:
                    should_contain_word = should_contain_word.lower()
                    content = content.lower()
                print_content = (
                    f"{Fore.BLUE}Word that should exist{Style.RESET_ALL}"
                    f" - {should_contain_word}:"
                )
                if should_contain_word not in content:
                    print(print_content, "False")
                    return 0.0
                else:
                    print(print_content, "True")
                    return 1.0

        if ground.should_not_contain:
            for should_not_contain_word in ground.should_not_contain:
                if not ground.case_sensitive:
                    should_not_contain_word = should_not_contain_word.lower()
                    content = content.lower()
                print_content = (
                    f"{Fore.BLUE}Word that should not exist{Style.RESET_ALL}"
                    f" - {should_not_contain_word}:"
                )
                if should_not_contain_word in content:
                    print(print_content, "False")
                    return 0.0
                else:
                    print(print_content, "True")
                    return 1.0

    @classmethod
    def score_result_with_llm(
        cls, content: str, ground: BuiltinChallengeSpec.Ground, *, mock: bool = False
    ) -> float:
        if mock:
            return 1.0

        # the validation for this is done in the Eval BaseModel
        scoring = SCORING_MAP[ground.eval.scoring]  # type: ignore
        prompt = PROMPT_MAP[ground.eval.template].format(  # type: ignore
            task=cls._spec.task, scoring=scoring, answer=ground.answer, response=content
        )

        if ground.eval.examples:
            prompt += FEW_SHOT_EXAMPLES.format(examples=ground.eval.examples)

        prompt += END_PROMPT

        answer = get_openai_client().chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
            ],
        )

        return float(answer.choices[0].message.content)  # type: ignore


def load_builtin_challenges() -> Iterator[type[BuiltinChallenge]]:
    logger.info("Loading built-in challenges...")

    challenges_path = Path(__file__).parent
    logger.debug(f"Looking for challenge spec files in {challenges_path}...")

    json_files = deque(challenges_path.rglob("data.json"))

    logger.debug(f"Found {len(json_files)} built-in challenges.")

    loaded, ignored = 0, 0
    while json_files:
        # Take and remove the first element from json_files
        json_file = json_files.popleft()
        if _challenge_should_be_ignored(json_file):
            ignored += 1
            continue

        challenge = BuiltinChallenge.from_challenge_spec_file(json_file)
        logger.debug(f"Generated test for {challenge.info.name}")
        yield challenge

        loaded += 1

    logger.info(
        f"Loading built-in challenges complete: loaded {loaded}, ignored {ignored}."
    )


def _challenge_should_be_ignored(json_file_path: Path):
    return (
        "challenges/deprecated" in json_file_path.as_posix()
        or "challenges/library" in json_file_path.as_posix()
    )
