import logging
import os
from abc import ABC, abstractmethod
from typing import ClassVar, Iterator, Literal

import pytest
import requests
from agent_protocol_client import AgentApi, Step
from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator

from agbenchmark.config import AgentBenchmarkConfig
from agbenchmark.utils.data_types import Category, EvalResult

from .base import BaseChallenge, ChallengeInfo

logger = logging.getLogger(__name__)


EvalType = Literal["string_match", "url_match", "program_html"]
WebArenaSite = Literal[
    "gitlab", "map", "reddit", "shopping", "shopping_admin", "wikipedia"
]
ReferenceAnswerType = Literal["exact_match", "fuzzy_match", "must_include"]


class WebArenaSiteInfo(BaseModel):
    base_url: str
    available: bool = True
    additional_info: str = ""
    unavailable_reason: str = ""


_git_user, _git_password = os.getenv("WEBARENA_GIT_CREDENTIALS", ":").split(":")

site_info_map: dict[WebArenaSite, WebArenaSiteInfo] = {
    "gitlab": WebArenaSiteInfo(
        base_url="http://git.junglegym.ai",
        available=bool(_git_user and _git_password),
        additional_info=(
            f"To log in to {{url}}, use the username '{_git_user}' "
            f"and password '{_git_password}'."
        ),
        unavailable_reason=(
            "WEBARENA_GIT_CREDENTIALS not set (correctly): "
            f"'{os.getenv('WEBARENA_GIT_CREDENTIALS', '')}', "
            "should be USERNAME:PASSWORD."
        ),
    ),
    "map": WebArenaSiteInfo(
        base_url="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000/"
    ),
    "reddit": WebArenaSiteInfo(base_url="http://forum.junglegym.ai"),
    "shopping": WebArenaSiteInfo(base_url="http://shop.junglegym.ai"),
    "shopping_admin": WebArenaSiteInfo(
        base_url="http://cms.junglegym.ai/admin",
        additional_info=(
            "To log in to {url}, use the username 'admin' and password 'admin1234'."
        ),
    ),
    "wikipedia": WebArenaSiteInfo(base_url="http://wiki.junglegym.ai"),
}


def get_site_info(site: WebArenaSite) -> WebArenaSiteInfo:
    if site not in site_info_map:
        raise ValueError(f"JungleGym site '{site}' unknown, cannot resolve URL")
    return site_info_map[site]


def get_site_url(site: WebArenaSite) -> str:
    return get_site_info(site).base_url


def resolve_uri(uri: str) -> str:
    """
    Resolves URIs with mock hosts, like `__WIKI__/wiki/Octopus`, with the corresponding
    JungleGym site mirror host.
    """
    segments = uri.split("__")
    if len(segments) > 2 and (site := segments[1]).lower() in site_info_map:
        return uri.replace(f"__{site}__", get_site_url(site.lower()))  # type: ignore
    return uri


class Eval(ABC):
    @abstractmethod
    def evaluate(self, string: str) -> bool:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...


class BaseStringEval(BaseModel, Eval):
    # type: ReferenceAnswerType
    pass


class ExactStringMatchEval(BaseStringEval):
    type: Literal["exact_match"] = "exact_match"
    reference_answer: str

    @property
    def description(self) -> str:
        return f"Answer must be '{self.reference_answer}'"

    def evaluate(self, string: str) -> bool:
        return string == self.reference_answer


class FuzzyStringMatchEval(BaseStringEval):
    type: Literal["fuzzy_match"] = "fuzzy_match"
    reference_answer: str

    @property
    def description(self) -> str:
        return f"Answer must contain something like '{self.reference_answer}'"

    def evaluate(self, string: str) -> bool:
        # TODO: use LLM for matching (or something else that's flexible/robust)
        return self.reference_answer.lower() in string.lower()


class MustIncludeStringEval(BaseStringEval):
    type: Literal["must_include"] = "must_include"
    reference_answer: str

    @property
    def description(self) -> str:
        return f"Answer must include '{self.reference_answer}'"

    def evaluate(self, string: str) -> bool:
        return self.reference_answer.lower() in string.lower()


StringEval = ExactStringMatchEval | FuzzyStringMatchEval | MustIncludeStringEval


class UrlMatchEval(BaseModel, Eval):
    url: str
    """Example: `"__WIKI__/wiki/Octopus"`"""

    @property
    def description(self) -> str:
        return f"Agent must navigate to '{self.url}'"

    def evaluate(self, string: str) -> bool:
        return string == resolve_uri(self.url)


class ProgramHtmlEval(BaseModel):
    url: str
    locator: str
    """JavaScript code that returns the value to check"""
    required_contents: str

    @property
    def description(self) -> str:
        return (
            f"On the webpage {self.url}, "
            f"`{self.locator}` should contain '{self.required_contents}'"
        )

    def evaluate(self, selenium_instance) -> bool:
        result = selenium_instance.execute_script(
            self.locator or "return document.body.innerHTML;"
        )
        return self.required_contents in result


_Eval = StringEval | UrlMatchEval | ProgramHtmlEval


class WebArenaChallengeSpec(BaseModel):
    task_id: int
    sites: list[WebArenaSite]
    """The sites needed to complete the task"""
    start_url: str
    """The full URL at which to start"""
    start_url_junglegym: str
    """The JungleGym site (base URL) at which to start"""
    require_login: bool
    require_reset: bool
    storage_state: str | None = None

    intent: str
    intent_template: str
    intent_template_id: int
    instantiation_dict: dict[str, str | list[str]]

    available: bool = True
    unavailable_reason: str = ""

    class EvalSet(BaseModel):
        class StringMatchEvalSet(BaseModel):
            exact_match: str | None = None
            fuzzy_match: list[str] | None = None
            must_include: list[str] | None = None

        reference_answers: StringMatchEvalSet | None = None
        """For string_match eval, a set of criteria to judge the final answer"""
        reference_answer_raw_annotation: str | None = None
        string_note: str | None = None
        annotation_note: str | None = None

        reference_url: str | None = None
        """For url_match eval, the last URL that should be visited"""
        url_note: str | None = None

        program_html: list[ProgramHtmlEval]
        """For program_html eval, a list of criteria to judge the site state by"""

        eval_types: list[EvalType]

        @field_validator("eval_types")
        def check_eval_parameters(cls, value: list[EvalType], info: ValidationInfo):
            if "string_match" in value and not info.data["reference_answers"]:
                raise ValueError("'string_match' eval_type requires reference_answers")
            if "url_match" in value and not info.data["reference_url"]:
                raise ValueError("'url_match' eval_type requires reference_url")
            if "program_html" in value and not info.data["program_html"]:
                raise ValueError(
                    "'program_html' eval_type requires at least one program_html eval"
                )
            return value

        @property
        def evaluators(self) -> list[_Eval]:
            evaluators: list[_Eval] = []
            if self.reference_answers:
                if self.reference_answers.exact_match:
                    evaluators.append(
                        ExactStringMatchEval(
                            reference_answer=self.reference_answers.exact_match
                        )
                    )
                if self.reference_answers.fuzzy_match:
                    evaluators.extend(
                        FuzzyStringMatchEval(reference_answer=a)
                        for a in self.reference_answers.fuzzy_match
                    )
                if self.reference_answers.must_include:
                    evaluators.extend(
                        MustIncludeStringEval(reference_answer=a)
                        for a in self.reference_answers.must_include
                    )
            if self.reference_url:
                evaluators.append(UrlMatchEval(url=self.reference_url))
            evaluators.extend(self.program_html)
            return evaluators

    eval: EvalSet
    """Evaluation criteria by which to judge the agent's performance"""

    @property
    def assignment_for_agent(self):
        sites = [get_site_info(s) for s in self.sites]
        nav_constraint = (
            "You are ONLY allowed to access URLs in "
            f"{' and '.join(s.base_url for s in sites)}.\n\n"
            + "\n".join(
                s.additional_info.format(url=s.base_url)
                for s in sites
                if s.additional_info
            )
        ).strip()

        return (
            f"First of all, go to {self.start_url}. "
            f"{self.intent.rstrip('.')}.\n"
            f"{nav_constraint}"
        )


class WebArenaChallenge(BaseChallenge):
    _spec: ClassVar[WebArenaChallengeSpec]

    SOURCE_URI_PREFIX = "__JUNGLEGYM__/webarena/tasks/"
    SOURCE_URI_TEMPLATE = f"{SOURCE_URI_PREFIX}{{task_id}}"

    @classmethod
    def from_source_uri(cls, source_uri: str) -> type["WebArenaChallenge"]:
        if not source_uri.startswith(cls.SOURCE_URI_PREFIX):
            raise ValueError(f"Invalid source_uri for WebArenaChallenge: {source_uri}")

        source_url = source_uri.replace(
            cls.SOURCE_URI_PREFIX,
            "https://api.junglegym.ai/get_webarena_by_task_id?task_id=",
        )
        results = requests.get(source_url).json()["data"]
        if not results:
            raise ValueError(f"Could not fetch challenge {source_uri}")
        return cls.from_challenge_spec(WebArenaChallengeSpec.model_validate(results[0]))

    @classmethod
    def from_challenge_spec(
        cls, spec: WebArenaChallengeSpec
    ) -> type["WebArenaChallenge"]:
        challenge_info = ChallengeInfo(
            eval_id=f"junglegym-webarena-{spec.task_id}",
            name=f"WebArenaTask_{spec.task_id}",
            task=spec.assignment_for_agent,
            category=[
                Category.GENERALIST,
                Category.WEB,
            ],  # TODO: make categories more specific
            reference_answer=spec.eval.reference_answer_raw_annotation,
            source_uri=cls.SOURCE_URI_TEMPLATE.format(task_id=spec.task_id),
            available=spec.available,
            unavailable_reason=spec.unavailable_reason,
        )
        return type(
            f"Test{challenge_info.name}",
            (WebArenaChallenge,),
            {
                "info": challenge_info,
                "_spec": spec,
            },
        )

    @classmethod
    def evaluate_answer(cls, answer: str) -> list[tuple[_Eval, EvalResult]]:
        results: list[tuple[_Eval, EvalResult]] = []
        for evaluator in cls._spec.eval.evaluators:
            if isinstance(evaluator, StringEval):  # string_match
                results.append(
                    (
                        evaluator,
                        EvalResult(
                            result=answer,
                            result_source="step_output",
                            score=evaluator.evaluate(answer),
                            passed=evaluator.evaluate(answer),
                        ),
                    )
                )
        return results

    @classmethod
    def evaluate_step_result(
        cls, step: Step, *, mock: bool = False
    ) -> list[tuple[_Eval, EvalResult]]:
        if mock:
            step.output = cls.info.reference_answer
        assert step.output
        eval_results = cls.evaluate_answer(step.output)
        for eval in cls._spec.eval.evaluators:
            if isinstance(eval, UrlMatchEval):
                passed = resolve_uri(eval.url) in step.output  # HACK: url_match bodge
                eval_results.append(
                    (
                        eval,
                        EvalResult(
                            result=step.output,
                            result_source="step_output",
                            score=1.0 if passed else 0.0,
                            passed=passed,
                        ),
                    )
                )
            # TODO: add support for program_html evals
        return eval_results

    @classmethod
    async def evaluate_task_state(
        cls, agent: AgentApi, task_id: str
    ) -> list[EvalResult]:
        steps: list[Step] = (await agent.list_agent_task_steps(task_id)).steps

        eval_results_per_step = [cls.evaluate_step_result(step) for step in steps]
        # Get the column aggregate (highest scored EvalResult for each Eval)
        # from the matrix of EvalResults per step.
        return [
            max(step_results_for_eval, key=lambda r: r[1].score)[1]
            for step_results_for_eval in zip(*eval_results_per_step)
        ]

    @pytest.mark.asyncio
    async def test_method(
        self,
        config: AgentBenchmarkConfig,
        request: pytest.FixtureRequest,
        i_attempt: int,
    ) -> None:
        if not self._spec.available:
            pytest.skip(self._spec.unavailable_reason)

        # if os.environ.get("HELICONE_API_KEY"):
        #     from helicone.lock import HeliconeLockManager

        #     HeliconeLockManager.write_custom_property("challenge", self.info.name)

        timeout = 120
        if request.config.getoption("--nc"):
            timeout = 100000
        elif cutoff := request.config.getoption("--cutoff"):
            timeout = int(cutoff)  # type: ignore

        assert isinstance(request.node, pytest.Item)

        n_steps = 0
        timed_out = None
        agent_task_cost = None
        steps: list[Step] = []
        eval_results_per_step: list[list[tuple[_Eval, EvalResult]]] = []
        try:
            async for step in self.run_challenge(
                config, timeout, mock=bool(request.config.getoption("--mock"))
            ):
                if not step.output:
                    logger.warn(f"Step has no output: {step}")
                    continue

                n_steps += 1
                steps.append(step)
                if step.additional_output:
                    agent_task_cost = step.additional_output.get(
                        "task_total_cost",
                        step.additional_output.get("task_cumulative_cost"),
                    )

                step_eval_results = self.evaluate_step_result(
                    step, mock=bool(request.config.getoption("--mock"))
                )
                logger.debug(f"Intermediary results: {step_eval_results}")
                eval_results_per_step.append(step_eval_results)
                if step.is_last:
                    request.node.user_properties.append(
                        (
                            "answers",
                            step.output
                            if request.config.getoption("--keep-answers")
                            else None,
                        )
                    )
            timed_out = False
        except TimeoutError:
            timed_out = True
        request.node.user_properties.append(("steps", steps))
        request.node.user_properties.append(("n_steps", n_steps))
        request.node.user_properties.append(("timed_out", timed_out))
        request.node.user_properties.append(("agent_task_cost", agent_task_cost))

        # Get the column aggregate (highest score for each Eval)
        # from the matrix of EvalResults per step.
        evals_results = [
            max(step_results_for_eval, key=lambda r: r[1].score)
            for step_results_for_eval in zip(*eval_results_per_step)
        ]

        if not evals_results:
            if timed_out:
                raise TimeoutError("Timed out, no results to evaluate")
            else:
                raise ValueError("No results to evaluate")

        request.node.user_properties.append(
            ("scores", [r[1].score for r in evals_results])
        )

        # FIXME: arbitrary threshold
        assert all(r[1].score > 0.9 for r in evals_results), (
            "Scores insufficient:\n\n"
            if not timed_out
            else "Timed out; scores insufficient:\n\n"
        ) + "\n".join(f"{repr(r[0])}\n  -> {repr(r[1])}" for r in evals_results)


def load_webarena_challenges(
    skip_unavailable: bool = True,
) -> Iterator[type[WebArenaChallenge]]:
    logger.info("Loading WebArena challenges...")

    for site, info in site_info_map.items():
        if not info.available and skip_unavailable:
            logger.warning(
                f"JungleGym site '{site}' is not available: {info.unavailable_reason} "
                "Skipping all challenges which use this site."
            )

    # response = requests.get("https://api.junglegym.ai/get_full_webarena_dataset")
    # challenge_dicts = response.json()["data"]

    # Until the full WebArena challenge set is supported, use a hand-picked selection
    import json
    from pathlib import Path

    challenge_dicts = json.loads(
        (Path(__file__).parent / "webarena_selection.json").read_bytes()
    )

    logger.debug(
        "Fetched WebArena dataset. "
        f"Constructing {len(challenge_dicts)} WebArenaChallenges..."
    )
    loaded = 0
    failed = 0
    skipped = 0
    for entry in challenge_dicts:
        try:
            challenge_spec = WebArenaChallengeSpec.model_validate(entry)
        except ValidationError as e:
            failed += 1
            logger.warning(f"Error validating WebArena challenge entry: {entry}")
            logger.warning(f"Error details: {e}")
            continue

        # Check all required sites for availability
        for site in challenge_spec.sites:
            site_info = site_info_map.get(site)
            if site_info is None:
                challenge_spec.available = False
                challenge_spec.unavailable_reason = (
                    f"WebArena task {challenge_spec.task_id} requires unknown site "
                    f"'{site}'"
                )
            elif not site_info.available:
                challenge_spec.available = False
                challenge_spec.unavailable_reason = (
                    f"WebArena task {challenge_spec.task_id} requires unavailable "
                    f"site '{site}'"
                )

        if not challenge_spec.available and skip_unavailable:
            logger.debug(f"{challenge_spec.unavailable_reason}; skipping...")
            skipped += 1
            continue

        yield WebArenaChallenge.from_challenge_spec(challenge_spec)
        loaded += 1

    logger.info(
        "Loading WebArena challenges complete: "
        f"loaded {loaded}, skipped {skipped}."
        + (f" {failed} challenges failed to load." if failed else "")
    )
