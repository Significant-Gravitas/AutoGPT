"""Pydantic models for the direct benchmark harness."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

# Type aliases
StrategyName = Literal[
    "one_shot",
    "rewoo",
    "plan_execute",
    "reflexion",
    "tree_of_thoughts",
    "lats",
    "multi_agent_debate",
]
ReasoningEffort = Literal["low", "medium", "high"]

STRATEGIES: list[StrategyName] = [
    "one_shot",
    "rewoo",
    "plan_execute",
    "reflexion",
    "tree_of_thoughts",
    "lats",
    "multi_agent_debate",
]


class ModelConfig(BaseModel):
    """LLM model configuration."""

    name: str
    smart_llm: str
    fast_llm: str
    thinking_budget_tokens: Optional[int] = None
    reasoning_effort: Optional[ReasoningEffort] = None

    def __str__(self) -> str:
        parts = [f"smart={self.smart_llm}", f"fast={self.fast_llm}"]
        if self.thinking_budget_tokens:
            parts.append(f"thinking={self.thinking_budget_tokens}")
        if self.reasoning_effort:
            parts.append(f"reasoning={self.reasoning_effort}")
        return f"{self.name} ({', '.join(parts)})"


# Preset model configurations
MODEL_PRESETS: dict[str, ModelConfig] = {
    # Claude configurations
    "claude": ModelConfig(
        name="claude",
        smart_llm="claude-sonnet-4-20250514",
        fast_llm="claude-3-5-haiku-20241022",
    ),
    "claude-smart": ModelConfig(
        name="claude-smart",
        smart_llm="claude-sonnet-4-20250514",
        fast_llm="claude-sonnet-4-20250514",
    ),
    "claude-fast": ModelConfig(
        name="claude-fast",
        smart_llm="claude-3-5-haiku-20241022",
        fast_llm="claude-3-5-haiku-20241022",
    ),
    "claude-opus": ModelConfig(
        name="claude-opus",
        smart_llm="claude-opus-4-5-20251101",
        fast_llm="claude-sonnet-4-20250514",
    ),
    "claude-opus-only": ModelConfig(
        name="claude-opus-only",
        smart_llm="claude-opus-4-5-20251101",
        fast_llm="claude-opus-4-5-20251101",
    ),
    # OpenAI configurations
    "openai": ModelConfig(
        name="openai",
        smart_llm="gpt-4o",
        fast_llm="gpt-4o-mini",
    ),
    "openai-smart": ModelConfig(
        name="openai-smart",
        smart_llm="gpt-4o",
        fast_llm="gpt-4o",
    ),
    "openai-fast": ModelConfig(
        name="openai-fast",
        smart_llm="gpt-4o-mini",
        fast_llm="gpt-4o-mini",
    ),
    "gpt5": ModelConfig(
        name="gpt5",
        smart_llm="gpt-5",
        fast_llm="gpt-4o",
    ),
    "gpt5-only": ModelConfig(
        name="gpt5-only",
        smart_llm="gpt-5",
        fast_llm="gpt-5",
    ),
    "o1": ModelConfig(
        name="o1",
        smart_llm="o1",
        fast_llm="gpt-4o-mini",
    ),
    "o1-mini": ModelConfig(
        name="o1-mini",
        smart_llm="o1-mini",
        fast_llm="gpt-4o-mini",
    ),
    # Claude extended thinking configurations
    "claude-thinking-10k": ModelConfig(
        name="claude-thinking-10k",
        smart_llm="claude-sonnet-4-20250514",
        fast_llm="claude-3-5-haiku-20241022",
        thinking_budget_tokens=10000,
    ),
    "claude-thinking-25k": ModelConfig(
        name="claude-thinking-25k",
        smart_llm="claude-sonnet-4-20250514",
        fast_llm="claude-3-5-haiku-20241022",
        thinking_budget_tokens=25000,
    ),
    "claude-thinking-50k": ModelConfig(
        name="claude-thinking-50k",
        smart_llm="claude-sonnet-4-20250514",
        fast_llm="claude-3-5-haiku-20241022",
        thinking_budget_tokens=50000,
    ),
    "claude-opus-thinking": ModelConfig(
        name="claude-opus-thinking",
        smart_llm="claude-opus-4-5-20251101",
        fast_llm="claude-sonnet-4-20250514",
        thinking_budget_tokens=25000,
    ),
    "claude-opus-thinking-50k": ModelConfig(
        name="claude-opus-thinking-50k",
        smart_llm="claude-opus-4-5-20251101",
        fast_llm="claude-sonnet-4-20250514",
        thinking_budget_tokens=50000,
    ),
    # OpenAI reasoning effort configurations
    "o1-low": ModelConfig(
        name="o1-low",
        smart_llm="o1",
        fast_llm="gpt-4o-mini",
        reasoning_effort="low",
    ),
    "o1-medium": ModelConfig(
        name="o1-medium",
        smart_llm="o1",
        fast_llm="gpt-4o-mini",
        reasoning_effort="medium",
    ),
    "o1-high": ModelConfig(
        name="o1-high",
        smart_llm="o1",
        fast_llm="gpt-4o-mini",
        reasoning_effort="high",
    ),
    "o3-low": ModelConfig(
        name="o3-low",
        smart_llm="o3",
        fast_llm="gpt-4o-mini",
        reasoning_effort="low",
    ),
    "o3-medium": ModelConfig(
        name="o3-medium",
        smart_llm="o3",
        fast_llm="gpt-4o-mini",
        reasoning_effort="medium",
    ),
    "o3-high": ModelConfig(
        name="o3-high",
        smart_llm="o3",
        fast_llm="gpt-4o-mini",
        reasoning_effort="high",
    ),
    "gpt5-low": ModelConfig(
        name="gpt5-low",
        smart_llm="gpt-5",
        fast_llm="gpt-4o",
        reasoning_effort="low",
    ),
    "gpt5-medium": ModelConfig(
        name="gpt5-medium",
        smart_llm="gpt-5",
        fast_llm="gpt-4o",
        reasoning_effort="medium",
    ),
    "gpt5-high": ModelConfig(
        name="gpt5-high",
        smart_llm="gpt-5",
        fast_llm="gpt-4o",
        reasoning_effort="high",
    ),
}


class BenchmarkConfig(BaseModel):
    """Configuration for a single benchmark run (strategy + model)."""

    strategy: StrategyName
    model: ModelConfig
    max_steps: int = 50
    timeout_seconds: int = 900

    @property
    def config_name(self) -> str:
        """Return unique name for this configuration."""
        return f"{self.strategy}/{self.model.name}"


class HarnessConfig(BaseModel):
    """Overall harness configuration."""

    workspace_root: Path
    challenges_dir: Path
    reports_dir: Path = Field(default_factory=lambda: Path("./reports"))
    categories: Optional[list[str]] = None
    skip_categories: Optional[list[str]] = None
    test_names: Optional[list[str]] = None
    max_parallel: int = 4
    configs: list[BenchmarkConfig] = Field(default_factory=list)

    # Execution options
    attempts: int = 1  # Number of times to run each challenge
    no_cutoff: bool = False  # Disable time limit
    no_dep: bool = False  # Ignore challenge dependencies

    # Challenge selection modes
    maintain: bool = False  # Run only regression tests (previously beaten)
    improve: bool = False  # Run only non-regression tests (not consistently beaten)
    explore: bool = False  # Run only never-beaten challenges

    # Debug options
    keep_answers: bool = False  # Keep answer files for debugging
    debug: bool = False  # Enable debug output

    # Resume options
    fresh: bool = False  # Clear state and start fresh (don't resume)
    retry_failures: bool = False  # Reset and re-run only failed challenges
    reset_strategies: Optional[list[str]] = None  # Reset specific strategies
    reset_models: Optional[list[str]] = None  # Reset specific models
    reset_challenges: Optional[list[str]] = None  # Reset specific challenges

    # External benchmark options
    external_benchmark: Optional[str] = None  # gaia, swe-bench, agent-bench
    benchmark_split: str = "validation"  # train, validation, test
    benchmark_subset: Optional[str] = None  # Difficulty level, repo name, etc.
    benchmark_limit: Optional[int] = None  # Max challenges to load
    benchmark_cache_dir: Optional[Path] = None  # Cache directory for downloads

    model_config = {"arbitrary_types_allowed": True}


class Challenge(BaseModel):
    """Loaded challenge data from data.json."""

    name: str
    task: str
    category: list[str]
    difficulty: str
    cutoff: int  # timeout in seconds
    ground_truth: dict
    artifacts_dir: Path
    source_path: Path

    model_config = {"arbitrary_types_allowed": True}


class StepResult(BaseModel):
    """Result of a single agent step."""

    step_num: int
    tool_name: str
    tool_args: dict
    result: str
    is_error: bool
    cumulative_cost: float


class ChallengeResult(BaseModel):
    """Result of running a challenge."""

    challenge_name: str
    config_name: str
    attempt: int = 1  # Which attempt number (1-indexed)
    success: bool = False
    score: float = 0.0
    steps: list[StepResult] = Field(default_factory=list)
    n_steps: int = 0
    run_time_seconds: float = 0.0
    cost: float = 0.0
    timed_out: bool = False
    error_message: Optional[str] = None
    output_files: dict[str, str] = Field(default_factory=dict)


class ExecutionProgress(BaseModel):
    """Progress update for a running execution."""

    config_name: str
    challenge_name: str
    status: Literal["starting", "running", "completed", "failed"]
    step_num: Optional[int] = None
    result: Optional[ChallengeResult] = None
