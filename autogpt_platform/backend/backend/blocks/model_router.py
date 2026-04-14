"""
Model Router — Auto-selects Standard (local Ollama) vs Max (NVIDIA NIM) mode
based on task complexity analysis.

Standard mode: locally hosted small model (14B-32B) via Ollama on home PC
Max mode:      NVIDIA NIM Qwen3-Coder-480B-A35B-Instruct (build.nvidia.com)
"""

import logging
import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.llm import LlmModel
from backend.data.model import SchemaField

logger = logging.getLogger(__name__)


class ModelMode(str, Enum):
    """Execution mode for the coding agent."""

    STANDARD = "standard"  # Local Ollama model
    MAX = "max"  # NVIDIA NIM Qwen3-Coder-480B


class ComplexityLevel(str, Enum):
    """Task complexity classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Complexity signals that trigger Max mode
MAX_MODE_KEYWORDS = [
    "refactor entire",
    "architecture",
    "full stack",
    "migrate",
    "security audit",
    "performance optimization",
    "multi-service",
    "distributed",
    "design system",
    "reverse engineer",
    "large codebase",
    "complex algorithm",
    "machine learning",
    "neural network",
    "cryptography",
    "concurrency",
    "parallelism",
    "database schema",
    "api design",
    "microservices",
]

# Token count threshold above which Max mode is preferred
MAX_MODE_TOKEN_THRESHOLD = 2000

# Standard mode models (local Ollama) — user configures their preferred model
STANDARD_MODE_MODELS = [
    LlmModel.OLLAMA_LLAMA3_3,
    LlmModel.OLLAMA_LLAMA3_2,
    LlmModel.OLLAMA_LLAMA3_8B,
]

# Max mode model (NVIDIA NIM)
MAX_MODE_MODEL = LlmModel.NVIDIA_NIM_QWEN3_CODER_480B


class TaskComplexityAnalysis(BaseModel):
    """Result of task complexity analysis."""

    complexity: ComplexityLevel
    recommended_mode: ModelMode
    recommended_model: LlmModel
    token_estimate: int
    reason: str
    keyword_matches: list[str]


def analyze_task_complexity(
    task_prompt: str,
    context_length: int = 0,
    force_mode: Optional[ModelMode] = None,
    standard_model: LlmModel = LlmModel.OLLAMA_LLAMA3_3,
) -> TaskComplexityAnalysis:
    """
    Analyze a task prompt and return the recommended model and mode.

    Args:
        task_prompt: The user's task description.
        context_length: Additional context tokens (e.g., codebase index size).
        force_mode: Override auto-routing with a specific mode.
        standard_model: The local Ollama model to use in Standard mode.

    Returns:
        TaskComplexityAnalysis with recommendation details.
    """
    prompt_lower = task_prompt.lower()
    token_estimate = len(task_prompt.split()) + context_length

    # Find matching complexity keywords
    keyword_matches = [kw for kw in MAX_MODE_KEYWORDS if kw in prompt_lower]

    # Determine complexity
    if len(keyword_matches) >= 3 or token_estimate > MAX_MODE_TOKEN_THRESHOLD * 2:
        complexity = ComplexityLevel.HIGH
    elif len(keyword_matches) >= 1 or token_estimate > MAX_MODE_TOKEN_THRESHOLD:
        complexity = ComplexityLevel.MEDIUM
    else:
        complexity = ComplexityLevel.LOW

    # Determine mode
    if force_mode:
        mode = force_mode
        reason = f"Mode manually forced to {force_mode.value}."
    elif complexity == ComplexityLevel.HIGH:
        mode = ModelMode.MAX
        reason = (
            f"High complexity detected: {len(keyword_matches)} complexity keywords "
            f"and ~{token_estimate} tokens. Routing to Max mode."
        )
    elif complexity == ComplexityLevel.MEDIUM:
        mode = ModelMode.MAX
        reason = (
            f"Medium complexity: {len(keyword_matches)} keywords or "
            f"~{token_estimate} tokens exceed threshold. Routing to Max mode."
        )
    else:
        mode = ModelMode.STANDARD
        reason = (
            f"Low complexity: ~{token_estimate} tokens, no complexity keywords. "
            "Routing to Standard (local) mode."
        )

    recommended_model = MAX_MODE_MODEL if mode == ModelMode.MAX else standard_model

    return TaskComplexityAnalysis(
        complexity=complexity,
        recommended_mode=mode,
        recommended_model=recommended_model,
        token_estimate=token_estimate,
        reason=reason,
        keyword_matches=keyword_matches,
    )


class ModelRouterBlockInput(BlockSchemaInput):
    task_prompt: str = SchemaField(
        description="The task description to analyze for model routing."
    )
    context_length: int = SchemaField(
        default=0,
        description="Additional context token count (e.g., codebase index size).",
    )
    force_mode: Optional[ModelMode] = SchemaField(
        default=None,
        description="Force Standard or Max mode, bypassing auto-routing.",
    )
    standard_model: LlmModel = SchemaField(
        default=LlmModel.OLLAMA_LLAMA3_3,
        description="Local Ollama model to use in Standard mode.",
    )
    auto_routing_enabled: bool = SchemaField(
        default=True,
        description="Enable automatic model routing based on task complexity.",
    )


class ModelRouterBlockOutput(BlockSchemaOutput):
    recommended_model: str = SchemaField(
        description="The recommended LlmModel value string."
    )
    mode: str = SchemaField(description="Selected mode: 'standard' or 'max'.")
    complexity: str = SchemaField(description="Detected complexity: low/medium/high.")
    reason: str = SchemaField(description="Explanation for the routing decision.")
    token_estimate: int = SchemaField(description="Estimated token count of the task.")


class ModelRouterBlock(Block):
    """
    Analyzes task complexity and recommends the optimal model:
    - Standard mode uses a local Ollama model (14B-32B, RTX 3070 Ti).
    - Max mode uses NVIDIA NIM Qwen3-Coder-480B-A35B-Instruct.
    """

    class Input(ModelRouterBlockInput):
        pass

    class Output(ModelRouterBlockOutput):
        pass

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            description=(
                "Auto-routes tasks to Standard (local Ollama) or Max (NVIDIA NIM) "
                "mode based on complexity analysis. Supports per-task or global toggle."
            ),
            categories={BlockCategory.AI},
            input_schema=ModelRouterBlock.Input,
            output_schema=ModelRouterBlock.Output,
            test_input={
                "task_prompt": "Write a simple hello world function in Python.",
                "context_length": 0,
                "force_mode": None,
                "standard_model": LlmModel.OLLAMA_LLAMA3_3.value,
                "auto_routing_enabled": True,
            },
            test_output=[
                (
                    "recommended_model",
                    LlmModel.OLLAMA_LLAMA3_3.value,
                ),
                ("mode", ModelMode.STANDARD.value),
                ("complexity", ComplexityLevel.LOW.value),
            ],
        )

    def run(
        self, input_data: Input, *, execution_stats=None, **kwargs
    ) -> BlockOutput:
        if not input_data.auto_routing_enabled and input_data.force_mode is None:
            # Default to standard when routing is disabled and no force
            force = ModelMode.STANDARD
        else:
            force = input_data.force_mode

        analysis = analyze_task_complexity(
            task_prompt=input_data.task_prompt,
            context_length=input_data.context_length,
            force_mode=force,
            standard_model=input_data.standard_model,
        )

        logger.info(
            f"[ModelRouter] {analysis.mode.value.upper()} mode selected "
            f"({analysis.complexity.value} complexity): {analysis.reason}"
        )

        yield "recommended_model", analysis.recommended_model.value
        yield "mode", analysis.mode.value
        yield "complexity", analysis.complexity.value
        yield "reason", analysis.reason
        yield "token_estimate", analysis.token_estimate
