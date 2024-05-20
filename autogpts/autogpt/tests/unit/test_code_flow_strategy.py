import logging
import pytest

from autogpt.agents.prompt_strategies.code_flow import CodeFlowAgentPromptStrategy

from forge.components.code_flow_executor.code_flow_executor import (
    CodeFlowExecutionComponent,
)
from forge.config.ai_profile import AIProfile
from forge.config.ai_directives import AIDirectives
from forge.llm.providers import AssistantChatMessage
from forge.llm.providers.schema import CompletionModelFunction, JSONSchema


logger = logging.getLogger(__name__)
config = CodeFlowAgentPromptStrategy.default_configuration.copy(deep=True)
prompt_strategy = CodeFlowAgentPromptStrategy(config, logger)


@pytest.mark.asyncio
async def test_code_flow_build_prompt():
    commands = [
        CompletionModelFunction(
            name="web_search",
            description="Searches the web",
            parameters={
                "query": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description="The search query",
                    required=True,
                ),
                "num_results": JSONSchema(
                    type=JSONSchema.Type.INTEGER,
                    description="The number of results to return",
                    minimum=1,
                    maximum=10,
                    required=False,
                ),
            },
        ),
    ]

    ai_profile = AIProfile()
    ai_profile.ai_name = "DummyGPT"
    ai_profile.ai_goals = "A model for testing purpose"
    ai_profile.ai_role = "Help Testing"

    ai_directives = AIDirectives()
    ai_directives.resources = ["resource_1"]
    ai_directives.constraints = ["constraint_1"]
    ai_directives.best_practices = ["best_practice_1"]

    prompt = str(
        prompt_strategy.build_prompt(
            task="Figure out from file1.csv and file2.csv how much was spent on utilities",
            messages=[],
            ai_profile=ai_profile,
            ai_directives=ai_directives,
            commands=commands,
        )
    )
    assert "DummyGPT" in prompt
    assert "async def web_search(query: str, num_results: int = None)" in prompt


@pytest.mark.asyncio
async def test_code_flow_parse_response():
    response_content = """
{
"thoughts": {
  "past_action_summary": "This is the past action summary.",
  "observations": "This is the observation.",
  "text": "Some text on the AI's thoughts.",
  "reasoning": "This is the reasoning.",
  "self_criticism": "This is the self-criticism.",
  "plan": [
    "Plan 1",
    "Plan 2",
    "Plan 3"
  ],
  "speak": "This is what the AI would say."
},
"immediate_plan": "Objective[objective1] Plan[plan1] Output[out1]", 
"python_code": "async def main() -> str:\n    return 'You passed the test.'",
}
    """
    response = await CodeFlowAgentPromptStrategy(config, logger).parse_response_content(
        AssistantChatMessage(content=response_content)
    )
    assert "This is the observation." == response.thoughts.observations
    assert "This is the reasoning." == response.thoughts.reasoning

    assert "execute_code_flow" == response.use_tool.name
    assert "async def main() -> str" in response.use_tool.arguments["python_code"]
    assert (
        "Objective[objective1] Plan[plan1] Output[out1]"
        in response.use_tool.arguments["plan_text"]
    )


@pytest.mark.asyncio
async def test_code_flow_execution():
    executor = CodeFlowExecutionComponent()
    executor.available_functions = {"test_func": lambda: "You've passed the test!"}

    result = await executor.execute_code_flow(
        python_code="async def main() -> str:\n    return test_func()",
        plan_text="This is the plan text.",
    )
    assert "You've passed the test!" in result
