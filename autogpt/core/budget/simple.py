import logging
import math
from typing import Any, Dict

from autogpt.core.budget.base import BudgetManager, ResourceBudget
from autogpt.core.configuration.base import Configuration
from autogpt.core.model.base import ModelResponse


class OpenAIBudget(ResourceBudget):
    configuration_defaults = {
        "openai_budget": {
            "total_budget": math.inf,
            "total_cost": 0,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
            "graceful_shutdown_threshold": 0.005,
            "warning_threshold": 0.01,
        }
    }

    def __init__(self, llm_configuration: dict):
        # Parent should be able to pass subset of the config by reference here.
        self._configuration = llm_configuration

    @property
    def total_budget(self) -> float:
        return self._configuration["total_budget"]

    @property
    def total_cost(self) -> float:
        return self._configuration["total_cost"]

    @property
    def remaining_budget(self) -> float:
        return self.total_budget - self.total_cost

    @property
    def usage(self) -> Dict[str, int]:
        return self._configuration["usage"]

    def update_usage_and_cost(self, llm_response: ModelResponse) -> None:
        model_info = llm_response.model_info
        self._configuration["usage"]["prompt_tokens"] += llm_response.prompt_tokens_used
        self._configuration["usage"][
            "completion_tokens"
        ] += llm_response.completion_tokens_used
        self._configuration["total_cost"] += (
            llm_response.prompt_tokens_used * model_info.prompt_token_cost
            + llm_response.completion_tokens_used * model_info.completion_token_cost
        ) / 1000

    def get_resource_budget_prompt(self) -> str:
        if self.total_budget == math.inf:
            return ""

        graceful_shutdown_threshold = self._configuration["graceful_shutdown_threshold"]
        warning_threshold = self._configuration["warning_threshold"]

        resource_prompt = f"Your remaining API budget is ${self.remaining_budget:.3f}"
        if self.remaining_budget <= 0:
            resource_prompt += " BUDGET EXCEEDED! SHUT DOWN!\n\n"
        elif self.remaining_budget < graceful_shutdown_threshold:
            resource_prompt += " Budget very nearly exceeded! Shut down gracefully!\n\n"
        elif self.remaining_budget < warning_threshold:
            resource_prompt += " Budget nearly exceeded. Finish up.\n\n"

        return resource_prompt

    def __repr__(self):
        return f"LLMBudget(total_budget={self.total_budget}, total_cost={self.total_cost}, remaining_budget={self.remaining_budget}, usage={self.usage})"


class SimpleBudgetManager(BudgetManager):
    configuration_defaults = {
        "budget_manager": {
            # TODO: this should be more dynamically configurable in a later
            #  iteration so that users can budget multiple resources.
            "openai_budget": OpenAIBudget.configuration_defaults["openai_budget"],
        }
    }

    def __init__(
        self,
        configuration: Configuration,
        logger: logging.Logger,
    ):
        self._configuration = configuration.budget_manager
        self._resources: dict[str, ResourceBudget] = {
            "openai_budget": OpenAIBudget(self._configuration["openai_budget"])
        }

    def list_resources(self) -> list[str]:
        return list(self._configuration)

    def get_total_budget(self, resource_name: str) -> float:
        return self._resources[resource_name].total_budget

    def get_total_cost(self, resource_name: str) -> float:
        return self._resources[resource_name].total_cost

    def get_remaining_budget(self, resource_name: str) -> float:
        return self._resources[resource_name].remaining_budget

    def get_resource_usage(self, resource_name: str) -> Any:
        return self._resources[resource_name].usage

    def update_resource_usage_and_cost(
        self,
        resource_name: str,
        *args,
        **kwargs,
    ) -> None:
        self._resources[resource_name].update_usage_and_cost(*args, **kwargs)

    def get_resource_budget_prompt(self, resource_name: str) -> str:
        return self._resources[resource_name].get_resource_budget_prompt()

    def __repr__(self) -> str:
        return f"SimpleBudgetManager(resource_budgets={list(self._resources)})"
