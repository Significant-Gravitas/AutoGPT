from __future__ import annotations
import collections
from datetime import datetime
import math
import time

from typing import List, Optional

import openai
from openai import Model

from autogpt.llm.base import CompletionModelInfo
from autogpt.logs import logger
from autogpt.singleton import Singleton

from llm.providers.openai_models import OPEN_AI_MODELS


class ModelApiManager:
    def __init__(self, model) -> None:
        self.model_info = OPEN_AI_MODELS.get(model)


class ApiManager(metaclass=Singleton):
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0
        self.models: Optional[list[Model]] = None
        self.known_token_limit = 10000  # TODO: defaults
        self.known_req_limit = 200
        self.tokens_on_slot = collections.defaultdict(int)
        self.slot_count_per_minute = 80
        self.number_of_slots = 500
        self.last_update_date = None
        self.avg_tokens = 0
        self.nu_of_requests = 0
        self.steps = {0.1: 3, 0.3: 8, 0.5: 15}
        self.api_by_model = {}

    def reset(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0.0
        self.models = None

    def update_rates(self, current_tokens: int, sleep: bool = True) -> None:
        logger.debug("update rates called with current tokens: %s" % (current_tokens,))
        if self.last_update_date is None:
            self.last_update_date = datetime.now()
        else:
            if (
                datetime.now() - self.last_update_date
            ).total_seconds() > self.number_of_slots * (
                60 / self.slot_count_per_minute
            ):
                self.tokens_on_slot = collections.defaultdict(int)  # reset everything

        # update current tokens rate in the last minute returns if above the limit
        ts = datetime.now().timestamp()
        modified_ts = math.floor(ts * self.slot_count_per_minute / 60)
        our_slot = modified_ts % self.number_of_slots
        changed_set = set()
        new_tokens_on_slot = self.tokens_on_slot.copy()
        max_on_last_minute = 0
        sum_on_last = 0

        for k in range(self.slot_count_per_minute - 1):
            cur = (our_slot - k) % self.number_of_slots
            changed_set.add(cur)
            new_tokens_on_slot[cur] += current_tokens
            max_on_last_minute = max(max_on_last_minute, new_tokens_on_slot[cur])

        for ind in range(self.number_of_slots):
            if ind not in changed_set:
                new_tokens_on_slot[ind] = 0

        try:
            tokens = sum(self.tokens_on_slot.values()) / len(
                self.tokens_on_slot.values()
            )
            for k, v in self.steps.items():
                if tokens > k * self.known_token_limit:
                    time.sleep((60 / self.slot_count_per_minute) * v)
                    logger.debug(f"preemptive waiting for {v} intervals")
        except ZeroDivisionError:
            self.tokens_on_slot = new_tokens_on_slot
            return

        if max_on_last_minute > self.known_token_limit and sleep:
            time.sleep((60 / self.slot_count_per_minute) * 3)  # let 3 intervals pass
            logger.debug(f"token limit exceeded, waiting for one interval")
            return self.update_rates(
                current_tokens
            )  # no real risk of infinite recursion here
        else:
            self.tokens_on_slot = new_tokens_on_slot
            return

    def update_cost(self, prompt_tokens, completion_tokens, model):
        """
        Update the total cost, prompt tokens, and completion tokens.

        Args:
        prompt_tokens (int): The number of tokens used in the prompt.
        completion_tokens (int): The number of tokens used in the completion.
        model (str): The model used for the API call.
        """
        # the .model property in API responses can contain version suffixes like -v2
        from autogpt.llm.providers.openai import OPEN_AI_MODELS

        self.update_rates(completion_tokens, sleep=False)
        self.update_rates(prompt_tokens, sleep=False)

        model = model[:-3] if model.endswith("-v2") else model
        model_info = OPEN_AI_MODELS[model]

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += prompt_tokens * model_info.prompt_token_cost / 1000
        if issubclass(type(model_info), CompletionModelInfo):
            self.total_cost += (
                completion_tokens * model_info.completion_token_cost / 1000
            )

        logger.debug(f"Total running cost: ${self.total_cost:.3f}")

    def set_total_budget(self, total_budget):
        """
        Sets the total user-defined budget for API calls.

        Args:
        total_budget (float): The total budget for API calls.
        """
        self.total_budget = total_budget

    def get_total_prompt_tokens(self):
        """
        Get the total number of prompt tokens.

        Returns:
        int: The total number of prompt tokens.
        """
        return self.total_prompt_tokens

    def get_total_completion_tokens(self):
        """
        Get the total number of completion tokens.

        Returns:
        int: The total number of completion tokens.
        """
        return self.total_completion_tokens

    def get_total_cost(self):
        """
        Get the total cost of API calls.

        Returns:
        float: The total cost of API calls.
        """
        return self.total_cost

    def get_total_budget(self):
        """
        Get the total user-defined budget for API calls.

        Returns:
        float: The total budget for API calls.
        """
        return self.total_budget

    def get_models(self, **openai_credentials) -> List[Model]:
        """
        Get list of available GPT models.

        Returns:
        list: List of available GPT models.

        """
        if self.models is None:
            all_models = openai.Model.list(**openai_credentials)["data"]
            self.models = [model for model in all_models if "gpt" in model["id"]]
            self.api_by_model = {m["id"]: ModelApiManager(m["id"]) for m in self.models}

        return self.models

    def model(self, model):
        return self.api_by_model.get(model)
