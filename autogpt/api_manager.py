from typing import List

import openai

from autogpt.config import Config

cfg = Config()
openai.api_key = cfg.openai_api_key
print_total_cost = True

# Define the cost per thousand tokens for each model
# TODO: make this a json file that we can update separate from the code
COSTS = {
    "gpt-3.5-turbo": {"prompt": 0.002, "completion": 0.002},
    "gpt-3.5-turbo-0301": {"prompt": 0.002, "completion": 0.002},
    "gpt-4-0314": {"prompt": 0.03, "completion": 0.06},
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "text-embedding-ada-002": {"prompt_tokens": 0.0004}
}


# TODO: route all API's through this manager, so we can keep track of
# the cost of all API calls, not just chat completions
class ApiManager:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0

    def create_chat_completion(
        self,
        messages: list,  # type: ignore
        model: str | None = None,
        temperature: float = cfg.temperature,
        max_tokens: int | None = None,
        deployment_id=None,
    ) -> str:
        """
        Create a chat completion and update the cost.

        Args:
        messages (list): The list of messages to send to the API.
        model (str): The model to use for the API call.
        temperature (float): The temperature to use for the API call.
        max_tokens (int): The maximum number of tokens for the API call.

        Returns:
        str: The AI's response.
        """
        if deployment_id is not None:
            response = openai.ChatCompletion.create(
                deployment_id=deployment_id,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        self.update_cost(prompt_tokens, completion_tokens, model)
        return response

    def embedding_create(
        self,
        text_list: List[str],
        model: str = "text-embedding-ada-002",
    ) -> List[float]:
        """
        Create an embedding for the given input text using the specified model.

        Args:
        text_list (List[str]): Input text for which the embedding is to be created.
        model (str, optional): The model to use for generating the embedding.

        Returns:
        List[float]: The generated embedding as a list of float values.
        """
        if cfg.use_azure:
            response = openai.Embedding.create(
                input=text_list,
                engine=cfg.get_azure_deployment_id_for_model(
                    model
                ),
            )
        else:
            response = openai.Embedding.create(
                input=text_list, model=model
            )

        self.update_cost(response.usage.prompt_tokens, 0, model)
        return response["data"][0]["embedding"]

    def update_cost(self, prompt_tokens, completion_tokens, model):
        """
        Update the total cost, prompt tokens, and completion tokens.

        Args:
        prompt_tokens (int): The number of tokens used in the prompt.
        completion_tokens (int): The number of tokens used in the completion.
        model (str): The model used for the API call.
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += (
            prompt_tokens * COSTS[model]["prompt"]
            + completion_tokens * COSTS[model]["completion"]
        ) / 1000
        if print_total_cost:
            print(f"Total running cost: ${self.total_cost:.3f}")

    def set_total_budget(self, total_budget):
        """
        Sets the total user-defined budget for API calls.

        Args:
        prompt_tokens (int): The number of tokens used in the prompt.
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


api_manager = ApiManager()
