import openai
from config import Config
from token_counter import count_message_tokens
import llm_utils

cfg = Config()
openai.api_key = cfg.openai_api_key
print_total_cost = True

# Define the cost per token for each model
COSTS = {
    "gpt-3.5-turbo": {
        "prompt": 0.002,
        "completion": 0.002
    },
    "gpt-4-0314": {
        "prompt": 0.03,
        "completion": 0.06
    }
}

# TODO: route all API's through this manager, so we can keep track of
# the cost of all API calls, not just chat completions
class ApiManager:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0

    def create_chat_completion(self, messages, model=None, temperature=None, max_tokens=None) -> str:
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
        response = llm_utils.create_chat_completion(messages, model, temperature, max_tokens)
        prompt_tokens = count_message_tokens(messages, model)
        completion_tokens = count_message_tokens([{"role": "assistant", "content": response}], model)
        self.update_cost(prompt_tokens, completion_tokens, model)
        return response

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
        self.total_cost += (prompt_tokens * COSTS[model]["prompt"] +
                            completion_tokens * COSTS[model]["completion"]) / 1000
        if print_total_cost:
            print(f"Total running cost: ${self.total_cost:.3f}")

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

api_manager = ApiManager()