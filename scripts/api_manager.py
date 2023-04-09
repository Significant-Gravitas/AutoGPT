import openai
from config import Config
from token_counter import count_message_tokens
import llm_utils

cfg = Config()
openai.api_key = cfg.openai_api_key
print_total_cost = True

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

class ApiManager:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0

    def create_chat_completion(self, messages, model=None, temperature=None, max_tokens=None) -> str:
        response = llm_utils.create_chat_completion(messages, model, temperature, max_tokens)
        prompt_tokens = count_message_tokens(messages, model)
        completion_tokens = count_message_tokens([{"role": "assistant", "content": response}], model)
        self.update_cost(prompt_tokens, completion_tokens, model)
        return response

    def update_cost(self, prompt_tokens, completion_tokens, model):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += (prompt_tokens * COSTS[model]["prompt"] +
                            completion_tokens * COSTS[model]["completion"]) / 1000
        if print_total_cost:
            print(f"Total running cost: ${self.total_cost:.3f}")

    def get_total_prompt_tokens(self):
        return self.total_prompt_tokens

    def get_total_completion_tokens(self):
        return self.total_completion_tokens

    def get_total_cost(self):
        return self.total_cost

api_manager = ApiManager()