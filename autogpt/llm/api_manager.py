from __future__ import annotations

import openai

from autogpt.config import Config
from autogpt.llm.modelsinfo import COSTS
from autogpt.logs import logger
from autogpt.singleton import Singleton

import websocket
import json
import sys

class ApiManager(metaclass=Singleton):
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0

    def reset(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0.0

    def create_chat_completion(
        self,
        messages: list,  # type: ignore
        model: str | None = None,
        temperature: float = None,
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
        cfg = Config()
        if temperature is None:
            temperature = cfg.temperature
        if deployment_id is not None:
            response = openai.ChatCompletion.create(
                deployment_id=deployment_id,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=cfg.openai_api_key,
            )
        else:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=cfg.openai_api_key,
            )
        logger.debug(f"Response: {response}")
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        self.update_cost(prompt_tokens, completion_tokens, model)
        return response

    def create_chat_completion_webui(
        self,
        messages: list,  # type: ignore
        model: str | None = None,
        temperature: float = None,
        max_tokens: int | None = None,
        deployment_id=None,
    ) -> str:
        """
        Create a chat completion using the oobabooga webui.
        Args:
        messages (list): The list of messages to send to the API.
        model (str): The model to use for the API call.
        temperature (float): The temperature to use for the API call.
        max_tokens (int): The maximum number of tokens for the API call.
        Returns:
        str: The AI's response.
        """
        # some classes to mimic the response from openai.ChatCompletion.create
        class Choice():
            def __init__(self, content):
                self.message = {
                    "content": content,
                }

        class Responses():
            def __init__(self):
                self.choices = []
        cfg = Config()
        if temperature is None:
            temperature = cfg.temperature
        
        r = Responses()
        # format the prompt for vicuna
        payload = "%s\n### Human:\n%s\n### Assistant:\n" % (messages[0]["content"], messages[1]["content"])
        logger.debug(f"Payload: {payload}")

        # use the stream endpoint instead via websocket to print each token as it's generated
        ws = websocket.create_connection("ws://localhost:5005/api/v1/stream")

        ws.send(json.dumps({
            "prompt": payload,
            "temperature": 0.1,
            "stopping_strings": ["\n###"],
        }))

        # stream the response, printing each token as it's generated
        output = ""
        while True:
            message = ws.recv()
            message = json.loads(message)
            if message["event"] == "text_stream":
                output += message["text"]
                sys.stdout.write(message["text"])
                sys.stdout.flush()
            elif message["event"] == "stream_end":
                ws.close()
                break

        response_text = output
        logger.debug(f"Response: {response_text}")
        # remove "\n##" from the end of the response
        response_text = response_text[: response_text.rfind("\n##")]
        prompt_tokens = len(payload.split())
        completion_tokens = len(response_text.split())
        self.update_cost(prompt_tokens, completion_tokens, model)
        r.choices.append(Choice(response_text))
        logger.debug("Choices: %s" % [choice.message["content"] for choice in r.choices])
        return r

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
