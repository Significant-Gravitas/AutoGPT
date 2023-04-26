from __future__ import annotations

import numpy as np
import openai
import spacy

from autogpt.config import Config
from autogpt.logs import logger
from autogpt.modelsinfo import COSTS
from autogpt.singleton import Singleton


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

        # Define the maximum context length and chunk size
        max_context_length = self.get_total_completion_tokens()
        spacy_max_length = spacy_max_length = 1_000_000

        # This needs to be text, not a list
        text = " ".join(text_list)

        # Tokenize the input text using spaCy and process it in chunks
        nlp = spacy.load("en_core_web_sm")
        tokens = []
        for start in range(0, len(text), spacy_max_length):
            chunk_text = text[start:start+spacy_max_length]
            doc = nlp(chunk_text)
            tokens.extend(token.text for token in doc)

        # Split the tokens into smaller chunks
        text_chunks = [tokens[i:i+max_context_length] for i in range(0, len(tokens), max_context_length)]
        embeddings = []

        # Create embeddings and average together
        for chunk in text_chunks:
            # Convert tokens back to text for each chunk
            chunk_text = " ".join(chunk)
            
            # Submit
            if cfg.use_azure:
                response = openai.Embedding.create(
                    input=chunk_text,
                    engine=cfg.get_azure_deployment_id_for_model(model),
                )
            else:
                response = openai.Embedding.create(input=chunk_text, model=model)

            # Update cost and embeddings bits
            self.update_cost(response.usage.prompt_tokens, 0, model)
            embedding_for_chunk = response["data"][0]["embedding"]
            embeddings.append(embedding_for_chunk)

        combined_embedding = np.mean(embeddings, axis=0)

        return combined_embedding.tolist()

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
