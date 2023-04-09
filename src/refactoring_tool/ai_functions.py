from typing import List, Optional
import json
import openai
from functools import lru_cache


class AIFunctionCaller:

    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.5):
        self.model = model
        self.temperature = temperature

    @staticmethod
    def validate_and_sanitize_input(function: str, args: List[str], description: str) -> bool:
        if not function or not isinstance(function, str):
            return False

        if not args or not isinstance(args, (List, tuple)):
            return False

        if not description or not isinstance(description, str):
            return False

        return True

    @lru_cache(maxsize=100)
    def call_ai_function(self, function: str, args: List[str], description: str) -> str:
        # Validate and sanitize inputs
        if not self.validate_and_sanitize_input(function, args, description):
            raise ValueError("Invalid input parameters")

        # Truncate or summarize input if necessary (assuming a 4096 token limit)
        if len(function) + len(args) + len(description) > 4000:
            function = function[:1000]
            args = [arg[:500] for arg in args]
            description = description[:1000]

        # Parse args to comma separated string
        args = ", ".join(args)
        messages = [
            {
                "role": "system",
                "content": f"You are now the following python function: ```# {description}\n{function}```\n\nOnly respond with your `return` value.",
            },
            {"role": "user", "content": args},
        ]

        try:
            response = openai.ChatCompletion.create(
                model=self.model, messages=messages, temperature=self.temperature
            )
            # Add this line to print the full response
            print("API Response:", response)
        except Exception as e:
            print(f"Error calling AI API: {e}")
            return ""

        ai_response = response.choices[0].message["content"]

        # Implement post-processing if needed (e.g., formatting or minor adjustments)
        # ai_response = post_process(ai_response)

        return ai_response
