from typing import List, Optional
import json
import openai
import dirtyjson
from config import Config
cfg = Config()

# This is a magic function that can do anything with no-code. See
# https://github.com/Torantulino/AI-Functions for more info.
def call_ai_function(function, args, description, model=cfg.smart_llm_model):
    # For each arg, if any are None, convert to "None":
    args = [str(arg) if arg is not None else "None" for arg in args]
    # parse args to comma seperated string
    args = ", ".join(args)
    messages = [
        {
            "role": "system",
            "content": f"You are now the following python function: ```# {description}\n{function}```\n\nOnly respond with your `return` value.",
        },
        {"role": "user", "content": args},
    ]

    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0
    )

    return response.choices[0].message["content"]
