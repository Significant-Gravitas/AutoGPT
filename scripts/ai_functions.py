from typing import List, Optional
import json
import openai

def call_ai_function(function, args, description, model = "gpt-4"):
    # parse args to comma seperated string
    args = ", ".join(args)
    messages = [{"role": "system", "content": f"You are now the following python function: ```# {description}\n{function}```\n\nOnly respond with your `return` value."},{"role": "user", "content": args}]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message["content"]
