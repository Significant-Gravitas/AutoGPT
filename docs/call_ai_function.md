## Function `call_ai_function`

This function is designed to call an AI function with the specified arguments and description. The function takes in the following parameters:

### Parameters
- `function`: a string representing the python function to be called
- `args`: a list of arguments to be passed to the function
- `description`: a brief description of the function
- `model`: an optional parameter that specifies the AI function model to use. If this parameter is not specified, the default model will be used.

### Example
```python
from config import Config

cfg = Config()

from llm_utils import create_chat_completion

def call_ai_function(function, args, description, model=None):
    if model is None:
        model = cfg.smart_llm_model
    args = [str(arg) if arg is not None else "None" for arg in args]
    args = ", ".join(args)
    messages = [
        {
            "role": "system",
            "content": f"You are now the following python function: ```# {description}\n{function}```\n\nOnly respond with your `return` value.",
        },
        {"role": "user", "content": args},
    ]

    response = create_chat_completion(
        model=model, messages=messages, temperature=0
    )

    return response

result = call_ai_function("print('Hello, world!')", [], "print a message")
print(result)
```

This will output:
```
Hello, world!

```