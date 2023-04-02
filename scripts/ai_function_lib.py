import json
import openai
import configparser

def call_ai_function(function, args, description, config_path="config.ini"):
    # Load the configuration file
    config = configparser.ConfigParser()
    config.read(config_path)

    # Get the chosen model from the config file
    model = config.get("AI", "Chosen_Model", fallback="gpt-4")

    # Parse args to comma separated string
    args = ", ".join(args)
    messages = [
        {
            "role": "system",
            "content": f"You are now the following python function: ```# {description}\n{function}```\n\nOnly respond with your `return` value.",
        },
        {"role": "user", "content": args},
    ]

  # Use different AI APIs based on the chosen model
    if model == "gpt-4":
        response = openai.ChatCompletion.create(
            model=model, messages=messages, temperature=0
        )
       
    
    
    
    # GPT-3.5 with a json verification loop.
    elif model == "gpt-3.5":
         try:
            json.dumps(messages)
        except (TypeError, ValueError):
            return "I did not return a valid json format. I should repeat the last instructions and try again."

        response = openai.ChatCompletion.create(
            model=model, messages=messages, temperature=0
        )
    elif model == "some_other_api":
        # Add code to call another AI API with the appropriate parameters
        response = some_other_api_call(parameters)
    else:
        raise ValueError(f"Unsupported model: {model}")

    return response.choices[0].message["content"]
