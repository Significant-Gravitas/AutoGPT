import json
import openai
import configparser

config = configparser.ConfigParser()
config.read("config.ini")
model = config.get("AI", "Chosen_Model", fallback="gpt-4")

gpt4all_model = None

# Load the GPT4All model conditionally
def load_gpt4all_model():
    global gpt4all_model
    from nomic.gpt4all import GPT4All
    gpt4all_model = GPT4All()
    gpt4all_model.open()

if model == "gpt4all":
    load_gpt4all_model()

def call_ai_function(function, args, description):
    # Get the chosen model from the config file
    model = config.get("AI", "Chosen_Model", fallback="gpt-4")

    # Parse args to comma-separated string
    args = ", ".join(args)
    messages = [
        {
            "role": "system",
            "content": f"You are now the following python function: ```# {description}\n{function}```\n\nOnly respond with your `return` value.",
        },
        {"role": "user", "content": args},
    ]

    # If the chosen model is GPT-4, call the OpenAI API with the appropriate parameters
    if model == "gpt-4":
        response = openai.ChatCompletion.create(
            model=model, messages=messages, temperature=0
        )

    # If the chosen model is GPT-3.5, check if messages are valid JSON before calling the OpenAI API
    elif model == "gpt-3.5":
        try:
            json.dumps(messages)
        except (TypeError, ValueError):
            response = "I did not return a valid json format. I should repeat the last instructions and try again."
        else:
            response = openai.ChatCompletion.create(
                model=model, messages=messages, temperature=0
            )
            response = response.choices[0].message["content"]
            
            
    #If the chosen model is gpt4all, call the GPT4all Python Client
    
    elif model == "gpt4all":
        response = call_gpt4all(parameters)

    
    # If the chosen model is some other API, call that API with the appropriate parameters
    elif model == "some_other_api":
        response = some_other_api_call(description=messages)

    # If the chosen model is unsupported, raise an error
    else:
        raise ValueError(f"Unsupported model: {model}")

    return response


# Call the GPT4All model to process the function, arguments, and description
def call_gpt4all(description):
    #def call_gpt4all(function, args, description):
    #prompt = f"{description}. The function should have the following signature:\n{function}\n\nExample input: {', '.join(args)}"
    prompt = description
    result = gpt4all_model.prompt(prompt)
    return result.strip()
