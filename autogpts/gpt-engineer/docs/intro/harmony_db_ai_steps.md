# Harmony of AI, DB, and Steps
GPT-Engineer is a powerful tool that uses AI to automate software engineering tasks. It is designed with a modular architecture that makes it easy to extend and customize. The core components of GPT-Engineer are the AI class, the DB class, and the steps module. These components work together in harmony to provide a flexible and powerful system for automating software engineering tasks.

<br>

## AI Class
The AI class is the main interface to the GPT-3 model. It provides methods to start a conversation with the model, continue an existing conversation, and format system and user messages. The AI class is responsible for interacting with the GPT-3 model and generating the AI's responses.

<br>

## DB Class
The DB class represents a simple database that stores its data as files in a directory. It is a key-value store, where keys are filenames and values are file contents. The DB class is responsible for managing the data used by the GPT-Engineer system.

<br>

## Steps Module
The steps module defines a series of steps that the AI can perform to generate code, clarify instructions, generate specifications, and more. Each step is a function that takes an AI and a set of databases as arguments and returns a list of messages. The steps module is responsible for controlling the flow of the GPT-Engineer system.

<br>

### How Each Step is Made
Each step in the steps module is a function that takes an AI and a set of databases as arguments. The function performs a specific task, such as generating code or clarifying instructions, and returns a list of messages. The messages are then saved to the databases and used in subsequent steps.

Here is an example of a step function:

<br>

```python
def simple_gen(ai: AI, dbs: DBs):
    """Generate code based on the main prompt."""
    system = dbs.preprompts["generate"]
    user = dbs.input["main_prompt"]
    messages = ai.start(system, user)
    dbs.workspace["code.py"] = messages[-1]["content"]
    return messages
```
<br>

This function uses the AI to generate code based on the main prompt. It reads the main prompt from the input database, generates the code, and saves the code to the workspace database.

<br>

### How to Make Your Own Step
To make your own step, you need to define a function that takes an AI and a set of databases as arguments. Inside the function, you can use the AI to generate responses and the databases to store and retrieve data. Here is an example:

<br>

```python
def generate_function(ai: AI, dbs: DBs):
    """Generate a simple Python function."""
    function_name = dbs.input["function_name"]
    function_description = dbs.input["function_description"]
    system = "Please generate a Python function."
    user = f"I want a function named '{function_name}' that {function_description}."
    messages = ai.start(system, user)
    dbs.workspace[f"{function_name}.py"] = messages[-1]["content"]
    return messages
```

<br>

In this custom step, we're asking the AI to generate a Python function based on a function name and a description provided by the user. The function name and description are read from the input database. The generated function is saved to the workspace database with a filename that matches the function name. You would simply need to provide a `function_name` file and `function_description` file with necessary details in the input database (your project folder) to use this step.

<br>

For example, if the user provides the function name "greet" and the description "prints 'Hello, world!'", the AI might generate the following Python function:

```python
def greet():
    print('Hello, world!')
```

<br>

This function would be saved to the workspace database as `greet.py`.
