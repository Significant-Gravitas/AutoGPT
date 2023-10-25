# GPT-Engineer Documentation

GPT-Engineer is a project that uses GPT-4 to automate the process of software engineering. It includes several Python scripts that interact with the GPT-4 model to generate code, clarify requirements, generate specifications, and more.

<br>

## Core Components
### 1. AI Class (`gpt_engineer/ai.py`)
The AI class is the main interface to the GPT-4 model. It provides methods to start a conversation with the model, continue an existing conversation, and format system and user messages.

<br>

### 2. Chat to Files (`gpt_engineer/chat_to_files.py`)
This module contains two main functions:

`parse_chat(chat)`: This function takes a chat conversation and extracts all the code blocks and preceding filenames. It returns a list of tuples, where each tuple contains a filename and the corresponding code block.

`to_files_and_memory(chat, dbs)`: This function takes the chat and the DBs as arguments. DBs contains the workspace and memory path. The function first saves the entire chat as a text file in the memory path. Then it calls the to_files function to  write each file to the workspace.

`to_files(chat, db)`: This function takes the chat and workspace DB as arguments. It calls the parse_chat function to parse the chat and get the files. Each file is then saved to the workspace.

<br>

### 3. DB Class (`gpt_engineer/db.py`)
The DB class represents a simple database that stores its data as files in a directory. It provides methods to check if a key (filename) exists in the database, get the value (file content) associated with a key, and set the value associated with a key.

The DBs class is a dataclass that contains instances of the DB class for different types of data (memory, logs, input, workspace, and preprompts).

<br>

### 4. Main Script (`gpt_engineer/main.py`)
The main script uses the `Typer` library to create a command-line interface. It sets up the AI model and the databases, and then runs a series of steps based on the provided configuration.

<br>

### 5. Steps (`gpt_engineer/steps.py`)
This module defines a series of steps that can be run in the main script.  Each step is a function that takes an instance of the AI class and an instance of the DBs class, and returns a list of messages.

<br>

The steps include:

`simple_gen(ai, dbs)`: Run the AI on the main prompt and save the results. <br>
`clarify(ai, dbs)`: Ask the user if they want to clarify anything and save the results to the workspace. <br>
`gen_spec(ai, dbs)`: Generate a spec from the main prompt + clarifications and save the results to the workspace. <br>
`respec(ai, dbs)`: Ask the AI to reiterate the specification and save the results to the workspace. <br>
`gen_unit_tests(ai, dbs)`: Generate unit tests based on the specification. <br>
`gen_clarified_code(ai, dbs)`: Generate code based on the main prompt and clarifications. <br>
`gen_code(ai, dbs)`: Generate code based on the specification and unit tests. <br>
`execute_entrypoint(ai, dbs)`: Execute the entrypoint script in the workspace. <br>
`gen_entrypoint(ai, dbs)`: Generate an entrypoint script based on the code in the workspace. <br>
`use_feedback(ai, dbs)`: Ask the AI to generate code based on feedback. <br>
`fix_code(ai, dbs)`: Ask the AI to fix any errors in the code. <br>

The steps are grouped into different configurations (default, benchmark, simple, tdd, tdd+, clarify, respec, execute_only, use_feedback), which can be selected when running the main script.
