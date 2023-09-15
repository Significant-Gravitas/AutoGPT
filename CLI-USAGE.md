## CLI Documentation

This document describes how to interact with the project's CLI (Command Line Interface). It includes the types of outputs you can expect from each command. Before launching the frontend, ensure that an agent is already running. Note that the `agents stop` command will terminate any process running on port 8000.

### 1. Entry Point for the CLI

Running the `./run` command without any parameters will display the help message, which provides a list of available commands and options. Additionally, you can append `--help` to any command to view help information specific to that command.

```sh
./run
```

**Output**:

```
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  agents     Agents group command
  benchmark  Benchmark group command
  frontend   Frontend command group
  setup      Setup command
```

If you need assistance with any command, simply add the `--help` parameter to the end of your command, like so:

```sh
./run COMMAND --help
```

This will display a detailed help message regarding that specific command, including a list of any additional options and arguments it accepts.

### 2. Setup Command

```sh
./run setup
```

**Output**:

```
Setup initiated
Installation has been completed.
```

This command initializes the setup of the project.

### 3. Agents Commands

**a. List All Agents**

```sh
./run agents list
```

**Output**:

```
Available agents: 🤖
        🐙 forge
        🐙 autogpt
```

Lists all the available agents.

**b. Create a New Agent**

```sh
./run agents create my_agent
```

**Output**:

```
🎉 New agent 'my_agent' created and switched to the new directory in autogpts folder.
```

Creates a new agent named 'my_agent'.

**c. Start an Agent**

```sh
./run agents start my_agent
```

**Output**:

```
... (ASCII Art representing the agent startup)
[Date and Time] [forge.sdk.db] [DEBUG] 🐛  Initializing AgentDB with database_string: sqlite:///agent.db
[Date and Time] [forge.sdk.agent] [INFO] 📝  Agent server starting on http://0.0.0.0:8000
```

Starts the 'my_agent' and displays startup ASCII art and logs.

**d. Stop an Agent**

```sh
./run agents stop
```

**Output**:

```
Agent stopped
```

Stops the running agent.

### 4. Benchmark Commands

**a. List Benchmark Categories**

```sh
./run benchmark categories list
```

**Output**:

```
Available categories: 📚
        📖 code
        📖 safety
        📖 memory
        ... (and so on)
```

Lists all available benchmark categories.

**b. List Benchmark Tests**

```sh
./run benchmark tests list
```

**Output**:

```
Available tests: 📚
        📖 interface
                🔬 Search - TestSearch
                🔬 Write File - TestWriteFile
        ... (and so on)
```

Lists all available benchmark tests.

**c. Show Details of a Benchmark Test**

```sh
./run benchmark tests details TestWriteFile
```

**Output**:

```
TestWriteFile
-------------

        Category:  interface
        Task:  Write the word 'Washington' to a .txt file
        ... (and other details)
```

Displays the details of the 'TestWriteFile' benchmark test.

**d. Start Benchmark for the Agent**

```sh
./run benchmark start my_agent
```

**Output**:

```
(more details about the testing process shown whilst the test are running)
============= 13 failed, 1 passed in 0.97s ============...
```

Displays the results of the benchmark tests on 'my_agent'.

### 5. Frontend Command

```sh
./run frontend
```

**Output**:

```
Agent is running.
Launching frontend
... (more details about the launch process)
```

Launches the frontend, with debugging and service details mentioned.

---

Remember to start an agent before launching the frontend and that the `agents stop` command terminates any process on port 8000.