# Next Gen AutoGPT 

This is a research project into creating the next generation of autogpt, which is an autogpt agent server.

The agent server will enable the creation of composite multi-agent system that utilize AutoGPT Agent as its default agent.


## Project Outline

Currently the project mainly consist of these components:

*agent_api*
A component that will expose API endpoints for the creation & execution of agents.
This component will make connections to the database to persist and read the agents.
It will also trigger the agent execution by pushing its execution request to the ExecutionQueue.

*agent_executor*
A component that will execute the agents.
This component will be a pool of processes/threads that will consume the ExecutionQueue and execute the agent accordingly. 
The result and progress of its execution will be persisted in the database.

## Setup

This setup is for MacOS/Linux.
To setup the project follow these steps inside the project directory:

1. Enter poetry shell
   ```
   poetry shell
   ```

2. Install dependencies
   ```
   poetry install
   ```

3. Generate prisma client
   ```
   poetry run prisma generate
   ```

   In case prisma generates client for the global python installation instead of the virtual environment the current mitigation is to just uninstall the global prisma package:
   ```
   pip uninstall prisma
   ```

   And then run the generation again.
   The path *should* look something like this:  
   `<some path>/pypoetry/virtualenvs/autogpt-server-TQIRSwR6-py3.12/bin/prisma`

4. Migrate the database, be careful because this deletes current data in the database
   ```
   poetry run prisma migrate dev
   ```

# Running The Server

## Starting the server directly

Run the following command:

```
poetry run cli app
```

## Running the App in the Background

1. Start the server, this starts the server in the background
   ```
   poetry run cli start
   ```
    
   You may need to change the permissions of the file to make it executable
   ```
   chmod +x autogpt_server/cli.py
   ```

2. Stop the server
   ```
   poetry run cli stop
   ```

## Adding Test Data

1. Start the server using 1 of the above methods

2. Run the populate db command

```
poetry run cli test populate-db http://0.0.0.0:8000
```

This will add a graph, a graph execution and a cron schedule to run the graph every 5 mins

### Reddit Graph

There is a command to add the test reddit graph

```
poetry run cli test reddit http://0.0.0.0:8000
```

For help run:
```
poetry run cli test reddit --help

```

# Testing

To run the tests
```
poetry run pytest
```
