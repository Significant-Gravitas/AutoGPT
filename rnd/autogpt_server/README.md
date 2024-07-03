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

1. Install dependencies
   ```
   poetry install
   ```

1. Generate prisma client
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

1. Migrate the database, be careful because this deletes current data in the database
   ```
   poetry run prisma migrate dev
   ```
   
1. Start the server, this starts the server in the background
   ```
   poetry run python ./autogpt_server/cli.py start
   ```
    
   You may need to change the permissions of the file to make it executable
   ```
   chmod +x autogpt_server/cli.py
   ```

1. Stop the server
   ```
   poetry run python ./autogpt_server/cli.py stop
   ```

1. To run the tests
   ```
   poetry run pytest
   ```
