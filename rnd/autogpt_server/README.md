# Next Gen AutoGPT 

This is a research project into creating the next generation of autogpt, which is an autogpt agent server.

The agent server will enable the creation of composite multi-agent system that utilize AutoGPT Agent as its default agent.

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
poetry run app
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

## Project Outline

The current project has the following main modules:

### **blocks**

This module stores all the Agent Block, a reusable component to build a graph that represents the agent's behavior.

### **data**

This module stores the logical model that is persisted in the database.
This module abstracts the database operation into a function that can be called by the service layer.
Any code that interacts with Prisma objects or databases should live in this module.
The main models are:
* `block`: anything related to the block used in the graph
* `execution`: anything related to the execution graph execution
* `graph`: anything related to the graph, node, and its relation

### **execution**

This module stores the business logic of executing the graph.
It currently has the following main modules:
* `manager`: A service that consumes the queue of the graph execution and executes the graph. It contains both of the logic.
* `scheduler`: A service that triggers scheduled graph execution based on cron expression. It will push an execution request to the manager.

### **server**

This module stores the logic for the server API.
It stores all the logic used for the API that allows the client to create/execute/monitor the graph and its execution.
This API service will interact with other services like the ones defined in `manager` and `scheduler`.

### **utils**

This module stores the utility functions that are used across the project.
Currently, it only has two main modules:
* `process`: A module that contains the logic to spawn a new process.
* `service`: A module that becomes a parent class for all the services in the project.

## Service Communication

Currently, there are only 3 active services:

- AgentServer (the API, defined in `server.py`)
- ExecutionManager (the executor, defined in `manager.py`)
- ExecutionScheduler (the scheduler, defined in `scheduler.py`)

The service is running in an independent python process and communicates through an IPC.
A communication layer (`service.py`) is created to decouple the communication library from the implementation.

Currently, the IPC is done using Pyro5 and abstracted in a way that it allows a function that is decorated with an `@expose` function can be called from the different process.

## Adding a new Agent Block

To add a new agent block, you need to create a new class that inherits from `Block` that provide these information:
* `input_schema`: the schema of the input data, represented by a pydantic object.
* `output_schema`: the schema of the output data, represented by a pydantic object.
* `run` method: the main logic of the block
* `test_input` & `test_output`: the sample input and output data for the block, this will be used to auto-test the block.
* You can mock the functions declared in the block using the `test_mock` field for your unit test.
* If you introduced a new module under the `blocks` package, you need to import the module in `blocks/__init__.py` to make it available to the server.
* Once you finished creating the block, you can test it by running the test using `pytest test/block/test_block.py`.
