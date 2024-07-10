# AutoGPT Agent Server 

This is an initial project for creating the next generation of agent execution, which is an AutoGPT agent server.
The agent server will enable the creation of composite multi-agent systems that utilize AutoGPT agents and other non-agent components as its primitives.

## Setup

To set up the project, follow these steps inside the project directory:

1. Enter the poetry shell

   ```sh
   poetry shell
   ```
   
2. Install dependencies

   ```sh
   poetry install
   ```
   
3. Generate the Prisma client

   ```sh
   poetry run prisma generate
   ```
   
   In case Prisma generates the client for the global Python installation instead of the virtual environment, the current mitigation is to just uninstall the global Prisma package:

   ```sh
   pip uninstall prisma
   ```
   
   Then run the generation again. The path *should* look something like this:  
   `<some path>/pypoetry/virtualenvs/autogpt-server-TQIRSwR6-py3.12/bin/prisma`

4. Migrate the database. Be careful because this deletes current data in the database.

   ```sh
   poetry run prisma migrate dev
   ```
   
## Running The Server

### Starting the server directly

Run the following command:

```sh
poetry run app
```

### Running the App in the Background

**Note: this is a Unix feature and can fail on Windows. If it fails, you can run the previous command and manually move the process to the background.*  

1. Start the server. This starts the server in the background.

   ```sh
   poetry run cli start
   ```
   
2. Stop the server.

   ```sh
   poetry run cli stop
   ```
   
## Adding Test Data

1. Start the server using one of the above methods.

2. Run the populate DB command:

   ```sh
   poetry run cli test populate-db http://0.0.0.0:8000
   ```
   
   This will add a graph, a graph execution, and a cron schedule to run the graph every 5 minutes.

### Reddit Graph

There is a command to add the test Reddit graph:

```sh
poetry run cli test reddit http://0.0.0.0:8000
```

For help, run:

```sh
poetry run cli test reddit --help
```

## Testing

To run the tests:

```sh
poetry run pytest
```

## Project Outline

The current project has the following main modules:

### **blocks**

This module stores all the Agent Blocks, which are reusable components to build a graph that represents the agent's behavior.

### **data**

This module stores the logical model that is persisted in the database.
It abstracts the database operations into functions that can be called by the service layer.
Any code that interacts with Prisma objects or the database should reside in this module.
The main models are:
* `block`: anything related to the block used in the graph
* `execution`: anything related to the execution graph execution
* `graph`: anything related to the graph, node, and its relations

### **execution**

This module stores the business logic of executing the graph.
It currently has the following main modules:
* `manager`: A service that consumes the queue of the graph execution and executes the graph. It contains both pieces of logic.
* `scheduler`: A service that triggers scheduled graph execution based on a cron expression. It pushes an execution request to the manager.

### **server**

This module stores the logic for the server API.
It contains all the logic used for the API that allows the client to create, execute, and monitor the graph and its execution.
This API service interacts with other services like those defined in `manager` and `scheduler`.

### **utils**

This module stores utility functions that are used across the project.
Currently, it has two main modules:
* `process`: A module that contains the logic to spawn a new process.
* `service`: A module that serves as a parent class for all the services in the project.

## Service Communication

Currently, there are only 3 active services:

- AgentServer (the API, defined in `server.py`)
- ExecutionManager (the executor, defined in `manager.py`)
- ExecutionScheduler (the scheduler, defined in `scheduler.py`)

The services run in independent Python processes and communicate through an IPC.
A communication layer (`service.py`) is created to decouple the communication library from the implementation.

Currently, the IPC is done using Pyro5 and abstracted in a way that allows a function decorated with `@expose` to be called from a different process.

## Adding a New Agent Block

To add a new agent block, you need to create a new class that inherits from `Block` and provides the following information:
* `input_schema`: the schema of the input data, represented by a Pydantic object.
* `output_schema`: the schema of the output data, represented by a Pydantic object.
* `run` method: the main logic of the block.
* `test_input` & `test_output`: the sample input and output data for the block, which will be used to auto-test the block.
* You can mock the functions declared in the block using the `test_mock` field for your unit tests.
* If you introduce a new module under the `blocks` package, you need to import the module in `blocks/__init__.py` to make it available to the server.
* Once you finish creating the block, you can test it by running `pytest test/block/test_block.py`.
