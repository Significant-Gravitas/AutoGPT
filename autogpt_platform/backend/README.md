# AutoGPT Agent Server 

This is an initial project for creating the next generation of agent execution, which is an AutoGPT agent server.
The agent server will enable the creation of composite multi-agent systems that utilize AutoGPT agents and other non-agent components as its primitives.

## Docs

You can access the docs for the [AutoGPT Agent Server here](https://docs.agpt.co/server/setup).

## Setup

We use the Poetry to manage the dependencies. To set up the project, follow these steps inside this directory:

0. Install Poetry
    ```sh
    pip install poetry
    ```
    
1. Configure Poetry to use .venv in your project directory
    ```sh
    poetry config virtualenvs.in-project true
    ```

2. Enter the poetry shell

   ```sh
   poetry shell
   ```
   
3. Install dependencies

   ```sh
   poetry install
   ```

4. Copy .env.example to .env

   ```sh
   cp .env.example .env
   ```
   
5. Generate the Prisma client

   ```sh
   poetry run prisma generate
   ```
   

   > In case Prisma generates the client for the global Python installation instead of the virtual environment, the current mitigation is to just uninstall the global Prisma package:
   >
   > ```sh
   > pip uninstall prisma
   > ```
   >
   > Then run the generation again. The path *should* look something like this:  
   > `<some path>/pypoetry/virtualenvs/backend-TQIRSwR6-py3.12/bin/prisma`

6. Migrate the database. Be careful because this deletes current data in the database.

   ```sh
   docker compose up db -d
   poetry run prisma migrate deploy
   ```

## Running The Server

### Starting the server without Docker

To run the server locally, start in the autogpt_platform folder:

```sh
cd ..
```

Run the following command to run database in docker but the application locally:

```sh
docker compose --profile local up deps --build --detach
cd backend
poetry run app
```

### Starting the server with Docker

Run the following command to build the dockerfiles:

```sh
docker compose build
```

Run the following command to run the app:

```sh
docker compose up
```

Run the following to automatically rebuild when code changes, in another terminal:

```sh
docker compose watch
```

Run the following command to shut down:

```sh
docker compose down
```

If you run into issues with dangling orphans, try:

```sh
docker compose down --volumes --remove-orphans && docker-compose up --force-recreate --renew-anon-volumes --remove-orphans  
```

## Testing

To run the tests:

```sh
poetry run test
```

## Development

### Formatting & Linting
Auto formatter and linter are set up in the project. To run them:

Install:
```sh
poetry install --with dev
```

Format the code:
```sh
poetry run format
```

Lint the code:
```sh
poetry run lint
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


By default the daemons run on the following ports: 

Execution Manager Daemon: 8002
Execution Scheduler Daemon: 8003
Rest Server Daemon: 8004

## Adding a New Agent Block

To add a new agent block, you need to create a new class that inherits from `Block` and provides the following information:
* All the block code should live in the `blocks` (`backend.blocks`) module.
* `input_schema`: the schema of the input data, represented by a Pydantic object.
* `output_schema`: the schema of the output data, represented by a Pydantic object.
* `run` method: the main logic of the block.
* `test_input` & `test_output`: the sample input and output data for the block, which will be used to auto-test the block.
* You can mock the functions declared in the block using the `test_mock` field for your unit tests.
* Once you finish creating the block, you can test it by running `poetry run pytest -s test/block/test_block.py`.
