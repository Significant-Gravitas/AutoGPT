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
