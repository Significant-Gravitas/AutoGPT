
## Agent Executor

### What it is
A specialized block that allows you to run an existing agent within your current agent, enabling nested agent execution capabilities.

### What it does
This block executes a pre-configured agent (identified by a graph) and monitors its execution, collecting and forwarding any outputs produced during the execution process.

### How it works
1. Takes information about which agent to run (graph details) and input data
2. Starts the execution of the specified agent
3. Listens for events from the running agent
4. Monitors the execution progress
5. Collects and forwards any outputs produced by the agent
6. Continues until the execution is completed, terminated, or fails

### Inputs
- User ID: The identifier of the user running the agent
- Graph ID: The identifier of the specific agent (graph) to execute
- Graph Version: The version number of the agent to run
- Data: The actual input data that will be processed by the agent
- Input Schema: The structure definition for the input data
- Output Schema: The structure definition for the expected output data

### Outputs
This block's outputs are dynamic and depend on the executed agent. It will yield:
- Output Name: The name/identifier of each output produced
- Output Data: The actual data/content produced by the executed agent

### Possible use case
Imagine you have a complex task that requires multiple agents to work together. For example, you might have a main agent that needs to:
1. Analyze customer feedback
2. Generate responses
3. Translate those responses to different languages

Instead of building all this functionality into one agent, you could use the Agent Executor to run specialized agents for each subtask. The main agent would use this block to execute each specialized agent in sequence, passing the results from one to the next.

This block would help coordinate these nested agents, ensuring each one runs properly and their outputs are correctly captured and passed along to the next stage of processing.

