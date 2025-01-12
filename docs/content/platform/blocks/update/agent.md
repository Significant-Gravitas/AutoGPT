
# Agent Block Documentation

## Agent Executor

### What it is
A specialized block that allows one agent to execute another agent, enabling nested agent functionality within your automation workflows.

### What it does
This block takes an existing agent (identified by its graph information) and executes it as part of another agent's workflow. It manages the execution process, monitors the progress, and handles the output data produced by the executed agent.

### How it works
1. The block receives necessary identification and input data
2. It initiates the execution of the specified agent
3. Listens for events from the executed agent
4. Monitors the execution status and progress
5. Collects and processes any output produced
6. Returns the processed output data to the parent agent

### Inputs
- User ID: The identifier for the user running the agent
- Graph ID: The unique identifier for the agent (graph) to be executed
- Graph Version: The specific version of the agent to run
- Data: The input data that will be passed to the executed agent
- Input Schema: The structure definition for the input data
- Output Schema: The structure definition for the expected output data

### Outputs
This block has a dynamic output system that:
- Yields output data as it's produced by the executed agent
- Each output includes:
  - Output Name: Identifier for the specific output
  - Output Data: The actual data produced by the executed agent

### Possible use case
Imagine you have a main agent that handles customer service requests. Within this agent, you might want to execute a specialized agent that performs detailed product analysis. The Agent Executor block would allow your main agent to:
1. Receive a customer query
2. Execute the product analysis agent
3. Wait for and collect the analysis results
4. Use those results to provide a comprehensive response to the customer

The block handles all the complexity of managing this nested agent execution, making it appear seamless to the end user.

### Notes
- The block provides detailed logging of the execution process
- It can handle various execution statuses (completed, terminated, failed)
- It specifically processes output blocks from the executed agent
- The execution is monitored in real-time through an event bus system

