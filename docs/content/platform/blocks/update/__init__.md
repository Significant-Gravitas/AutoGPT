
## Agent Executor

### What it is
A specialized block that enables the execution of an existing agent within another agent, allowing for nested agent operations.

### What it does
This block manages the execution of a pre-configured agent (graph) and processes its outputs. It handles the entire lifecycle of the agent execution, including monitoring its status and collecting results.

### How it works
1. Receives execution parameters including user identification and graph details
2. Initiates the execution of the specified agent
3. Monitors the execution progress through an event system
4. Collects and processes outputs from the agent
5. Reports execution status and results
6. Continues monitoring until the execution is completed, terminated, or fails

### Inputs
- User ID: The identifier for the user initiating the execution
- Graph ID: The unique identifier of the agent (graph) to be executed
- Graph Version: The specific version number of the agent to run
- Data: The input data that will be provided to the agent for processing
- Input Schema: The structure definition that describes the expected input format
- Output Schema: The structure definition that describes the expected output format

### Outputs
- Dynamic Outputs: The block yields outputs based on the executed agent's results
- Each output consists of:
  - Output Name: A label identifying the type of output
  - Output Data: The actual data produced by the executed agent

### Possible use case
A customer service automation system where a main agent needs to delegate specific tasks to specialized sub-agents. For example, a primary customer service agent might use this block to execute a specialized product recommendation agent when a customer asks for product suggestions. The main agent can continue its conversation while the recommendation agent processes the request and returns detailed product recommendations.

