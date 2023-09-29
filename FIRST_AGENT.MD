# Building an agent : Part 2 `loop.py`

## Introduction:
The `loop.py` module within the afaas framework is responsible for the execution logic of agents. In this tutorial, we'll go through how to construct and use the `loop.py` module, we will use as example  on the `UserContextLoop` class located in `autogpts/autogpt/autogpt/core/agents/usercontext/loop.py`. The agent `UsercontextAgent` is in `autogpts/autogpt/autogpt/core/agents/usercontext/main.py` file. 

❗ Reminder ❗ :
**ALL Agent** should inherit from `BaseAgent`, they willwith a `_loop` property that inherit from `BaseLoop`.
`UsercontextAgent` has a `_loop` property that is an instance of `UserContextLoop`

## Pre-requisites:
- Familiarity with Python's async/await syntax.
- A basic understanding of the afaas framework structure, particularly the Agent architecture which comprises `agent.py`, `loop.py`, `models.py`, and the strategies directory.

## Steps:

### 1. Inheriting from the BaseLoop:
Your loop should inherit from the `BaseLoop` class. This ensures that it has access to essential methods and attributes for agent execution logic.
\```python
class UserContextLoop(BaseLoop):
    ...
\```

### 2. Initialization:
During initialization, the loop is set to inactive by default. It is necessary to associate your loop with an agent instance.
\```python
def __init__(self, agent: BaseAgent) -> None:
    super().__init__(agent)
    self._active = False
\```

### 3. Running the Loop:
The core logic of refining user context is defined in the run method. This method receives an agent, hooks for processing, and handlers for user input and messages.
\```python
async def run(
    self,
    agent: BaseAgent,
    hooks: LoophooksDict,
    user_input_handler: Callable[[str], Awaitable[str]],
    user_message_handler: Callable[[str], Awaitable[str]]
) -> dict:
    ...
\```

# Objects, Classes, Methods, and Properties Breakdown:

## 1. **BaseAgent**
- **Usage:** 
    ```python
    async def run(self, agent: BaseAgent, ...)
    ```
    It indicates that the `agent` argument should be an instance derived from or of type `BaseAgent`, in our case `UserContextAgent`

- **Main Methods/Properties :**
    - `_logger`: This is a logger associated with the agent. Logging is essential for debugging and tracking the execution of the code.
        ```python
        self._agent._logger.info(...)
        ```
    - `execute_strategy()` interact with a LLM

## 2. **ChatModelResponse**
The `ChatModelResponse` class is critical because it encapsulates the response from a call to a Large Language Model (LLM) through the method `execute_strategy`. Understanding how to interact with the model and interpret its responses is essential when developing an agent loop.

### Structure:
- `response`: Contains the raw response from the assistant in a dictionary format. This might include the text of the message, any attachments, or other metadata.
  
- `parsed_result`: Holds the processed or interpreted results from the response. Developers can easily access actionable data without constantly parsing the raw response.

- `content`: Contains additional content or metadata about the model's response.

### Interacting with ChatModelResponse:
When you invoke `await self.execute_strategy(strategy_name="refine_user_context", **kwargs)`, it sends the strategy's name and parameters to the LLM. The strategy dictates the model's behavior.

For instance, in the provided `UserContextLoop`, there's a pattern that checks the `parsed_result` to act based on the model's feedback:

```python
if model_response.parsed_result["name"] == RefineUserContextFunctionNames.REFINE_REQUIREMENTS:
    ...
elif model_response.parsed_result["name"] == RefineUserContextFunctionNames.REQUEST_SECOND_CONFIRMATION:
    ...
```

## 3. **user_input_handler**
- **Usage:**
    ```python
    user_objectives = await user_input_handler(input_dict)
    ```
    An async function passed to the `run` method, it appears to handle user input and return a response.

## 4. **_is_running & _is_active**
Flags that control the loop's flow. `_is_running` keeps the loop iterating, and `_is_active` dictates if the loop is active or paused.

## 5. **save_agent()**
- **Usage:**
    ```python
    await self.save_agent()
    ```
    Potentially a method that saves the current agent state. It's flagged as TODO, indicating future implementation plans.

## 6. **LoophooksDict**
- **Usage:**
    ```python
    hooks: LoophooksDict in the run method signature.
    ```
- **Description:** A dictionary type designed to hold hooks, potentially user-defined actions or modifications performed at certain loop stages.
  
**Disclaimer:** LoophooksDict is for advanced users and allows the creation of plugins. We will not detail its usage in this tutorial.


# Conclusion:
The `loop.py` module plays a crucial role in refining user contexts within agents in the afaas framework. By understanding and effectively using this module, developers can optimize agent-user interactions, making them more intuitive and accurate.
