# Tutorial: Implementing Your First `loop.py` Module in the afaas Framework

In the afaas framework, the `loop.py` module houses the execution logic of agents, orchestrating how agents interact with users and refine understanding over a conversation. In this tutorial, we'll walk through the steps to implement your own `loop.py` module, using `UserContextLoop` as an illustrative example.

## **Prerequisites:**

- Familiarity with Python's async/await syntax.
- A basic understanding of the afaas framework, especially the Agent architecture.

## **Step 1: Inheriting from BaseLoop**

Your loop should inherit from the `BaseLoop` class to access essential methods and attributes for agent execution logic.

```python
class MyLoop(BaseLoop):
    ...
```

## **Step 2: Initializing Your Loop**

During initialization, set the loop to inactive by default and associate it with an agent instance.

```python
def __init__(self, agent: BaseAgent) -> None:
    super().__init__(agent)
    self._active = False
```

## **Step 3: Implementing the `run` Method**

Define the core logic in the `run` method. Any argument passed to the `run()` method is for convenience. The agent running the loop is always accessible via `self._agent`.

```python
async def run(
    self,
    agent: BaseAgent,
    hooks: LoophooksDict,
    user_input_handler: Callable[[str], Awaitable[str]],
    user_message_handler: Callable[[str], Awaitable[str]]
) -> dict:
    ...
```

## **Step 4: Handling User Interaction**

In the `run` method, utilize the `user_input_handler` and `user_message_handler` to manage user interaction and refine the agent's understanding.

```python
# ... within run method
user_input = await user_input_handler(input_str)
user_message = await user_message_handler(message)
```

## **Step 5: Interacting with Large Language Models (LLM)**

Utilize the `execute_strategy` method to interact with a Large Language Model (LLM), sending a strategy name and parameters to dictate the model's behavior. The `ChatModelResponse` object encapsulates the model's response, which we will further explore in Section C.1.

```python
# ... within run method
model_response : ChatModelResponse = await self.execute_strategy(
    strategy_name="refine_user_context",
    user_objective=user_objectives,
)
```

## **Step 6: Parsing Model Responses**

Check the `parsed_result` to act based on the model's feedback. Your logic may vary based on the requirements of your agent.

```python
# ... within run method
if model_response.parsed_result["name"] == RefineUserContextFunctionNames.REFINE_REQUIREMENTS:
    ...
elif model_response.parsed_result["name"] == RefineUserContextFunctionNames.REQUEST_SECOND_CONFIRMATION:
    ...
# and so on...
```

## **Step 7: Saving Agent State**

Optionally, you might want to save the current state of your agent for future reference.

```python
# ... within run method
await self.save_agent()
```

# **Conclusion:**
The `loop.py` module is pivotal in managing the logic and user interactions within your agent. By following the steps outlined, you can craft a custom loop to suit the specific needs of your agent, making your interactions more intuitive and responsive.

# **Resources:**
- [UserContextLoop Example](https://raw.githubusercontent.com/ph-ausseil/afaas/first-agent-tutorial/autogpts/autogpt/autogpt/core/agents/usercontext/loop.py)
