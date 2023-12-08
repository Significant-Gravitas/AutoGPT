# Interacting with the LLM, hands on with chat completion classes

In this segment, we're going to simulate a real-world scenario where an agent interacts with a user to refine and understand their request better. This process utilizes the afaas framework, specifically the chat completion classes, to facilitate the interaction with the Large Language Model (LLM). Our journey will traverse through creating chat messages, defining model functions, building a chat prompt, strategizing the prompt processing, and finally, handling the model's response within a loop.

## 1. Crafting Conversations with `ChatMessage`

The conversation begins by creating messages from both the user and the assistant. Each message is encapsulated as a `ChatMessage` object, identifying the sender and the content.

```python
from AFAAS.core.agents.usercontext.strategies.refine_user_context import Role, ChatMessage

# User initiates the conversation
user_message = ChatMessage(role=Role.USER, content="I need help with my math homework.")

# Assistant acknowledges and asks for specifics
assistant_message1 = ChatMessage(role=Role.ASSISTANT, content="Of course! What topic are you struggling with?")
```
[Source File](https://github.com/ph-ausseil/afaas/blob/5as-autogpt-integration/autogpts/autogpt/autogpt/core/agents/usercontext/strategies/refine_user_context.py)

## 2. Defining The Action with `CompletionModelFunction`

Now, let's define some functions that the assistant could perform to assist the user better using `CompletionModelFunction`.

```python
from AFAAS.core.core.resource.model_providers.openai import CompletionModelFunction, FunctionParameters, Property

# Define a function to solve quadratic equations
solve_quad = CompletionModelFunction(
    name='solve_quadratic',
    parameters=FunctionParameters(
        properties=[
            Property(name='a', type='float'),
            Property(name='b', type='float'),
            Property(name='c', type='float')
        ]
    )
)

# Define a function to solve linear equations
solve_linear = CompletionModelFunction(
    name='solve_linear',
    parameters=FunctionParameters(
        properties=[
            Property(name='m', type='float'),
            Property(name='b', type='float'),
            Property(name='x', type='float')
        ]
    )
)
```
[Source File](https://github.com/ph-ausseil/afaas/blob/5as-autogpt-integration/autogpts/autogpt/autogpt/core/core/resource/model_providers/openai.py)

## 3. Assembling The Request with `ChatPrompt`

The `ChatPrompt` is constructed using the messages and functions defined earlier. This prompt will be sent to the LLM.

```python
from AFAAS.core.core.resource.model_providers.chat_schema import ChatPrompt

# Build the chat prompt
chat_prompt = ChatPrompt(
    messages=[user_message, assistant_message1],
    functions=[solve_quad, solve_linear]
)
```

## 4. Strategizing with `PromptStrategy`

Upon receiving the LLM's response, `PromptStrategy` aids in parsing and understanding the response to determine the next course of action.

```python
# (Within a PromptStrategy method)
parsed_result = self.parse_prompt_response(llm_response)
```
[Source File](https://github.com/ph-ausseil/afaas/blob/5as-autogpt-integration/autogpts/autogpt/autogpt/core/agents/usercontext/strategies/refine_user_context.py)

## 5. Harnessing the Response with `ChatModelResponse` in `loop.py`

Lastly, `ChatModelResponse` in `loop.py` encapsulates the LLM's response, enabling the agent to act based on the parsed results.

```python
# (Within UserContextLoop.run method in loop.py)
model_response : ChatModelResponse = await self.execute_strategy(
    strategy_name="refine_user_context",
    user_objective=user_objectives,
)
```
[Source File](https://raw.githubusercontent.com/ph-ausseil/afaas/5as-autogpt-integration/autogpts/autogpt/autogpt/core/agents/usercontext/loop.py)

In this walkthrough, we have woven the chat completion classes within a narrative that mirrors a real-world interaction between a user and an agent. Through each step, we have seen how these classes play pivotal roles in orchestrating a coherent and meaningful dialogue, steering towards a helpful resolution for the user's request.
