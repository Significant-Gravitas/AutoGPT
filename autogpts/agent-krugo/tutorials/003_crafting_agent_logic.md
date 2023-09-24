### AutoGPT Forge: Crafting Intelligent Agent Logic

![Header](../../../docs/content/imgs/quickstart/t3_01.png)

Greetings, AI enthusiasts! Today, we're about to embark on an enlightening journey of crafting intelligent agent logic. We'll transition from a 'shortcut' method in the `execute_step` to a more sophisticated approach, harnessing the full potential of Large Language Models (LLMs). By the end of this tutorial, we aim to get our agent to ace the 'write file' test with flying colors.

So, without further ado, let's dive right in!

### Step 1: Grasping the `execute_step` Essence

The `execute_step` function is the central mechanism that drives our agent's decisions. Whenever an agent is posed with a task, this function determines how the agent responds.

Inputs:
1. **task_id**: A unique identifier for the task.
2. **step_request**: Contains additional instructions or context for the step.

By the end, the agent should provide a well-calculated output based on the given inputs.

### Step 2: Retrieving Task Data

To kick things off, we fetch the task's details:

```python
task = await self.db.get_task(task_id)
```

This helps us access the task's input, which will guide our next steps.

### Step 3: Registering a New Step

Before we proceed, it's essential to log this step in our database:

```python
step = await self.db.create_step(
    task_id=task_id, input=step_request, is_last=True
)
```

By setting `is_last=True`, we're indicating that this is the final step. As our agent logic evolves, this might change to accommodate multiple steps.

### Step 4: Initializing the PromptEngine

The PromptEngine is our interface to the GPT-3.5 model. It facilitates and formats our interactions.

```python
prompt_engine = PromptEngine("gpt-3.5-turbo")
system_prompt = prompt_engine.load_prompt("system-format")
```

### Step 5: Constructing the Message

To harness the power of GPT-3.5, we need to send a series of messages. These messages typically start with a system-level instruction, followed by the task description.

```python
messages = [
    {"role": "system", "content": system_prompt},
]
task_kwargs = {
    "task": task.input,
    "abilities": self.abilities.list_abilities_for_prompt(),
}
task_prompt = prompt_engine.load_prompt("task-step", **task_kwargs)
messages.append({"role": "user", "content": task_prompt})
```

### Step 6: Engaging with GPT-3.5

With our messages ready, we engage with the GPT-3.5 model to derive a solution.

```python
chat_response = await chat_completion_request(**chat_completion_kwargs)
answer = json.loads(chat_response["choices"][0]["message"]["content"])
```

### Step 7: Executing the Derived Ability

From the GPT-3.5's response, we'll determine which ability to execute to solve the task.

```python
ability = answer["ability"]
output = await self.abilities.run_ability(
    task_id, ability["name"], **ability["args"]
)
```

### Step 8: Finalizing the Output

We then set the step's output, which is the agent's final response.

```python
step.output = answer["thoughts"]["speak"]
return step
```

### Wrap Up

Congratulations! You've successfully transitioned from a more rudimentary approach to leveraging the intelligence of Large Language Models in crafting agent logic. This approach not only makes your agent smarter but also equips it to handle a broader range of tasks with efficiency.

In our next tutorial, we'll further refine this process, enhancing the agent's capabilities. Until then, keep experimenting and pushing the boundaries of AI. Happy coding! 🚀