# AutoGPT Forge: Crafting Intelligent Agent Logic

![Header](../../../docs/content/imgs/quickstart/t3_01.png)

Greetings, AI enthusiasts! Today, we're about to embark on an enlightening journey of crafting intelligent agent logic. This is part 3 in a tutorial series on using the AutoGPT Forge, you can find the earlier parts here:

Part 1: AutoGPT Forge: A Comprehensive Guide to Your First Step
Part 2: AutoGPT Forge: The Blueprint of an AI Agent

Alright, folks, let's dive right into the fun part: coding! We're about to set up a nifty system that showcases how to use an LLM as the brainpower behind our agent. The mission? To tackle the simple task of jotting down the capital of the United States into a txt file. The coolest part? We won't spoon-feed our agent the steps. Instead, we'll just hand over the task: "Write the word 'Washington' to a .txt file," and watch in awe as it figures out the 'how-to' all by itself, then swiftly executes the necessary commands. How cool is that?

---
## Setting Up Your Smart Agent Project

Before diving in, ensure you've prepped your project and crafted an agent as detailed in our kick-off tutorial. Missed that step? No worries! Just hop over to the project setup by clicking here. Once you're all set, come back and we'll hit the ground running.
In the following screenshot, you'll notice I've crafted an agent named "SmartAgent" and then accessed the agent.py file located in the 'forge' subfolder. This will be our workspace for integrating the LLM-driven logic. While our previous tutorial touched upon the project layout and agent operations, don't fret! I'll highlight the essentials as we delve into the logic implementation.

---

## The Task Lifecycle

The lifecycle of a task, from its creation to execution, is outlined in the agent protocol. In simple terms: a task is initiated, its steps are systematically executed, and it concludes once completed.

Want your agent to perform an action? Start by dispatching a create_task request. This crucial step involves specifying the task details, much like how you'd send a prompt to ChatGPT, using the input field. If you're giving this a shot on your own, the UI is your best friend; it effortlessly handles all the API calls on your behalf.

Once your agent receives this, it triggers the create_task function. The method super().create_task(task_request) effortlessly manages all the requisite protocol record keeping on your behalf. Subsequently, it simply logs  the task's creation. For the scope of this tutorial, there's no need to tweak this function.

```python
async def create_task(self, task_request: TaskRequestBody) -> Task:
    """
    The agent protocol, which is the core of the Forge, works by creating a task and then
    executing steps for that task. This method is called when the agent is asked to create
    a task.

    We are hooking into function to add a custom log message. Though you can do anything you
    want here.
    """
    task = await super().create_task(task_request)
    LOG.info(
        f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
    )
    return task
```

Once a task is initiated, the execute_step function is invoked repeatedly until the very last step is executed. Below is the initial look of the execute_step, and note that I've omitted the lengthy docstring explanation for the sake of brevity, but you'll encounter it in your project.

```python
async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
    # An example that
      step = await self.db.create_step(
          task_id=task_id, input=step_request, is_last=True
      )

      self.workspace.write(task_id=task_id, path="output.txt", data=b"Washington D.C")

      await self.db.create_artifact(
          task_id=task_id,
          step_id=step.step_id,
          file_name="output.txt",
          relative_path="",
          agent_created=True,
      )
      
      step.output = "Washington D.C"

      LOG.info(f"\t✅ Final Step completed: {step.step_id}")

      return step
```

Here's what you're witnessing: a clever way to pass the 'write file' test, broken down into four clear-cut stages:

1. Database Step Creation: The first stage is all about creating a step within the database, an essential aspect of the agent protocol. You'll observe that while setting up this step, we've flagged it with is_last=True. This signals to the agent protocol that no more steps are pending. For the purpose of this guide, let's work under the assumption that our agent will only tackle single-step tasks. However, hang tight for future tutorials, where we'll level up and let the agent determine its completion point.

2. File Writing: Next, we pen down "Washington D.C." using the workspace.write function. Simple, right?

3. Artifact Database Update: Once the file is written, it's time to record this file in the agent's artifact database, ensuring everything's documented.

4. Step Output Setting & Logging: To wrap things up, we align the step output with what we've penned in the file, jot down in the logs that our step has been executed, and then bring the step object into play.

Now that we've demystified the process to ace the 'write file' test, it's time to crank things up a notch. Let's mold this into a truly intelligent agent, empowering it to navigate and conquer the challenge autonomously. Ready to dive in?

---

## Building the Foundations For Our Smart Agent

Alright, first order of business: Let's purge that cheeky excuse_step function of its deceptive logic and lay the groundwork for our brainy agent. Remember, when our execute_step function gets the call, it's initially clueless about the specific task at hand. So, our initial task is to rectify this.

To bridge this knowledge gap, we'll summon the task details using the task_id provided. Here's the code magic to make it happen:

```python
task = await self.db.get_task(task_id)
Additionally, we're not forgetting the crucial step of creating a database record. As we did previously, we'll emphasize this is a one-off task with is_last=True:
step = await self.db.create_step(
    task_id=task_id, input=step_request, is_last=True
)
```

With these additions, your execute_step function should now have a minimalistic yet essential structure:

```python
async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
    # Firstly we get the task this step is for so we can access the task input
    task = await self.db.get_task(task_id)

    # Create a new step in the database
    step = await self.db.create_step(
        task_id=task_id, input=step_request, is_last=True
    )
    return step
```

With these foundational bricks laid down, let's plunge into something truly fascinating: introducing, The PromptEngine.

---

**The Art of Prompting**  
Prompting is akin to a craftsman meticulously shaping messages tailored for powerful language models like ChatGPT. With these models being highly attuned to input nuances, designing the perfect prompt to elicit awe-inspiring behavior can be a labyrinthine challenge. Enter: the **PromptEngine**.

While "PromptEngine" might sound high-brow, its essence is elegantly simple. It lets you store your prompts in text files or, to be precise, in Jinja2 templates. The advantage? You can refine the prompts given to your agent without diving into the code. Plus, it offers the flexibility to customize prompts for specific LLMs. Let's break this down.

Firstly, integrate the PromptEngine from the SDK:

```python
from .sdk import PromptEngine
```

Next, within your `execute_step` function, initialize the engine tailored for, say, the `gpt-3.5-turbo` LLM:

```python
prompt_engine = PromptEngine("gpt-3.5-turbo")
```

Loading a prompt is straightforward. For instance, loading the `system-format` prompt, which dictates the response format from the LLM, is as easy as:

```python
system_prompt = prompt_engine.load_prompt("system-format")
```

For intricate use cases, like the `task-step` prompt which requires parameters, employ the following method:

```python
# Specifying the task parameters
task_kwargs = {
    "task": task.input,
    "abilities": self.abilities.list_abilities_for_prompt(),
}

# Then, load the task prompt with the designated parameters
task_prompt = prompt_engine.load_prompt("task-step", **task_kwargs)
```

Delving deeper, let's peek at the `task-step` prompt template, housed at `prompts/gpt-3.5-turbo/task-step.j2`:

```jinja
{% extends "techniques/expert.j2" %}
{% block expert %}Planner{% endblock %}
{% block prompt %}
Your task is:

{{ task }}

Ensure to respond in the given format. Always make autonomous decisions, devoid of user guidance. Harness the power of your LLM, opting for straightforward tactics sans any legal entanglements.

{% if constraints %}
## Constraints
Operate under these confines:
{% for constraint in constraints %}
- {{ constraint }}
{% endfor %}
{% endif %}

{% if resources %}
## Resources
Utilize these resources:
{% for resource in resources %}
- {{ resource }}
{% endfor %}
{% endif %}

{% if abilities %}
## Abilities
Summon these abilities:
{% for ability in abilities %}
- {{ ability }}
{% endfor %}
{% endif %}

{% if best_practices %}
## Best Practices
{% for best_practice in best_practices %}
- {{ best_practice }}
{% endfor %}
{% endif %}
{% endblock %}
```

This template is a marvel of modularity. By using the `extends` directive, it builds upon the base `expert.j2` template. The different blocks – constraints, resources, abilities, and best practices – allow for a dynamic prompt that adjusts based on the context. It's like a conversation blueprint, guiding the LLM to understand the task, abide by constraints, and deploy resources and abilities to achieve the desired outcome.

The PromptEngine equips us with a potent tool to converse seamlessly with large language models. By externalizing prompts and using templates, we can ensure that our agent remains agile, adapting to new challenges without a code overhaul. As we march forward, keep this foundation in mind—it's the bedrock of our agent's intelligence.