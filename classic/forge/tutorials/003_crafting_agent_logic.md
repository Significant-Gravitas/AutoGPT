# AutoGPT Forge: Crafting Intelligent Agent Logic

![Header](..%2F..%2F..%2Fdocs/content/imgs/quickstart/t3_01.png)
**By Craig Swift & [Ryan Brandt](https://github.com/paperMoose)**

Hey there! Ready for part 3 of our AutoGPT Forge tutorial series? If you missed the earlier parts, catch up here:

- [Getting Started](001_getting_started.md)
- [Blueprint of an Agent](002_blueprint_of_an_agent.md)

Now, let's get hands-on! We'll use an LLM to power our agent and complete a task. The challenge? Making the agent write "Washington" to a .txt file. We won't give it step-by-step instructions—just the task. Let's see our agent in action and watch it figure out the steps on its own!


## Get Your Smart Agent Project Ready

Make sure you've set up your project and created an agent as described in our initial guide. If you skipped that part, [click here](#) to get started. Once you're done, come back, and we'll move forward.

In the image below, you'll see my "SmartAgent" and the agent.py file inside the 'forge' folder. That's where we'll be adding our LLM-based logic. If you're unsure about the project structure or agent functions from our last guide, don't worry. We'll cover the basics as we go!

![SmartAgent](..%2F..%2F..%2Fdocs/content/imgs/quickstart/t3_02.png)

---

## The Task Lifecycle

The lifecycle of a task, from its creation to execution, is outlined in the agent protocol. In simple terms: a task is initiated, its steps are systematically executed, and it concludes once completed.

Want your agent to perform an action? Start by dispatching a create_task request. This crucial step involves specifying the task details, much like how you'd send a prompt to ChatGPT, using the input field. If you're giving this a shot on your own, the UI is your best friend; it effortlessly handles all the API calls on your behalf.

When the agent gets this, it runs the create_task function. The code `super().create_task(task_request)` takes care of protocol steps. It then logs the task's start. For this guide, you don't need to change this function.

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

After starting a task, the `execute_step` function runs until all steps are done. Here's a basic view of `execute_step`. I've left out the detailed comments for simplicity, but you'll find them in your project.

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

Here's the breakdown of the 'write file' process in four steps:

1. **Database Step Creation**: The first stage is all about creating a step within the database, an essential aspect of the agent protocol. You'll observe that while setting up this step, we've flagged it with `is_last=True`. This signals to the agent protocol that no more steps are pending. For the purpose of this guide, let's work under the assumption that our agent will only tackle single-step tasks. However, hang tight for future tutorials, where we'll level up and let the agent determine its completion point.

2. **File Writing**: Next, we pen down "Washington D.C." using the workspace.write function.

3. **Artifact Database Update**: After writing, we record the file in the agent's artifact database.

4. **Step Output & Logging**: Finally, we set the step output to match the file content, log the executed step, and use the step object.

With the 'write file' process clear, let's make our agent smarter and more autonomous. Ready to dive in?

---

## Building the Foundations For Our Smart Agent

First, we need to update the `execute_step()` function. Instead of a fixed solution, it should use the given request.

To do this, we'll fetch the task details using the provided `task_id`:

```python
task = await self.db.get_task(task_id)
```

Next, remember to create a database record and mark it as a single-step task with `is_last=True`:

```python
step = await self.db.create_step(
    task_id=task_id, input=step_request, is_last=True
)
```

Your updated `execute_step` function will look like this:

```python
async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
    # Get the task details
    task = await self.db.get_task(task_id)

    # Add a new step to the database
    step = await self.db.create_step(
        task_id=task_id, input=step_request, is_last=True
    )
    return step
```

Now that we've set this up, let's move to the next exciting part: The PromptEngine.

---


**The Art of Prompting**  

![Prompting 101](..%2F..%2F..%2Fdocs/content/imgs/quickstart/t3_03.png)

Prompting is like shaping messages for powerful language models like ChatGPT. Since these models respond to input details, creating the right prompt can be a challenge. That's where the **PromptEngine** comes in.

The "PromptEngine" helps you store prompts in text files, specifically in Jinja2 templates. This means you can change the prompts without changing the code. It also lets you adjust prompts for different LLMs. Here's how to use it:

First, add the PromptEngine from the SDK:

```python
from .sdk import PromptEngine
```

In your `execute_step` function, set up the engine for the `gpt-3.5-turbo` LLM:

```python
prompt_engine = PromptEngine("gpt-3.5-turbo")
```

Loading a prompt is straightforward. For instance, loading the `system-format` prompt, which dictates the response format from the LLM, is as easy as:

```python
system_prompt = prompt_engine.load_prompt("system-format")
```

For intricate use cases, like the `task-step` prompt which requires parameters, employ the following method:

```python
# Define the task parameters
task_kwargs = {
    "task": task.input,
    "abilities": self.abilities.list_abilities_for_prompt(),
}

# Load the task prompt with those parameters
task_prompt = prompt_engine.load_prompt("task-step", **task_kwargs)
```



Delving deeper, let's look at the `task-step` prompt template in `prompts/gpt-3.5-turbo/task-step.j2`:

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

{% if abilities %}
## Abilities
Use these abilities:
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

This template is modular. It uses the `extends` directive to build on the `expert.j2` template. The different sections like constraints, resources, abilities, and best practices make the prompt dynamic. It guides the LLM in understanding the task and using resources and abilities.

The PromptEngine equips us with a potent tool to converse seamlessly with large language models. By externalizing prompts and using templates, we can ensure that our agent remains agile, adapting to new challenges without a code overhaul. As we march forward, keep this foundation in mind—it's the bedrock of our agent's intelligence.

---

## Engaging with your LLM

To make the most of the LLM, you'll send a series of organized instructions, not just one prompt. Structure your prompts as a list of messages for the LLM. Using the `system_prompt` and `task_prompt` from before, create the `messages` list:

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": task_prompt}
]
```

With the prompt set, send it to the LLM. This step involves foundational code, focusing on the `chat_completion_request`. This function gives the LLM your prompt, and then gets the LLM's output. The other code sets up our request and interprets the feedback:

```python
try:
    # Set the parameters for the chat completion
    chat_completion_kwargs = {
        "messages": messages,
        "model": "gpt-3.5-turbo",
    }
    # Get the LLM's response and interpret it
    chat_response = await chat_completion_request(**chat_completion_kwargs)
    answer = json.loads(chat_response.choices[0].message.content)

    # Log the answer for reference
    LOG.info(pprint.pformat(answer))

except json.JSONDecodeError as e:
    # Handle JSON decoding errors
    LOG.error(f"Can't decode chat response: {chat_response}")
except Exception as e:
    # Handle other errors
    LOG.error(f"Can't get chat response: {e}")
```

Extracting clear messages from LLM outputs can be complex. Our method is simple and works with GPT-3.5 and GPT-4. Future guides will show more ways to interpret LLM outputs. The goal? To go beyond JSON, as some LLMs work best with other response types. Stay tuned!

---


## Using and Creating Abilities

Abilities are the gears and levers that enable the agent to interact with tasks at hand. Let's unpack the mechanisms behind these abilities and how you can harness, and even extend, them.

In the Forge folder, there's a `actions` folder containing `registry.py`, `finish.py`, and a `file_system` subfolder. You can also add your own abilities here. `registry.py` is the main file for abilities. It contains the `@action` decorator and the `ActionRegister` class. This class actively tracks abilities and outlines their function. The base Agent class includes a default Action register available via `self.abilities`. It looks like this:

```python
self.abilities = ActionRegister(self)
```

The `ActionRegister` has two key methods. `list_abilities_for_prompt` prepares abilities for prompts. `run_action` makes the ability work. An ability is a function with the `@action` decorator. It must have specific parameters, including the agent and `task_id`.

```python
@action(
    name="write_file",
    description="Write data to a file",
    parameters=[
        {
            "name": "file_path",
            "description": "Path to the file",
            "type": "string",
            "required": True,
        },
        {
            "name": "data",
            "description": "Data to write to the file",
            "type": "bytes",
            "required": True,
        },
    ],
    output_type="None",
)
async def write_file(agent, task_id: str, file_path: str, data: bytes) -> None:
    pass
```

The `@action` decorator defines the ability's details, like its identity (name), functionality (description), and operational parameters.

## Example of a Custom Ability: Webpage Fetcher

```python
import requests

@action(
  name="fetch_webpage",
  description="Retrieve the content of a webpage",
  parameters=[
      {
          "name": "url",
          "description": "Webpage URL",
          "type": "string",
          "required": True,
      }
  ],
  output_type="string",
)
async def fetch_webpage(agent, task_id: str, url: str) -> str:
  response = requests.get(url)
  return response.text
```

This ability, `fetch_webpage`, accepts a URL as input and returns the HTML content of the webpage as a string. Custom abilities let you add more features to your agent. They can integrate other tools and libraries to enhance its functions. To make a custom ability, you need to understand the structure and add technical details. With abilities like "fetch_webpage", your agent can handle complex tasks efficiently.

## Running an Ability

Now that you understand abilities and how to create them, let's use them. The last piece is the `execute_step` function. Our goal is to understand the agent's response, find the ability, and use it. 

First, we get the ability details from the agent's answer:

```python
# Extract the ability from the answer
ability = answer["ability"]
```

With the ability details, we use it. We call the `run_ability` function:

```python
# Run the ability and get the output
# We don't actually use the output in this example
output = await self.abilities.run_action(
    task_id, ability["name"], **ability["args"]
)
```

Here, we’re invoking the specified ability. The task_id ensures continuity, ability['name'] pinpoints the exact function, and the arguments (ability["args"]) provide necessary context.

Finally, we make the step's output show the agent's thinking:

```python
# Set the step output to the "speak" part of the answer
step.output = answer["thoughts"]["speak"]

# Return the completed step
return step
```

And there you have it! Your first Smart Agent, sculpted with precision and purpose, stands ready to take on challenges. The stage is set. It’s showtime!

Here is what your function should look like:

```python
async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
    # Firstly we get the task this step is for so we can access the task input
    task = await self.db.get_task(task_id)

    # Create a new step in the database
    step = await self.db.create_step(
        task_id=task_id, input=step_request, is_last=True
    )

    # Log the message
    LOG.info(f"\t✅ Final Step completed: {step.step_id} input: {step.input[:19]}")

    # Initialize the PromptEngine with the "gpt-3.5-turbo" model
    prompt_engine = PromptEngine("gpt-3.5-turbo")

    # Load the system and task prompts
    system_prompt = prompt_engine.load_prompt("system-format")

    # Initialize the messages list with the system prompt
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    # Define the task parameters
    task_kwargs = {
        "task": task.input,
        "abilities": self.abilities.list_abilities_for_prompt(),
    }

    # Load the task prompt with the defined task parameters
    task_prompt = prompt_engine.load_prompt("task-step", **task_kwargs)

    # Append the task prompt to the messages list
    messages.append({"role": "user", "content": task_prompt})

    try:
        # Define the parameters for the chat completion request
        chat_completion_kwargs = {
            "messages": messages,
            "model": "gpt-3.5-turbo",
        }
        # Make the chat completion request and parse the response
        chat_response = await chat_completion_request(**chat_completion_kwargs)
        answer = json.loads(chat_response.choices[0].message.content)

        # Log the answer for debugging purposes
        LOG.info(pprint.pformat(answer))

    except json.JSONDecodeError as e:
        # Handle JSON decoding errors
        LOG.error(f"Unable to decode chat response: {chat_response}")
    except Exception as e:
        # Handle other exceptions
        LOG.error(f"Unable to generate chat response: {e}")

    # Extract the ability from the answer
    ability = answer["ability"]

    # Run the ability and get the output
    # We don't actually use the output in this example
    output = await self.abilities.run_action(
        task_id, ability["name"], **ability["args"]
    )

    # Set the step output to the "speak" part of the answer
    step.output = answer["thoughts"]["speak"]

    # Return the completed step
    return step
```

## Interacting with your Agent
> ⚠️ Heads up: The UI and benchmark are still in the oven, so they might be a tad glitchy.

With the heavy lifting of crafting our Smart Agent behind us, it’s high time to see it in action. Kick things off by firing up the agent with this command:
```bash
./run agent start SmartAgent.
```

Once your digital playground is all set, your terminal should light up with:
```bash


       d8888          888             .d8888b.  8888888b. 88888888888 
      d88888          888            d88P  Y88b 888   Y88b    888     
     d88P888          888            888    888 888    888    888     
    d88P 888 888  888 888888 .d88b.  888        888   d88P    888     
   d88P  888 888  888 888   d88""88b 888  88888 8888888P"     888     
  d88P   888 888  888 888   888  888 888    888 888           888     
 d8888888888 Y88b 888 Y88b. Y88..88P Y88b  d88P 888           888     
d88P     888  "Y88888  "Y888 "Y88P"   "Y8888P88 888           888     
                                                                      
                                                                      
                                                                      
                8888888888                                            
                888                                                   
                888                                                   
                8888888  .d88b.  888d888 .d88b.   .d88b.              
                888     d88""88b 888P"  d88P"88b d8P  Y8b             
                888     888  888 888    888  888 88888888             
                888     Y88..88P 888    Y88b 888 Y8b.                 
                888      "Y88P"  888     "Y88888  "Y8888              
                                             888                      
                                        Y8b d88P                      
                                         "Y88P"                v0.2.0


[2023-09-27 15:39:07,832] [forge.sdk.agent] [INFO]      📝  Agent server starting on http://localhost:8000

```
1. **Get Started**
   - Click the link to access the AutoGPT Agent UI.

2. **Login**
   - Log in using your Gmail or Github credentials.

3. **Navigate to Benchmarking**
   - Look to the left, and you'll spot a trophy icon. Click it to enter the benchmarking arena.
  
![Benchmarking page of the AutoGPT UI](..%2F..%2F..%2Fdocs/content/imgs/quickstart/t3_04.png)

4. **Select the 'WriteFile' Test**
   - Choose the 'WriteFile' test from the available options.

5. **Initiate the Test Suite**
   - Hit 'Initiate test suite' to start the benchmarking process.

6. **Monitor in Real-Time**
   - Keep your eyes on the right panel as it displays real-time output.

7. **Check the Console**
   - For additional information, you can also monitor your console for progress updates and messages.
```bash
📝  📦 Task created: 70518b75-0104-49b0-923e-f607719d042b input: Write the word 'Washington' to a .txt fi...
📝      ✅ Final Step completed: a736c45f-65a5-4c44-a697-f1d6dcd94d5c input: y
```
If you see this, you've done it!

8. **Troubleshooting**
   - If you encounter any issues or see cryptic error messages, don't worry. Just hit the retry button. Remember, LLMs are powerful but may occasionally need some guidance.

## Wrap Up
- Stay tuned for our next tutorial, where we'll enhance the agent's capabilities by adding memory!

## Keep Exploring
- Keep experimenting and pushing the boundaries of AI. Happy coding! 🚀

## Wrap Up
In our next tutorial, we’ll further refine this process, enhancing the agent’s capabilities, through the addition of memory!

Until then, keep experimenting and pushing the boundaries of AI. Happy coding! 🚀
