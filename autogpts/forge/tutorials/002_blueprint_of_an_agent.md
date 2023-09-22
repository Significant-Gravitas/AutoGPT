# AutoGPT Forge: The Blueprint of an AI Agent

Welcome back, fellow AI enthusiasts! In our first tutorial, we got our hands dirty setting up the project and learning how to stop and start our agents. Now, it's time to dive deeper. In this second tutorial of our series, we're going to dissect an agent, understanding its key components. We'll take a tour of the project structure and then roll up our sleeves to update the step function. By the end of this tutorial, you'll have an LLM Powered AI that can pass the write file test. So, let's get started!

## What are LLM-Based AI Agents?

Large Language Models (LLMs) are state-of-the-art machine learning models that harness vast amounts of web knowledge. But what happens when you blend these LLMs with autonomous agents? You get LLM-based AI agents - a new breed of artificial intelligence that promises more human-like decision-making.

Traditional autonomous agents operated with limited knowledge, often confined to specific tasks or environments. They were like calculators - efficient but limited to predefined functions. LLM-based agents, on the other hand, are akin to having an encyclopedia combined with a calculator. They don't just compute; they understand, reason, and then act, drawing from a vast reservoir of information.

The [Agent Landscape Survey](https://arxiv.org/abs/2308.11432) underscores this evolution, detailing the remarkable potential LLMs have shown in achieving human-like intelligence. They're not just about more data; they represent a more holistic approach to AI, bridging gaps between isolated task knowledge and expansive web information.

Further expanding on this, [The Rise and Potential of Large Language Model Based Agents: A Survey](https://arxiv.org/abs/2309.07864) portrays LLMs as the foundational blocks for the next generation of AI agents. These agents sense, decide, and act, all backed by the comprehensive knowledge and adaptability of LLMs. It is an incrediable source of knowledge on AI Agent Research with almost 700 papers referenced and organised by reseach area.

## Bridging Communication Gaps with the Agent Protocol

In the burgeoning field of AI agents, developers often find themselves forging unique paths, creating agents with distinctive characteristics. While this approach nurtures innovation, it also presents a significant challenge: establishing seamless communication between various agents, each equipped with a different interface. Furthermore, the absence of a standardized communication platform impedes the easy comparison of agents and the seamless development of universal devtools.

To tackle this challenge head-on, the AI Foundation has introduced the **Agent Protocol**, a unified communication interface designed to spur innovation and integration in agent development.

### A Unifying Communication Interface

The Agent Protocol emerges as a harmonizing force in the fragmented world of agent development, offering a well-defined API specification that dictates the endpoints agents should expose, along with standardized input and response models. What sets this protocol apart is its versatility, welcoming agents developed with various frameworks to adopt it seamlessly.

A glimpse into the protocol structure reveals:

- **POST /agent/tasks**: A route designated for task creation.
- **POST /agent/tasks/{id}/steps**: A route purposed for initiating the subsequent step of a task.
- **POST /agent/tasks/{id}/artifacts**: A route for creating an artifact associated with a task.
- **GET /agent/tasks/{id}/artifacts/{artifact_id}**: A route for downloading an artifact associated with a task.


For an in-depth exploration, visit the [Agent Protocol](https://agentprotocol.ai).

### Advantages of Adopting the Agent Protocol

Implementing the Agent Protocol offers a myriad of benefits, simplifying the development process substantially. Here are some noteworthy advantages:

- **Effortless Benchmarking**: Seamlessly integrate with benchmarking tools such as Agent Evals, facilitating straightforward testing and benchmarking of your agent against others.
- **Enhanced Integration and Collaboration**: Encourage seamless integration and collaboration, fostering a community of shared ideas and advancements.
- **General Devtools Development**: Enable the creation of universal devtools, streamlining development, deployment, and monitoring processes.
- **Focused Development**: Shift your focus from boilerplate API creation to core agent development, nurturing innovation and efficiency.

### Fostering a Collaborative Ecosystem

The Agent Protocol stands at the forefront of fostering a collaborative and rapidly evolving ecosystem. With a minimal core as a starting point, the objective is to expand iteratively, incorporating valuable insights from agent developers to meet their evolving needs.

Now, let's delve deeper into the core components that constitute an AI agent.

## Delineating the Anatomy of an AI Agent

To cultivate proficiency in the AI domain, a thorough understanding of the fundamental components forming an AI agent is indispensable. In this section, we elaborate on the cornerstone elements shaping an AI agent:

### Profile: Tailoring the Persona

An agent functions effectively by adopting specific roles, emulating personas such as a teacher, coder, or planner. The strategic utilization of the profile attribute in the language model (LLM) prompt significantly enhances output quality, a phenomenon substantiated by this [study](https://arxiv.org/abs/2305.14688). With the ability to dynamically switch profiles based on the task at hand, agents unlock a world of endless configuration possibilities with various LLMs.

### Memory: The Repository of Experiences

An adept memory system serves as a foundation for the agent to accumulate experiences, evolve, and respond in a consistent and efficient manner. Consider the following critical facets:

- **Long-term and Short-term Memory**: Foster strategies catering to both long-term retention and working memory.
- **Memory Reflection**: Encourage the agent's ability to scrutinize and reassess memories, facilitating the transition of short-term memories into long-term storage.

### Planning: Navigating Complex Tasks

The planning module bestows LLM-based agents with the ability to strategize and plan for intricate tasks, enhancing the agent's comprehensiveness and reliability. Consider integrating these methodologies:

- **Planning with Feedback**: Incorporate feedback mechanisms within the planning phase.
- **Planning without Feedback**: Develop strategies independent of external inputs.

### Abilities: Executing Decisions into Actions

The abilities component represents a pivotal section where the agent's decisions materialize into specific outcomes. Explore diverse approaches to implement actions, amplifying your agent's capabilities.

## Embarking on Your Forge Journey: Template and Layout

To initiate your voyage in AI agent development, begin by modifying the template found in `forge/agent.py`. Here is a foundational structure to kickstart your journey:

```python
from forge.sdk import Agent, AgentDB, Step, StepRequestBody, Workspace

class ForgeAgent(Agent):
 
    def __init__(self, database: AgentDB, workspace: Workspace):
        """
        The database is utilized to store tasks, steps, and artifact metadata, while the workspace is used for storing artifacts, represented as a directory on the filesystem. Feel free to create subclasses of the database and workspace to implement your own storage solutions.
        """
        super().__init__(database, workspace)

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        # An example that passes the write file test
        self.workspace.write(task_id=task_id, path="output.txt", data=b"Washington D.C")
        step = await self.db.create_step(
            task_id=task_id, input=step_request, is_last=True
        )
        artifact = await self.db.create_artifact(
            task_id=task_id,
            step_id=step.step_id,
            file_name="output.txt",
            relative_path="",
            agent_created=True,
        )
        step.output = "Washington D.C"

        return step
```

### Exploring the Forge Layout

Within the Forge layout, discover a plethora of folders and protocols essential for crafting a proficient agent:

- **Abilities Folder**: Houses the abilities component, a critical aspect defining the agent's capabilities. Path: `forge/sdk/abilities/`
- **Agent Protocol**: A central pillar of the Forge, overseeing task creation and execution processes. This can be found in `forge/sdk/routes/agent_protocol.py`
- **Schema**: Outlines the structure and regulations governing data within the Forge. Path: `forge/sdk/schema.py`
- **DB**: Core component entrusted with managing database operations. Path: `forge/sdk/db.py`
- **Memstore**: Component responsible for managing the memory system of the agent. Path: `forge/sdk/memory/memstore.py`
- **AI_(X)**: these files have examples of how the respective functionality can be implemented
- **Prompt Templates**: The Forge uses Jinja2-based prompt templates, allowing for easy modification of prompts without changing the code. These templates are stored in the `forge/prompts/` directory. This approach provides flexibility and ease of use in customizing the agent's prompts based on specific tasks or roles.


Moreover, the Forge initiates a FastAPI server, simplifying the process of serving the frontend on [http://localhost:8000](http://localhost:8000).

## Conclusion

Embarking on the AI agent development journey with the Forge promises not only an enriching learning experience but also a streamlined development journey. As you progress, immerse yourself in the vibrant landscape of AI agent development, leveraging the comprehensive tools and resources at your disposal.

Happy Developing!