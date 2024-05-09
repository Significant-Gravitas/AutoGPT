# 🤖 Agents

Agent is composed of [🧩 Components](./components.md) and responsible for executing pipelines and some additional logic. The base class for all agents is `BaseAgent`, it has the necessary logic to collect components and execute protocols.

## Important methods

`BaseAgent` provides two abstract methods needed for any agent to work properly:
1. `propose_action`: This method is responsible for proposing an action based on the current state of the agent, it returns `ThoughtProcessOutput`.
2. `execute`: This method is responsible for executing the proposed action, returns `ActionResult`.

## AutoGPT Agent

`Agent` is the main agent provided by AutoGPT. It's a subclass of `BaseAgent`. It has all the [Built-in Components](./built-in-components.md). `Agent` implements the essential abstract methods from `BaseAgent`: `propose_action` and `execute`.

## Building your own Agent

The easiest way to build your own agent is to extend the `Agent` class and add additional components. By doing this you can reuse the existing components and the default logic for executing [⚙️ Protocols](./protocols.md).

```py
class MyComponent(AgentComponent):
    pass

class MyAgent(Agent):
    def __init__(
        self,
        settings: AgentSettings,
        llm_provider: ChatModelProvider,
        file_storage: FileStorage,
        legacy_config: Config,
    ):
        # Call the parent constructor to bring in the default components
        super().__init__(settings, llm_provider, file_storage, legacy_config)
        # Add your custom component
        self.my_component = MyComponent()
```

For more customization, you can override the `propose_action` and `execute` or even subclass `BaseAgent` directly. This way you can have full control over the agent's components and behavior. Have a look at the [implementation of Agent](https://github.com/Significant-Gravitas/AutoGPT/tree/master/autogpts/autogpt/autogpt/agents/agent.py) for more details.
