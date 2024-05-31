# Creating Components

## The minimal component

Components can be used to implement various functionalities like providing messages to the prompt, executing code, or interacting with external services.

*Component* is a class that inherits from `AgentComponent` OR implements one or more *protocols*. Every *protocol* inherits `AgentComponent`, so your class automatically becomes a *component* once you inherit any *protocol*.

```py
class MyComponent(AgentComponent):
    pass
```

This is already a valid component, but it doesn't do anything yet. To add some functionality to it, you need to implement one or more *protocols*.

Let's create a simple component that adds "Hello World!" message to the agent's prompt. To do this we need to implement `MessageProvider` *protocol* in our component. `MessageProvider` is an interface with `get_messages` method:

```py
# No longer need to inherit AgentComponent, because MessageProvider already does it
class HelloComponent(MessageProvider):
    def get_messages(self) -> Iterator[ChatMessage]:
        yield ChatMessage.user("Hello World!")
```

Now we can add our component to an existing agent or create a new Agent class and add it there:

```py
class MyAgent(Agent):
    self.hello_component = HelloComponent()
```

`get_messages` will called by the agent each time it needs to build a new prompt and the yielded messages will be added accordingly.  

## Passing data to and between components

Since components are regular classes you can pass data (including other components) to them via the `__init__` method.
For example we can pass a config object and then retrieve an API key from it when needed:

```py
class ConfigurableComponent(MessageProvider):
    def __init__(self, config: Config):
        self.config = config

    def get_messages(self) -> Iterator[ChatMessage]:
        if self.config.openai_credentials.api_key:
            yield ChatMessage.system("API key found!")
        else:
            yield ChatMessage.system("API key not found!")
```

!!! note
    Component-specific configuration handling isn't implemented yet.

## Providing commands

To extend what an agent can do, you need to provide commands using `CommandProvider` protocol. For example to allow agent to multiply two numbers, you can create a component like this:

```py
class MultiplicatorComponent(CommandProvider):
    def get_commands(self) -> Iterator[Command]:
        # Yield the command so the agent can use it
        yield self.multiply

    @command(
    parameters={
        "a": JSONSchema(
            type=JSONSchema.Type.INTEGER,
            description="The first number",
            required=True,
        ),
        "b": JSONSchema(
            type=JSONSchema.Type.INTEGER,
            description="The second number",
            required=True,
        )})
    def multiply(self, a: int, b: int) -> str:
        """
        Multiplies two numbers.
        
        Args:
            a: First number
            b: Second number

        Returns:
            Result of multiplication
        """
        return str(a * b)
```

To learn more about commands see [🛠️ Commands](./commands.md).

## Prompt structure

After components provided all necessary data, the agent needs to build the final prompt that will be send to a llm.
Currently, `PromptStrategy` (*not* a protocol) is responsible for building the final prompt.

If you want to change the way the prompt is built, you need to create a new `PromptStrategy` class, and then call relevant methods in your agent class.
You can have a look at the default strategy used by the AutoGPT Agent: [OneShotAgentPromptStrategy](https://github.com/Significant-Gravitas/AutoGPT/tree/master/autogpt/autogpt/agents/prompt_strategies/one_shot.py), and how it's used in the [Agent](https://github.com/Significant-Gravitas/AutoGPT/tree/master/autogpt/autogpt/agents/agent.py) (search for `self.prompt_strategy`).

## Example `UserInteractionComponent`

Let's create a slightly simplified version of the component that is used by the built-in agent.
It gives an ability for the agent to ask user for input in the terminal.

1. Create a class for the component that inherits from `CommandProvider`.

    ```py
    class MyUserInteractionComponent(CommandProvider):
        """Provides commands to interact with the user."""
        pass
    ```

2. Implement command method that will ask user for input and return it.

    ```py
    def ask_user(self, question: str) -> str:
        """If you need more details or information regarding the given goals,
        you can ask the user for input."""
        print(f"\nQ: {question}")
        resp = input("A:")
        return f"The user's answer: '{resp}'"
    ```

3. The command needs to be decorated with `@command`.

    ```py
    @command(
        parameters={
            "question": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The question or prompt to the user",
                required=True,
            )
        },
    )
    def ask_user(self, question: str) -> str:
        """If you need more details or information regarding the given goals,
        you can ask the user for input."""
        print(f"\nQ: {question}")
        resp = input("A:")
        return f"The user's answer: '{resp}'"
    ```

4. We need to implement `CommandProvider`'s `get_commands` method to yield the command.

    ```py
    def get_commands(self) -> Iterator[Command]:
        yield self.ask_user
    ```

5. Since agent isn't always running in the terminal or interactive mode, we need to disable this component by setting `self._enabled` when it's not possible to ask for user input.

    ```py
    def __init__(self, config: Config):
        self.config = config
        self._enabled = not config.noninteractive_mode
    ```

The final component should look like this:

```py
# 1.
class MyUserInteractionComponent(CommandProvider):
    """Provides commands to interact with the user."""

    # We pass config to check if we're in noninteractive mode
    def __init__(self, config: Config):
        self.config = config
        # 5.
        self._enabled = not config.noninteractive_mode

    # 4.
    def get_commands(self) -> Iterator[Command]:
        # Yielding the command so the agent can use it
        # This won't be yielded if the component is disabled
        yield self.ask_user

    # 3.
    @command(
        # We need to provide a schema for ALL the command parameters
        parameters={
            "question": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The question or prompt to the user",
                required=True,
            )
        },
    )
    # 2.
    # Command name will be its method name and description will be its docstring
    def ask_user(self, question: str) -> str:
        """If you need more details or information regarding the given goals,
        you can ask the user for input."""
        print(f"\nQ: {question}")
        resp = input("A:")
        return f"The user's answer: '{resp}'"
```

Now if we want to use our user interaction *instead of* the default one we need to somehow remove the default one (if our agent inherits from `Agent` the default one is inherited) and add our own. We can simply override the `user_interaction` in `__init__` method:

```py
class MyAgent(Agent):
    def __init__(
        self,
        settings: AgentSettings,
        llm_provider: MultiProvider,
        file_storage: FileStorage,
        legacy_config: Config,
    ):
        # Call the parent constructor to bring in the default components
        super().__init__(settings, llm_provider, file_storage, legacy_config)
        # Disable the default user interaction component by overriding it
        self.user_interaction = MyUserInteractionComponent()
```

Alternatively we can disable the default component by setting it to `None`:

```py
class MyAgent(Agent):
    def __init__(
        self,
        settings: AgentSettings,
        llm_provider: MultiProvider,
        file_storage: FileStorage,
        legacy_config: Config,
    ):
        # Call the parent constructor to bring in the default components
        super().__init__(settings, llm_provider, file_storage, legacy_config)
        # Disable the default user interaction component
        self.user_interaction = None
        # Add our own component
        self.my_user_interaction = MyUserInteractionComponent(legacy_config)
```

## Learn more

The best place to see more examples is to look at the built-in components in the [autogpt/components](https://github.com/Significant-Gravitas/AutoGPT/tree/master/autogpt/autogpt/components/) and [autogpt/commands](https://github.com/Significant-Gravitas/AutoGPT/tree/master/autogpt/autogpt/commands/) directories.

Guide on how to extend the built-in agent and build your own: [🤖 Agents](./agents.md)  
Order of some components matters, see [🧩 Components](./components.md) to learn more about components and how they can be customized.  
To see built-in protocols with accompanying examples visit [⚙️ Protocols](./protocols.md).
