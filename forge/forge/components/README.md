# 🧩 Components

Components are the building blocks of [🤖 Agents](./agents.md). They are classes inheriting `AgentComponent` or implementing one or more [⚙️ Protocols](./protocols.md) that give agent additional abilities or processing. 

Components can be used to implement various functionalities like providing messages to the prompt, executing code, or interacting with external services.
They can be enabled or disabled, ordered, and can rely on each other.

Components assigned in the agent's `__init__` via `self` are automatically detected upon the agent's instantiation.
For example inside `__init__`: `self.my_component = MyComponent()`.
You can use any valid Python variable name, what matters for the component to be detected is its type (`AgentComponent` or any protocol inheriting from it).

Visit [Built-in Components](./built-in-components.md) to see what components are available out of the box.

```py
from forge.agent import BaseAgent
from forge.agent.components import AgentComponent

class HelloComponent(AgentComponent):
    pass

class SomeComponent(AgentComponent):
    def __init__(self, hello_component: HelloComponent):
        self.hello_component = hello_component

class MyAgent(BaseAgent):
    def __init__(self):
        # These components will be automatically discovered and used
        self.hello_component = HelloComponent()
        # We pass HelloComponent to SomeComponent
        self.some_component = SomeComponent(self.hello_component)
```

## Component configuration

Each component can have its own configuration defined using a regular pydantic `BaseModel`.
To ensure the configuration is loaded from the file correctly, the component must inherit from `ConfigurableComponent[T]` where `T` is the configuration model it uses.
`ConfigurableComponent` provides a `config` attribute that holds the configuration instance.
It's possible to either set the `config` attribute directly or pass the configuration instance to the component's constructor.
Extra configuration (i.e. for components that are not part of the agent) can be passed and will be silently ignored. Extra config won't be applied even if the component is added later.
To see the configuration of built-in components visit [Built-in Components](./built-in-components.md).

```py
from pydantic import BaseModel
from forge.agent.components import ConfigurableComponent

class MyConfig(BaseModel):
    some_value: str

class MyComponent(AgentComponent, ConfigurableComponent[MyConfig]):
    def __init__(self, config: MyConfig):
        super().__init__(config)
        # This has the same effect as above:
        # self.config = config

    def get_some_value(self) -> str:
        # Access the configuration like a regular model
        return self.config.some_value
```

### Sensitive information

While it's possible to pass sensitive data directly in code to the configuration it's recommended to use `UserConfigurable(from_env="ENV_VAR_NAME", exclude=True)` field for sensitive data like API keys.
The data will be loaded from the environment variable but keep in mind that value passed in code takes precedence.
All fields, even excluded ones (`exclude=True`) will be loaded when the configuration is loaded from the file.
Exclusion allows you to skip them during *serialization*, non excluded `SecretStr` will be serialized literally as a `"**********"` string.

```py
from pydantic import BaseModel, SecretStr
from forge.models.config import UserConfigurable

class SensitiveConfig(BaseModel):
    api_key: SecretStr = UserConfigurable(from_env="API_KEY", exclude=True)
```

## Ordering components

The execution order of components is important because some may depend on the results of the previous ones.
**By default, components are ordered alphabetically.**

### Ordering individual components

You can order a single component by passing other components (or their types) to the `run_after` method. This way you can ensure that the component will be executed after the specified one.
The `run_after` method returns the component itself, so you can call it when assigning the component to a variable:

```py
class MyAgent(Agent):
    def __init__(self):
        self.hello_component = HelloComponent()
        self.calculator_component = CalculatorComponent().run_after(self.hello_component)
        # This is equivalent to passing a type:
        # self.calculator_component = CalculatorComponent().run_after(HelloComponent)
```

!!! warning
    Be sure not to make circular dependencies when ordering components!

### Ordering all components

You can also order all components by setting `self.components` list in the agent's `__init__` method.
This way ensures that there's no circular dependencies and any `run_after` calls are ignored.

!!! warning
    Be sure to include all components - by setting `self.components` list, you're overriding the default behavior of discovering components automatically. Since it's usually not intended agent will inform you in the terminal if some components were skipped.

```py
class MyAgent(Agent):
    def __init__(self):
        self.hello_component = HelloComponent()
        self.calculator_component = CalculatorComponent()
        # Explicitly set components list
        self.components = [self.hello_component, self.calculator_component]
```

## Disabling components

You can control which components are enabled by setting their `_enabled` attribute.
Components are *enabled* by default.
Either provide a `bool` value or a `Callable[[], bool]`, will be checked each time
the component is about to be executed. This way you can dynamically enable or disable
components based on some conditions.
You can also provide a reason for disabling the component by setting `_disabled_reason`.
The reason will be visible in the debug information.

```py
class DisabledComponent(MessageProvider):
    def __init__(self):
        # Disable this component
        self._enabled = False
        self._disabled_reason = "This component is disabled because of reasons."

        # Or disable based on some condition, either statically...:
        self._enabled = self.some_property is not None
        # ... or dynamically:
        self._enabled = lambda: self.some_property is not None

    # This method will never be called
    def get_messages(self) -> Iterator[ChatMessage]:
        yield ChatMessage.user("This message won't be seen!")

    def some_condition(self) -> bool:
        return False
```

If you don't want the component at all, you can just remove it from the agent's `__init__` method. If you want to remove components you inherit from the parent class you can set the relevant attribute to `None`:

!!! Warning
    Be careful when removing components that are required by other components. This may lead to errors and unexpected behavior.

```py
class MyAgent(Agent):
    def __init__(self):
        super().__init__(...)
        # Disable WatchdogComponent that is in the parent class
        self.watchdog = None

```

## Exceptions

Custom errors are provided which can be used to control the execution flow in case something went wrong. All those errors can be raised in protocol methods and will be caught by the agent.  
By default agent will retry three times and then re-raise an exception if it's still not resolved. All passed arguments are automatically handled and the values are reverted when needed.
All errors accept an optional `str` message. There are following errors ordered by increasing broadness:

1. `ComponentEndpointError`: A single endpoint method failed to execute. Agent will retry the execution of this endpoint on the component.
2. `EndpointPipelineError`: A pipeline failed to execute. Agent will retry the execution of the endpoint for all components.
3. `ComponentSystemError`: Multiple pipelines failed.

**Example**

```py
from forge.agent.components import ComponentEndpointError
from forge.agent.protocols import MessageProvider

# Example of raising an error
class MyComponent(MessageProvider):
    def get_messages(self) -> Iterator[ChatMessage]:
        # This will cause the component to always fail 
        # and retry 3 times before re-raising the exception
        raise ComponentEndpointError("Endpoint error!")
```
