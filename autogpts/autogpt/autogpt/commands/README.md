# 🧩 Components

Components are the building blocks of [🤖 Agents](./agents.md). They are classes inheriting `AgentComponent` or implementing one or more [⚙️ Protocols](./protocols.md) that give agent additional abilities or processing. 

Components can be used to implement various functionalities like providing messages to the prompt, executing code, or interacting with external services.
They can be enabled or disabled, ordered, and can rely on each other.

Components assigned in the agent's `__init__` via `self` are automatically detected upon the agent's instantiation.
For example inside `__init__`: `self.my_component = MyComponent()`.
You can use any valid Python variable name, what matters for the component to be detected is its type (`AgentComponent` or any protocol inheriting from it).

Visit [Built-in Components](./built-in-components.md) to see what components are available out of the box.

```py
from autogpt.agents import Agent
from autogpt.agents.components import AgentComponent

class HelloComponent(AgentComponent):
    pass

class SomeComponent(AgentComponent):
    def __init__(self, hello_component: HelloComponent):
        self.hello_component = hello_component

class MyAgent(Agent):
    def __init__(self):
        # These components will be automatically discovered and used
        self.hello_component = HelloComponent()
        # We pass HelloComponent to SomeComponent
        self.some_component = SomeComponent(self.hello_component)
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
from autogpt.agents.components import ComponentEndpointError
from autogpt.agents.protocols import MessageProvider

# Example of raising an error
class MyComponent(MessageProvider):
    def get_messages(self) -> Iterator[ChatMessage]:
        # This will cause the component to always fail 
        # and retry 3 times before re-raising the exception
        raise ComponentEndpointError("Endpoint error!")
```
