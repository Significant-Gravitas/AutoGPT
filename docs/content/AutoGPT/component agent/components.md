# 🧩 Components

Components are the building blocks of [🤖 Agents](./agents.md). They are classes inherited from `Component` that implement one or more [⚙️ Protocols](./protocols.md) that give agent additional abilities or processing. 

Components assigned to attributes (fields) in agent's `__init__` are automatically discovered upon instantiation.
Each component can implement multiple protocols and can rely on other components if needed.

```py
from autogpt.agents import Agent
from autogpt.agents.components import Component

class MyAgent(Agent):
    def __init__(self):
        # These components will be automatically discovered and used
        self.hello_component = HelloComponent()
        # We pass HelloComponent to CalculatorComponent
        self.calculator_component = CalculatorComponent(self.hello_component)
```
## Ordering components

For some protocols, the order of components is important because the latter ones may depend on the results of the former ones.

### Implicit order

Components can be ordered implicitly by the agent; each component can set `run_after` list to specify which components should run before it. This is useful when components rely on each other or need to be executed in a specific order. Otherwise, the order of components is alphabetical.

```py
# This component will run after HelloComponent
class CalculatorComponent(Component):
    run_after = [HelloComponent]

    def __init__(self, hello_component: HelloComponent):
        self.hello_component = hello_component
```

### Explicit order

Sometimes it may be easier to order components explicitly by setting `self.components` list in the agent's `__init__` method. This way you can also ensure there's no circular dependencies and `run_after` is ignored.

> ⚠️ Be sure to include all components - by setting `self.components` list, you're overriding the default behavior of discovering components automatically. Since it's usually not intended agent will inform you in the terminal if some components were skipped.

```py
class MyAgent(Agent):
    def __init__(self):
        self.hello_component = HelloComponent()
        self.calculator_component = CalculatorComponent(self.hello_component)
        # Explicitly set components list
        self.components = [self.hello_component, self.calculator_component]
```

## Disabling components

You can control which components are enabled by setting their `enabled` attribute. You can either provide a `bool` value or a `callable[[], bool]` that will be called each time the component is about to be executed. This way you can dynamically enable or disable components based on some conditions.
You can also provide a reason for disabling the component by setting `disabled_reason`. The reason will be visible in the debug information.

```py
class DisabledComponent(Component, MessageProvider):
    def __init__(self):
        # Disable this component
        self.enabled = False
        self.disabled_reason = "This component is disabled because of reasons."
        # Or disable based on some condition
        self.enabled = self.some_condition

    # This method will never be called
    def get_messages(self) -> Iterator[ChatMessage]:
        yield ChatMessage.user("This message won't be seen!")

    def some_condition(self) -> bool:
        return False
```

If you don't want the component at all, you can just remove it from the agent's `__init__` method. If you want to remove components you inherit from the parent class you can set the relevant attribute to `None`:

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
All errors accept an optional `str` message.

1. `ComponentError`: A single component failed to execute. Agent will retry the execution of the component.
2. `ProtocolError`: An entire protocol failed to execute. Agent will retry the execution of the protocol method for all components.
3. `PipelineError`: An entire pipeline failed to execute. Agent will retry the execution of the pipeline for all protocols. This isn't implemented yet.
4. `ComponentSystemError`: The highest-level error occurred in the component system. This isn't used.

**Example**

```py
from autogpt.agents.components import Component, ComponentError
from autogpt.agents.protocols import MessageProvider

# Example of raising an error
class MyComponent(Component, MessageProvider):
    def get_messages(self) -> Iterator[ChatMessage]:
        raise ComponentError("Component error!")
```