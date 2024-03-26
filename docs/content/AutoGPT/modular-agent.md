# Modular Agents

This incremental re-architecture rebuilds `*Agent` classes so they are composed of `Components` instead of current multiple inheritance with Mixins.

>Due to technical debt the state of current codebase becomes increasingly unwieldy and is harder to implement new features. I'm aware of the failed past re-arch attempt and I took that into account when considering this change. Currently Mixins are very confusing - it's hard to track the order of execution and what's happening in the code. Also, plugins feel like additional baggage that needs to be maintained and not as a useful part of the system. ~kcze

This re-arch reuses much current code and is done in-place (not from scratch). With this fully implemented, Agent becomes a set of components, and that unifies Agent code and Plugins, now *everything* is just components.

This change directly addresses point 2 of Roadmap [Empowering Agent Builders](https://github.com/Significant-Gravitas/AutoGPT/discussions/6970) and may also have a positive impact on all the others (due to ease of use and extension).



### The main goals:
- Simplify Agent and remove redundant code
- Make it as easy as possible to create new Agents and Components (plugins)

### Tasks
- [ ] 

## How does it work
Agent is composed of *components*, and each `Component` implements a range of `Protocol`s (interfaces), each one providing a specific functionality, e.g. additional commands or messages.

Agent has methods (currently `propose_action` and `execute`) that execute *pipelines* that call methods on components in a specified order.

This system is flexible and doesn't hide any data (anything can still be passed or accessed directly from or between modules).

Example Protocol:
```py
@runtime_checkable
class MessageProvider(Protocol):
    def get_messages(self) -> Iterator[ChatMessage]:
        ...
```
Component (=plugin) that implements it:
```py
class MyComponent(Component, MessageProvider):
    def get_messages(self) -> Iterator[ChatMessage]:
        yield ChatMessage("This will be injected to prompt!")
```
Agent that uses the component:
```py
class MyAgent(Agent):
    def __init__(self):
        # Optional super call to bring default components
        super().__init__(...)
        self.my_component = MyComponent()
        # Can define explicit ordering, otherwise components are sorted automatically
        # self.components = [self.my_component]
```
And that's it! Components are automatically collected from the agent using python metaclass magic and are called when needed.

Now purpose-related things will be bunded together: so `FileManager` provides file-system related commands and resources information. This should cut out a lot of code and also make system more type safe.

## Other stuff
Debugging may be easier because we can inspect the exact components that were called and where the pipeline failed (current WIP pipeline):

![](../imgs/modular-pipeline.png)

Also that makes it possible to call component/pipeline/function again when failed and recover.

## Challenges
- Ordering
- Efficient and type safe pipelines code

## Future possibilities
- Adding or removing components during runtime
- Parallel component execution
- Cacheable pipelines