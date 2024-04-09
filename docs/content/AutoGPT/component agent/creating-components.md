# Creating Components

## The minimal component

Let's create a simple component that adds "Hello World!" message to the agent prompt.  
To create a component you just make a class that inherits from `Component`:

```py
# We recommend *Component suffix to make the type clear
class HelloComponent(Component):
    pass
```

This is already a valid component but it doesn't have any functionality yet.  
To make it do something we need to write a method that can be found and called by the agent. To put messages to the agent's prompt we need to implement `MessageProvider` Protocol in our component. `MessageProvider` is an interface with `get_messages` method:

```py
class HelloComponent(Component, MessageProvider):
    def get_messages(self) -> Iterator[ChatMessage]:
        yield ChatMessage.user("Hello World!")
```

Now we can add our component to an existing agent or create a new Agent class and add it there:

```py
class MyAgent(Agent):
    self.hello_component = HelloComponent()
```

`get_messages` will called by the agent each time it needs to build a new prompt and the yielded messages will be added accordingly.  

## Full example



```py

```

## Learn more

Guide on how to extend the built-in agent and build your own: [🤖 Agents](./agents.md)  
Order of some components matters, see [🧩 Components](./components.md) to learn more about components and how they can be customized.  
To see built-in protocols with accompanying examples visit [⚙️ Protocols](./protocols.md).
