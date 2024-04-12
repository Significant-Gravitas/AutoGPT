# Creating Components

## The minimal component

*Component* is a class that inherits from `AgentComponent` OR implements one or more *protocols*. Every *protocol* inherits `AgentComponent`, so your class automatically becomes a *component* once you inherit any *protocol*.

```py
class MyComponent(AgentComponent):
    pass
```

This is already a valid component, but it doesn't do anything yet. To add some functionality to it, you need to implement one or more *protocols*.

Let's create a simple component that adds "Hello World!" message to the agent's prompt. To do this we need to implement `MessageProvider` *protocol* in our component. `MessageProvider` is an interface with `get_messages` method:

```py
# We no longer need to inherit AgentComponent
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

## Learn more

Guide on how to extend the built-in agent and build your own: [🤖 Agents](./agents.md)  
Order of some components matters, see [🧩 Components](./components.md) to learn more about components and how they can be customized.  
To see built-in protocols with accompanying examples visit [⚙️ Protocols](./protocols.md).
