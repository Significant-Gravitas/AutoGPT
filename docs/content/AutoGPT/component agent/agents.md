# 🤖 Agents

Agent is composed of [🧩 Components](./components.md) and responsible for executing pipelines and some additional logic. The base class for all agents is `BaseAgent`, it has the necessary logic to collect components and execute protocols.

## AutoGPT Agent

`Agent` is the main agent provided by AutoGPT. It's a subclass of `BaseAgent`.

### Important methods

`propose_action` and `execute`