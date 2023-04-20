## Module `agent_manager`

### AgentManager

```python
class AgentManager(metaclass=Singleton)
```

#### Description

Agent manager for managing GPT agents.

This class provides methods for managing GPT agents.

#### Methods

##### `create_agent`

```python
def create_agent(self, task: str, prompt: str, model: str) -> tuple[int, str]
```

Create a new agent and return its key.

###### Arguments

* `task` (`str`): The task to perform.
* `prompt` (`str`): The prompt to use.
* `model` (`str`): The model to use.

###### Return Value

The key of the new agent.

##### `message_agent`

```python
def message_agent(self, key: Union[str, int], message: str) -> str
```

Send a message to an agent and return its response.

###### Arguments

* `key` (`Union[str, int]`): The key of the agent to message.
* `message` (`str`): The message to send to the agent.

###### Return Value

The response of the agent.

##### `list_agents`

```python
def list_agents(self) -> List[tuple[Union[str, int], str]]
```

Return a list of all agents.

###### Return Value

A list of tuples of the form (key, task).

##### `delete_agent`

```python
def delete_agent(self, key: Union[str, int]) -> bool
```

Delete an agent from the agent manager.

###### Arguments

* `key` (`Union[str, int]`): The key of the agent to delete.

###### Return Value

True if successful, False otherwise.

#### Example

```python
agent_manager = AgentManager()
agent_manager.create_agent("task", "prompt", "model")
```