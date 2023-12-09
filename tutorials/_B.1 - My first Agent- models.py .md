# Tutorial: Creating Your First Agent with a `models.py` Module in the afaas Framework

## **Prerequisites:**

- Familiarity with Python programming.
- A basic understanding of the afaas framework and its Agent architecture.

## **Step 1: Setting Up Your Project Structure**

## **1.1 Project Directory Structure:**

Creating a well-organized directory structure is crucial for managing your project effectively. Here is how you should structure your project:

```plaintext
/agents
    /mycustomagent
        /strategies
        agent.py
        loop.py
        models.py
```

- **/strategies**: This directory will hold your custom prompt strategies.
- **agent.py**: This file will house the logic for creating and managing your agent.
- **loop.py**: This file will handle the loop logic for your agent.
- **models.py**: This file will contain models required by your agent.

## **Step 2: Defining Your Agent Configurations in `models.py`**

In this step, we will define configurations for your agent. These configurations will determine how your agent operates within the afaas framework.

## **2.1 Importing Necessary Libraries and Classes:**

We'll start by importing the necessary libraries and classes. This includes importing the base models from the afaas framework, and other necessary libraries like `uuid` for generating unique identifiers, and `pydantic` for data validation.

```python
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from pydantic import Field

from autogpt.core.agents.base.models import (
    BaseAgentConfiguration,
    BaseAgentSettings,
    BaseAgentSystems,
    BaseAgentSystemSettings,
)
from autogpt.core.planning import PlannerSettings
from autogpt.core.plugin.simple import PluginLocation
from autogpt.core.resource.model_providers import OpenAISettings

```

### **2.1 Agent Systems Class:**

The Agent Systems class defines the systems that your agent will use. These systems could include memory, workspace, and other essential components for your agent's operation.

```python
class BaseAgentSystems(SystemConfiguration):
    memory: PluginLocation
    workspace: PluginLocation

    class Config(SystemConfiguration.Config):
        extra = "allow"
```

### **2.2 Agent Configuration Class:**

The Agent Configuration class specifies the configurations for your agent. It includes the systems it uses, cycle count, and identifiers like user ID and agent ID.

```python
class BaseAgentConfiguration(SystemConfiguration):
    cycle_count: int
    max_task_cycle_count: int
    creation_time: str
    systems: BaseAgentSystems
    user_id: Optional[uuid.UUID] = Field(default=None)
    agent_id: Optional[uuid.UUID] = Field(default=None)

    class Config(SystemConfiguration.Config):
        extra = "allow"
```

### **2.3 Agent System Settings Class:**

The Agent System Settings class specifies settings for your agent's systems. It holds a reference to the Agent Configuration class.

```python
class BaseAgentSystemSettings(SystemSettings):
    configuration: BaseAgentConfiguration

    class Config(SystemSettings.Config):
        extra = "allow"
```

### **2.4 Agent Settings Class:**

The Agent Settings class aggregates all settings required for initializing your agent. It includes references to the Agent System Settings class, Memory Settings, and Workspace Settings, among others.

```python
class BaseAgentSettings(BaseModel):
    agent: BaseAgentSystemSettings
    memory: MemorySettings
    workspace: WorkspaceSettings
    agent_class: str

    class Config(BaseModel.Config):
        # ... (rest of the content)
```

## **Example Implementation: UserContextAgent**

Now, let's look at an example implementation from `UserContextAgent` to see how these classes can be subclassed and used in a practical scenario.

### **2.2.1 Defining Agent Systems:**

```python
class UserContextAgentSystems(BaseAgentSystems):
    ability_registry: PluginLocation
    openai_provider: PluginLocation
    planning: PluginLocation

    class Config(BaseAgentSystems.Config):
        pass
```

### **2.3.1 Defining Agent Configuration:**

```python
class UserContextAgentConfiguration(BaseAgentConfiguration):
    systems: UserContextAgentSystems
    agent_name: str = Field(default="New Agent")

    class Config(BaseAgentConfiguration.Config):
        pass
```

### **2.4.1 Defining Agent System Settings:**

```python
class UserContextAgentSystemSettings(BaseAgentSystemSettings):
    configuration: UserContextAgentConfiguration
    user_id: Optional[uuid.UUID] = Field(default=None)
    agent_id: Optional[uuid.UUID] = Field(default=None)

    class Config(BaseAgentSystemSettings.Config):
        pass
```

### **2.5.1 Defining Agent Settings:**

```python
class UserContextAgentSettings(BaseAgentSettings):
    # ... (content from UserContextAgent models.py)
```

# Note

In this example, we subclass the base classes provided in the framework to create custom configurations, systems, and settings for the `UserContextAgent`. This pattern can be followed to create configurations for other types of agents within the afaas framework, adjusting the attributes and methods as needed for your specific use case.

# **Further Steps:**

Now, we will proceed to implement the `agent.py` and `loop.py` files, and initialize your agent. Each step will be explained in detail to ensure you have a clear understanding of how to create and manage your agent within the afaas framework.
