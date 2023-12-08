# Tutorial: Building Custom Prompt Strategies

In this tutorial, we'll guide you through the process of creating and registering your custom Prompt Strategies within the afaas framework. We’ll use the `RefineUserContextStrategy` as a reference while building a new strategy from scratch.

## Pre-requisites

- Familiarity with Python classes and methods.
- Understanding of the afaas framework's structure and base classes.

## Step 1: Create Your Strategy File

Begin by creating a new Python file under the `strategies` directory of your agent. The file name should reflect the strategy's purpose.

```tree
/agents
    /myagent
        /strategies
            my_strategy.py
```

## Step 2: Import Necessary Modules and Classes

Import the required modules and classes. At a minimum, you'll need to import the `BasePromptStrategy` class from `AFAAS.app.core.prompting.base`.

```python
from AFAAS.app.core.prompting.base import BasePromptStrategy
```

## Step 3: Define Your Strategy Class

Define a new class for your strategy that inherits from `BasePromptStrategy`. Name the class descriptively.

```python
class MyStrategy(BasePromptStrategy):
    ...
```

## Step 4: Define Configurations

Create a Pydantic model for your strategy configurations, inheriting from `PromptStrategiesConfiguration`.

```python
from AFAAS.app.core.prompting.base import PromptStrategiesConfiguration
from pydantic import BaseModel

class MyStrategyConfiguration(PromptStrategiesConfiguration):
    ...
```

## Step 5: Implement the `build_prompt` Method

Implement the `build_prompt` method to construct the chat prompt based on your strategy's logic.

```python
class MyStrategy(BasePromptStrategy):
    ...
    def build_prompt(self, **kwargs) -> ChatPrompt:
        ...
```

## Step 6: Register Your Strategy

Ensure your strategy is registered so it can be used by the agent. Update the `__init__.py` file within the `strategies` directory to import your strategy class. Then, update the `StrategiesConfiguration` and `Strategies` classes to include your new strategy.

```python
# In __init__.py
from .my_strategy import MyStrategy

class StrategiesConfiguration(PromptStrategiesConfiguration):
    my_strategy: MyStrategyConfiguration
    ...

class Strategies:
    from AFAAS.app.core.prompting.base import BasePromptStrategy, PromptStrategy

    @staticmethod
    def get_strategies(logger: Logger) -> list[PromptStrategy]:
        return [
            MyStrategy(logger=logger, **MyStrategy.default_configuration.dict()),
            ...
        ]
```

[Example on Registering Strategies](https://github.com/ph-ausseil/afaas/blob/first-agent-tutorial/autogpts/autogpt/autogpt/core/agents/simple/strategies/__init__.py)

Now your custom strategy is created and registered, ready to be used within the afaas framework. By following this pattern, you can create additional strategies tailored to your agent's needs.

## Resources

- [RefineUserContextStrategy File](https://raw.githubusercontent.com/ph-ausseil/afaas/first-agent-tutorial/autogpts/autogpt/autogpt/core/agents/usercontext/strategies/refine_user_context.py)
- [Example on Registering Strategies](https://github.com/ph-ausseil/afaas/blob/first-agent-tutorial/autogpts/autogpt/autogpt/core/agents/simple/strategies/__init__.py)
