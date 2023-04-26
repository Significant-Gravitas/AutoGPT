
from typing import Optional, Type, List

class AgentConfig(): 
    """
    A class representing an agent configuration.

    Attributes:
        agent_name (str): The name of the agent.
        agent_role (str): The role of the agent.
        agent_goals (List): A list of agent goals.
        agent_model (Optional[str]): The agent model name, if applicable.
        agent_model_type (Optional[str]): The agent model type, if applicable.
        prompt_generator (Any): The prompt generator instance.
        command_registry (Any): The command registry instance.
    """
    def __init__(self, 
            agent_name: str,
            agent_role: str,
            agent_goals: List,
            agent_model: Optional[str] = None,
            agent_model_type: Optional[str] = None,
            prompt_generator =  None,
            command_registry =  None) -> None:
        """
        Initializes the AgentConfig class with the given attributes.

        Args:
            agent_name (str): The name of the agent.
            agent_role (str): The role of the agent.
            agent_goals (List): A list of agent goals.
            agent_model (Optional[str], optional): The agent model name, if applicable. Defaults to None.
            agent_model_type (Optional[str], optional): The agent model type, if applicable. Defaults to None.
            prompt_generator (Any, optional): The prompt generator instance. Defaults to None.
            command_registry (Any, optional): The command registry instance. Defaults to None.
        """

        self.agent_name = agent_name
        self.agent_role = agent_role
        self.agent_goals = agent_goals
        self.agent_model = agent_model
        self.agent_model_type = agent_model_type
        self.prompt_generator= prompt_generator
        self.command_registry= command_registry