
# NOTE : Keep it simple or set a long term structure (Pain in the ass)    
# NOTE : Not seing it as very useful    
class AgentTeam:
    """
    A class representing a team of agents for anproject.

    Attributes:
        team_name (str): The name of the team.
        lead_agent (AgentModel): The lead agent for the team.
        delegated_agents (List[AgentModel]): A list of delegated agents for the team.

    Methods:
        __init__(self, team_name: str, lead_agent: AgentModel, delegated_agents: List[AgentModel])
            Initializes a new AgentTeam object with the given parameters.
        to_dict(self) -> dict
            Converts the AgentTeam object to a dictionary representation.

    """
    def __init__(self, team_name, lead_agent, delegated_agents):
        """
        Initializes a new AgentTeam object with the given parameters.

        Args:
            team_name (str): The name of the team.
            lead_agent (AgentModel): The lead agent for the team.
            delegated_agents (List[AgentModel]): A list of delegated agents for the team.
        """
        
        self.team_name = team_name
        self.lead_agent = lead_agent
        self.delegated_agents = delegated_agents
        
        # CREATE A LIST OF AGENT IF EVER TO BE USED 
        # NOT PUBLICLY AVAILABLE AS DEEMED DANGEROUS
        self._all_agent = []
        self._all_agent.append(lead_agent)
        for agent in delegated_agents :
            self._all_agent.append(agent)

    # def to_dict(self) -> dict : 
    #     """
    #     Converts the AgentTeam object to a dictionary representation.

    #     Returns:
    #         dict_representation (dict): A dictionary representation of the AgentTeam object.
    #     """
    #     lead_agent_dict = self.lead_agent.to_dict() 

    #     delegated_agents = []
    #     for agent in self.delegated_agents:
    #         agent_dict = agent.to_dict() 
    #         delegated_agents.append(agent_dict)

    #     return  {
    #         'team_name' : self.team_name,
    #         'lead_agent' : lead_agent_dict,
    #         'delegated_agents' : delegated_agents
    #     }
    
# TODO : test & migrate to this function
def object_to_dict(obj):
    obj_dict = {}
    for key in dir(obj):
        if not key.startswith('__'):
            value = getattr(obj, key)
            if not callable(value):
                if isinstance(value, object):
                    obj_dict[key] = object_to_dict(value)
                elif isinstance(value, list):
                    obj_dict[key] = []
                    for item in value:
                        if isinstance(item, object):
                            obj_dict[key].append(object_to_dict(item))
                        else:
                            obj_dict[key].append(item)
                else:
                    obj_dict[key] = value
    return obj_dict

