class AgentTraits:
    def __init__(self, openness: int = -1, agreeableness: int = -1, conscientiousness: int = -1,
                 emotional_stability: int = -1, assertiveness: int = -1, description: str = ""):
        self.openness = openness
        self.agreeableness = agreeableness
        self.conscientiousness = conscientiousness
        self.emotional_stability = emotional_stability
        self.assertiveness = assertiveness
        self.description = description

    def __str__(self):
        return (
            f"Openness: {self.openness}\n"
            f"Agreeableness: {self.agreeableness}\n"
            f"Conscientiousness: {self.conscientiousness}\n"
            f"Emotional Stability: {self.emotional_stability}\n"
            f"Assertiveness: {self.assertiveness}\n\n"
            f"{self.description}"
        )

from yaml.constructor import ConstructorError, SafeConstructor
from yaml.representer import SafeRepresenter

def agent_traits_constructor(loader, node):
    values = loader.construct_mapping(node)
    return AgentTraits(**values)

def agent_traits_representer(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:python/object:multigpt.agent_traits.AgentTraits', data.__dict__)

SafeConstructor.add_constructor('tag:yaml.org,2002:python/object:multigpt.agent_traits.AgentTraits', agent_traits_constructor)
SafeRepresenter.add_representer(AgentTraits, agent_traits_representer)
