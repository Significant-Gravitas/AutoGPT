import pickle

class AgentLoader:

    def __init__(self, agent_path):
        self._agent_path = agent_path

    def load(self):
        with open(self._agent_path, "rb") as f:
            return pickle.load(f)
