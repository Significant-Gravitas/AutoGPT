import pickle

class AgentPersist:

    def __init__(self):
        pass

    def save(self, agent):
        with open("agent.pkl", "wb") as f:
            pickle.dump(agent, f)
