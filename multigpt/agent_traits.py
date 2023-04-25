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
