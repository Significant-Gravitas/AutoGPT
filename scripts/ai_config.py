import yaml
import data

class AIConfig:
    def __init__(self, ai_name="", ai_role="", ai_goals=[]):
        self.ai_name = ai_name
        self.ai_role = ai_role
        self.ai_goals = ai_goals

    # @classmethod
    # def create_from_user_prompts(cls):
    #     ai_name = input("Name your AI: ") or "Entrepreneur-GPT"
    #     ai_role = input(f"{ai_name} is: ") or "an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth."
    #     print("Enter up to 5 goals for your AI: ")
    #     print("For example: \nIncrease net worth, Grow Twitter Account, Develop and manage multiple businesses autonomously'")
    #     print("Enter nothing to load defaults, enter nothing when finished.")
    #     ai_goals = []
    #     for i in range(5):
    #         ai_goal = input(f"Goal {i+1}: ")
    #         if ai_goal == "":
    #             break
    #         ai_goals.append(ai_goal)
    #     if len(ai_goals) == 0:
    #         ai_goals = ["Increase net worth", "Grow Twitter Account", "Develop and manage multiple businesses autonomously"]
    #     return cls(ai_name, ai_role, ai_goals)

    @classmethod
    def load(cls, config_file="config.yaml"):
        # Load variables from yaml file if it exists
        try:
            with open(config_file) as file:
                config_params = yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            config_params = {}

        ai_name = config_params.get("ai_name", "")
        ai_role = config_params.get("ai_role", "")
        ai_goals = config_params.get("ai_goals", [])

        return cls(ai_name, ai_role, ai_goals)

    def save(self, config_file="config.yaml"):
        config = {"ai_name": self.ai_name, "ai_role": self.ai_role, "ai_goals": self.ai_goals}
        with open(config_file, "w") as file:
            documents = yaml.dump(config, file)

    def construct_full_prompt(self):
        prompt_start = """Your decisions must always be made independently without seeking user assistance. Play to your strengths as an LLM and pursue simple strategies with no legal complications."""

        # Construct full prompt
        full_prompt = f"You are {self.ai_name}, {self.ai_role}\n{prompt_start}\n\nGOALS:\n\n"
        for i, goal in enumerate(self.ai_goals):
            full_prompt += f"{i+1}. {goal}\n"

        full_prompt += f"\n\n{data.load_prompt()}"
        return full_prompt
