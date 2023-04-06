import data
import yaml
from langchain import PromptTemplate


class AIConfig:
    def __init__(self, ai_name="", ai_role="", ai_goals=[]):
        self.ai_name = ai_name
        self.ai_role = ai_role
        self.ai_goals = ai_goals

    # Soon this will go in a folder where it remembers more stuff about the run(s)
    SAVE_FILE = "last_run_ai_settings.yaml"

    @classmethod
    def load(cls, config_file=SAVE_FILE):
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

    def save(self, config_file=SAVE_FILE):
        config = {"ai_name": self.ai_name, "ai_role": self.ai_role, "ai_goals": self.ai_goals}
        with open(config_file, "w") as file:
            yaml.dump(config, file)

    def construct_full_prompt(self, agent_id=None, agent_name=None, agent_task=None, agent_goals=None, agent_supervisor=None):
        full_prompt = ""

        prompt_start = """Your decisions must always be made independently without seeking user assistance. Play to your strengths as an LLM and pursue simple strategies with no legal complications."""

        # Construct full prompt
        full_prompt = f"You are {self.ai_name}, {self.ai_role}\n{prompt_start}\n\nGOALS:\n\n"
        for i, goal in enumerate(self.ai_goals):
            full_prompt += f"{i+1}. {goal}\n"

        full_prompt += f"\n\n{data.load_prompt()}"
        return full_prompt


def auto_construct_full_prompt(agent_id, agent_name, agent_task, agent_goals, agent_supervisor):

    template = """
        You are {agent_name}, an employee of {agent_supervisor}. You are tasked with: {agent_task}.\n
        Your decisions must be made as independantly as possible and you should report any issues, updates, or answers to your supervisor. \n 
        You can do this by messaging your supervisor. 
        Instructions on how to are displayed in your command list. \n
        These are your goals assigned by your supervisor. Here are your goals: \n {agent_goals}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["agent_name", "agent_supervisor", "agent_task", "agent_goals"]
    )

    full_prompt = prompt.format(agent_name=agent_name,
                                agent_supervisor=agent_supervisor,
                                agent_task=agent_task,
                                agent_goals=agent_goals)

    full_prompt += f"\n\n{data.load_prompt()}"

    return full_prompt