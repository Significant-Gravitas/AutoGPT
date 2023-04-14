import yaml
import os
from prompt import get_prompt


class AIConfig:
    """
    包含AI配置信息的类对象
    
    属性：  
        ai_name (str): 人工智能的名称。
        ai_role (str): 人工智能角色的描述。
        ai_goals (list): 人工智能应该完成的目标列表。
    """

    def __init__(self, ai_name: str="", ai_role: str="", ai_goals: list=[]) -> None:
        """
        初始化一个类实例

         参数：
            ai_name (str): 人工智能的名称。
            ai_role (str): 人工智能角色的描述。
            ai_goals (list): 人工智能应该完成的目标列表。
        Returns:
            None
        """

        self.ai_name = ai_name
        self.ai_role = ai_role
        self.ai_goals = ai_goals

    # Soon this will go in a folder where it remembers more stuff about the run(s)
    SAVE_FILE = os.path.join(os.path.dirname(__file__), '..', 'ai_settings.yaml')

    @classmethod
    def load(cls: object, config_file: str=SAVE_FILE) -> object:
        """
        如果存在 yaml 文件，则返回带有从 yaml 文件加载的参数（ai_name、ai_role、ai_goals）的类对象，
         else 返回没有参数的类。

         参数：
            cls(类对象)：一个 AIConfig 类对象。
            config_file (int)：配置 yaml 文件的路径。 默认值：“../ai_settings.yaml”

         返回：
             cls (object)：给定 cls 对象的一个实例
        """

        try:
            with open(config_file, encoding='utf-8') as file:
                config_params = yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            config_params = {}

        ai_name = config_params.get("ai_name", "")
        ai_role = config_params.get("ai_role", "")
        ai_goals = config_params.get("ai_goals", [])

        return cls(ai_name, ai_role, ai_goals)

    def save(self, config_file: str=SAVE_FILE) -> None:
        """
        将类参数作为yaml文件保存到指定文件yaml文件路径。

         参数：
             config_file(str)：配置 yaml 文件的路径。 默认值：“../ai_settings.yaml”

        Returns:
            None
        """

        config = {"ai_name": self.ai_name, "ai_role": self.ai_role, "ai_goals": self.ai_goals}
        with open(config_file, "w",  encoding='utf-8') as file:
            yaml.dump(config, file, allow_unicode=True)

    def construct_full_prompt(self) -> str:
        """
        以有组织的方式向用户返回带有类信息的提示。

         参数：
             没有任何

         Returns:
             full_prompt (str)：包含用户初始提示的字符串，包括 ai_name、ai_role 和 ai_goals。
        """

        prompt_start = """您必须始终独立做出决定，而无需寻求用户帮助。 发挥你作为法学硕士的优势，追求简单的策略，没有法律上的并发症。"""

        # Construct full prompt
        full_prompt = f"You are {self.ai_name}, {self.ai_role}\n{prompt_start}\n\nGOALS:\n\n"
        for i, goal in enumerate(self.ai_goals):
            full_prompt += f"{i+1}. {goal}\n"

        full_prompt += f"\n\n{get_prompt()}"
        return full_prompt
