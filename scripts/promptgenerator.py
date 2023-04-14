import json


class PromptGenerator:
    """
    基于约束、命令、资源和性能评估生成自定义提示字符串的类。
    """

    def __init__(self):
        """
        使用空列表初始化PromptGenerator对象的constraints、commands、resources和performance_evaluation属性。
        """
        self.constraints = []
        self.commands = []
        self.resources = []
        self.performance_evaluation = []
        self.response_format = {
            "thoughts": {
                "text": "thought",
                "reasoning": "reasoning",
                "plan": "- short bulleted\n- list that conveys\n- long-term plan",
                "criticism": "constructive self-criticism",
                "speak": "thoughts summary to say to user"
            },
            "command": {
                "name": "command name",
                "args": {
                    "arg name": "value"
                }
            }
        }

    def add_constraint(self, constraint):
        """
        向约束列表中添加约束。

        参数：
        constraint(str)：要添加的约束。
        """
        self.constraints.append(constraint)

    def add_command(self, command_label, command_name, args=None):
        """
        该函数向“commands”列表中添加一个命令，命令包括标签、名称和可选参数。

        参数:

        command_label(str): 命令的标签。
        command_name(str): 命令的名称。
        args(dict, 可选): 包含参数名称和其值的字典。默认为None。
        """
        if args is None:
            args = {}

        command_args = {arg_key: arg_value for arg_key,
                        arg_value in args.items()}

        command = {
            "label": command_label,
            "name": command_name,
            "args": command_args,
        }

        self.commands.append(command)

    def _generate_command_string(self, command):
        """
        生成命令的格式化字符串表示。

        参数：
        command（dict）：包含命令信息的字典。

        返回：
        str：格式化的命令字符串。
        """
        args_string = ', '.join(
            f'"{key}": "{value}"' for key, value in command['args'].items())
        return f'{command["label"]}: "{command["name"]}", args: {args_string}'

    def add_resource(self, resource):
        """
        将资源添加到资源列表中。

        参数：
        resource (str): 要添加的资源。
        """
        self.resources.append(resource)

    def add_performance_evaluation(self, evaluation):
        """
        将一个性能评估项添加到性能评估列表中。

        参数：
        evaluation（str）：要添加的评估项。
        """
        self.performance_evaluation.append(evaluation)

    def _generate_numbered_list(self, items, item_type='list'):
        """
        根据给定的项类型（item_type），生成一个带编号的列表。

        参数：
        items（list）：要编号的项的列表。
        item_type（str，可选）：列表中项的类型。默认为“list”。

        返回值：
        str：格式化后的编号列表。
        """
        if item_type == 'command':
            return "\n".join(f"{i+1}. {self._generate_command_string(item)}" for i, item in enumerate(items))
        else:
            return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))

    def generate_prompt_string(self):
        """
        该方法生成基于限制条件、命令、资源和性能评估的提示字符串。

        返回值：
        str: 生成的提示字符串。
        """
        formatted_response_format = json.dumps(self.response_format, indent=4)
        prompt_string = (
            f"约束条件:\n{self._generate_numbered_list(self.constraints)}\n\n"
            f"指令:\n{self._generate_numbered_list(self.commands, item_type='command')}\n\n"
            f"资源:\n{self._generate_numbered_list(self.resources)}\n\n"
            f"性能评估:\n{self._generate_numbered_list(self.performance_evaluation)}\n\n"
            f"你应该按照以下所描述的 JSON 格式进行回应\n响应格式： \n{formatted_response_format} \nEnsure the response can be parsed by Python json.loads"
        )

        return prompt_string
