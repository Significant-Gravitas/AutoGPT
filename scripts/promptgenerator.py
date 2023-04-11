import json


class PromptGenerator:
    def __init__(self):
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
        self.constraints.append(constraint)

    # {CommandLabel}: "{CommandName}", args: "{arg#Name}": "{arg#Prompt}"
    def add_command(self, command_label, command_name, args=None):
        if args is None:
            args = {}
            
        command_args = {arg_key: arg_value for arg_key, arg_value in args.items()}

        command = {
            "label": command_label,
            "name": command_name,
            "args": command_args,
        }

        self.commands.append(command)

    def _generate_command_string(self, command):
        args_string = ', '.join(f'"{key}": "{value}"' for key, value in command['args'].items())
        return f'{command["label"]}: "{command["name"]}", args: {args_string}'
    
    def add_resource(self, resource):
        self.resources.append(resource)

    def add_performance_evaluation(self, evaluation):
        self.performance_evaluation.append(evaluation)


    def _generate_numbered_list(self, items, item_type='list'):
        if item_type == 'command':
            return "\n".join(f"{i+1}. {self._generate_command_string(item)}" for i, item in enumerate(items))
        else:
            return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))

    def generate_prompt_string(self):
        formatted_response_format = json.dumps(self.response_format, indent=4)
        prompt_string = (
            f"Constraints:\n{self._generate_numbered_list(self.constraints)}\n\n"
            f"Commands:\n{self._generate_numbered_list(self.commands, item_type='command')}\n\n"
            f"Resources:\n{self._generate_numbered_list(self.resources)}\n\n"
            f"Performance Evaluation:\n{self._generate_numbered_list(self.performance_evaluation)}\n\n"
            f"You should only respond in JSON format as described below \nResponse Format: \n{formatted_response_format} \nEnsure the response can be parsed by Python json.loads"
        )

        return prompt_string
