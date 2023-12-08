# def parse_agent_name_and_goals(name_and_goals: dict) -> str:
#     parsed_response = f"Agent Name: {name_and_goals['agent_name']}\n"
#     parsed_response += f"Agent Role: {name_and_goals['agent_role']}\n"
#     parsed_response += "Agent Goals:\n"
#     for i, goal in enumerate(name_and_goals["agent_goals"]):
#         parsed_response += f"{i+1}. {goal}\n"
#     return parsed_response


# def parse_agent_plan(plan: dict) -> str:
#     parsed_response = f"Agent Plan:\n"
#     for i, task in enumerate(plan.tasks):
#         parsed_response += f"{i+1}. {task.name}\n"
#         parsed_response += f"Description {task.description}\n"
#         parsed_response += f"Task type: {task.type}  "
#         parsed_response += f"Priority: {task.priority}\n"
#         parsed_response += f"Ready Criteria:\n"
#         for j, criteria in enumerate(task.ready_criteria):
#             parsed_response += f"    {j+1}. {criteria}\n"
#         parsed_response += f"Acceptance Criteria:\n"
#         for j, criteria in enumerate(task.acceptance_criteria):
#             parsed_response += f"    {j+1}. {criteria}\n"
#         parsed_response += "\n"

#     return parsed_response


# def parse_next_tool(current_task, next_tool: dict) -> str:
#     parsed_response = f"Current Task: {current_task.objective}\n"
#     ability_args = ", ".join(
#         f"{k}={v}" for k, v in next_tool["ability_arguments"].items()
#     )
#     parsed_response += f"Next Tool: {next_tool['next_ability']}({ability_args})\n"
#     parsed_response += f"Motivation: {next_tool['motivation']}\n"
#     parsed_response += f"Self-criticism: {next_tool['self_criticism']}\n"
#     parsed_response += f"Reasoning: {next_tool['reasoning']}\n"
#     return parsed_response


# def parse_ability_result(ability_result) -> str:
#     parsed_response = f"Tool: {ability_result['ability_name']}\n"
#     parsed_response += f"Tool Arguments: {ability_result['ability_args']}\n"
#     parsed_response += f"Tool Result: {ability_result['success']}\n"
#     parsed_response += f"Message: {ability_result['message']}\n"
#     parsed_response += f"Data: {ability_result['new_knowledge']}\n"
#     return parsed_response
