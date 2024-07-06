def parse_agent_name_and_goals(name_and_goals: dict) -> str:
    parsed_response = f"Agent Name: {name_and_goals['agent_name']}\n"
    parsed_response += f"Agent Role: {name_and_goals['agent_role']}\n"
    parsed_response += "Agent Goals:\n"
    for i, goal in enumerate(name_and_goals["agent_goals"]):
        parsed_response += f"{i+1}. {goal}\n"
    return parsed_response


def parse_agent_plan(plan: dict) -> str:
    parsed_response = "Agent Plan:\n"
    for i, task in enumerate(plan["task_list"]):
        parsed_response += f"{i+1}. {task['objective']}\n"
        parsed_response += f"Task type: {task['type']}  "
        parsed_response += f"Priority: {task['priority']}\n"
        parsed_response += "Ready Criteria:\n"
        for j, criteria in enumerate(task["ready_criteria"]):
            parsed_response += f"    {j+1}. {criteria}\n"
        parsed_response += "Acceptance Criteria:\n"
        for j, criteria in enumerate(task["acceptance_criteria"]):
            parsed_response += f"    {j+1}. {criteria}\n"
        parsed_response += "\n"

    return parsed_response


def parse_next_ability(current_task, next_ability: dict) -> str:
    parsed_response = f"Current Task: {current_task.objective}\n"
    ability_args = ", ".join(
        f"{k}={v}" for k, v in next_ability["ability_arguments"].items()
    )
    parsed_response += f"Next Ability: {next_ability['next_ability']}({ability_args})\n"
    parsed_response += f"Motivation: {next_ability['motivation']}\n"
    parsed_response += f"Self-criticism: {next_ability['self_criticism']}\n"
    parsed_response += f"Reasoning: {next_ability['reasoning']}\n"
    return parsed_response


def parse_ability_result(ability_result) -> str:
    parsed_response = f"Ability: {ability_result['ability_name']}\n"
    parsed_response += f"Ability Arguments: {ability_result['ability_args']}\n"
    parsed_response += f"Ability Result: {ability_result['success']}\n"
    parsed_response += f"Message: {ability_result['message']}\n"
    parsed_response += f"Data: {ability_result['new_knowledge']}\n"
    return parsed_response
