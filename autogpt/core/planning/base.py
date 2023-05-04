import abc

class Planner(abc.ABC):
    """Build prompts based on inputs, can potentially store and retrieve planning state from the workspace"""

    def __init__(config, logger, plugin_manager, workspace, *arg, **kwargs):
        pass

    def construct_goals_prompt(big_task: str) -> str:
        """This method is called upon the creation of an agent to define in more detail its goals"""
        pass

    def construct_next_step_prompt(
        context,  # Like a dict of message history, prior commands, current time, etc.
        memory,  # Maybe an explicit handle for the memory backend
        goals,
        tasks_queue: dict = None,
        # Carefully consider if there are other dependencies
    ) -> str:
        """This function makes the agent figure out what to do next based on its state"""

        def check_tasks_notification_queue(n: int) -> dict:
            """Returns a number of events in the queue that are closest to the current time.
            This summarizes each task except the first one"""
            pass

        tasks_queue = check_tasks_notification_queue(5)

        def get_important_messages(context: list[str]) -> list[str]:
            """takes in the context and seniority of the message senders to infer the most important action to take on"""
            pass

        pass

    def update_plan(
        plan,
        context,  # Like a dict of message history, prior commands, etc.
    ) -> str:
        """This function reads the plan, updates the status of the tasks and returnts an updated plan"""
        pass
