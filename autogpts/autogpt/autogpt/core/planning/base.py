# class Planner(abc.ABC):
#     """
#     Manages the agent's planning and goal-setting
#     by constructing language model prompts.
#     """
#
#     @staticmethod
#     @abc.abstractmethod
#     async def decide_name_and_goals(
#         user_objective: str,
#     ) -> LanguageModelResponse:
#         """Decide the name and goals of an Agent from a user-defined objective.
#
#         Args:
#             user_objective: The user-defined objective for the agent.
#
#         Returns:
#             The agent name and goals as a response from the language model.
#
#         """
#         ...
#
#     @abc.abstractmethod
#     async def plan(self, context: PlanningContext) -> LanguageModelResponse:
#         """Plan the next ability for the Agent.
#
#         Args:
#             context: A context object containing information about the agent's
#                        progress, result, memories, and feedback.
#
#
#         Returns:
#             The next ability the agent should take along with thoughts and reasoning.
#
#         """
#         ...
#
#     @abc.abstractmethod
#     def reflect(
#         self,
#         context: ReflectionContext,
#     ) -> LanguageModelResponse:
#         """Reflect on a planned ability and provide self-criticism.
#
#
#         Args:
#             context: A context object containing information about the agent's
#                        reasoning, plan, thoughts, and criticism.
#
#         Returns:
#             Self-criticism about the agent's plan.
#
#         """
#         ...
