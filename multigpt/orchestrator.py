from autogpt.config import Config
from multigpt.expert import Expert
from multigpt.multi_agent import MultiAgent
from multigpt.multi_prompt_generator import MultiPromptGenerator


class Orchestrator(MultiAgent):

    def __init__(
            self,
            ai_name,
            memory,
            full_message_history,
            prompt,
            user_input,
            agent_id
    ):
        super().__init__(
            ai_name=ai_name,
            memory=memory,
            full_message_history=full_message_history,
            prompt=prompt,
            user_input=user_input,
            agent_id=agent_id,
        )

        self.auditory_buffer = []  # contains the non processed parts of the conversation
