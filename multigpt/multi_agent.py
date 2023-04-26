from autogpt.agent import Agent


class MultiAgent(Agent):

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
            next_action_count=0,
            prompt=prompt,
            user_input=user_input,
        )
        self.agent_id = agent_id
        self.auditory_buffer = []  # contains the non processed parts of the conversation

    def receive_message(self, speaker, message):
        self.auditory_buffer.append((speaker.ai_name, message))
