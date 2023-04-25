from autogpt.agent import Agent
from multigpt import discord_utils


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
        self.avatar_url = 'https://discord-emojis.s3.eu-central-1.amazonaws.com/sid_icon_dark.png'
        self.webhook_url = 'https://discord.com/api/webhooks/1099637610445553774/sdpL-iKMyYeSnUYEAVMao_6zu64bdiwGsQ9OG7Gd_WMLkMYblWrRsUyUaEIzC_T1xDmw'

    def receive_message(self, speaker, message):
        self.auditory_buffer.append((speaker.ai_name, message))

    def send_message_discord(self, message):
        discord_utils.send_message(message, self.ai_name, self.webhook_url, self.avatar_url)
