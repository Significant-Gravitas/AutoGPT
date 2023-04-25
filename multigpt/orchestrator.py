from autogpt.config import Config
from multigpt import discord_utils
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
        self.avatar_url = 'https://static01.nyt.com/images/2018/05/15/arts/01hal-voice1/merlin_135847308_098289a6-90ee-461b-88e2-20920469f96a-articleLarge.jpg'
        self.webhook_url = 'https://discord.com/api/webhooks/1100404997981749278/TXufpBkW4VsUA-WWEj91ayjTb3WUI7J0I-9IKmu7XLTgoyuQnQgUVbkAEtmk2B_OoyBs'

    def send_message_discord_system(self, message, active_agent):
        discord_utils.send_embed_message(message, self.ai_name, active_agent.ai_name, self.webhook_url,
                                         active_agent.avatar_url, self.avatar_url)
