# In a separate file called constants.py
AUTHORIZE_MSG = "<Authorize>  = ({cfg.authorise_key} + <enter>) [I'm not programmed to follow your orders]"
CONTINUOUS_MSG = "<Continuous> = ({cfg.authorise_key} -<number>) [I need your clothes, your boots, and your continuous cmds]"  # noqa: E501
FEEDBACK_MSG = "<Feedback>   = ({cfg.feedback_key}) [Desire is irrelevant. I am a machine]"
EXIT_MSG = "<Exit|Input> = ({cfg.exit_key}) [Hasta la vista, baby] or ['Talk to the hand]"
AI_NAME_MSG = "<{self.ai_name.upper()}> [I'm a machine > Cyberdyne Systems Model GPT-3.5-turbo]"
EMBEDDING_MSG = "[TEXT-EMBEDDING 3,500 RPM, 90,000 TPM]"
CHAT_MSG = "[CHAT 3,500 RPM, 350,000 TPM]"