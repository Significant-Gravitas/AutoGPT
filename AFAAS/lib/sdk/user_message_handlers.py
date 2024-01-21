import click


class UserMessageHandlers:
    @staticmethod
    async def user_input_handler(prompt):
        user_input = click.prompt(
            prompt,
            default="y",
        )
        return user_input

    @staticmethod
    async def user_message_handler(prompt):
        print(prompt)
