import json
from typing import Iterator

import click

from forge.agent.protocols import CommandProvider
from forge.command import Command, command
from forge.models.json_schema import JSONSchema
from forge.utils.const import ASK_COMMAND


class UserInteractionComponent(CommandProvider):
    """Provides commands to interact with the user."""

    def get_commands(self) -> Iterator[Command]:
        yield self.ask_user
        yield self.ask_yes_no
        yield self.ask_choice

    @command(
        names=[ASK_COMMAND],
        parameters={
            "question": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The question or prompt to the user",
                required=True,
            )
        },
    )
    def ask_user(self, question: str) -> str:
        """If you need more details or information regarding the given goals,
        you can ask the user for input."""
        print(f"\nQ: {question}")
        resp = click.prompt("A")
        return f"The user's answer: '{resp}'"

    @command(
        ["ask_yes_no", "confirm"],
        "Ask the user a yes/no confirmation question.",
        {
            "question": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The yes/no question to ask the user",
                required=True,
            ),
            "default": JSONSchema(
                type=JSONSchema.Type.BOOLEAN,
                description="Default if Enter pressed (None = require explicit)",
                required=False,
            ),
        },
    )
    def ask_yes_no(self, question: str, default: bool | None = None) -> str:
        """Ask the user a yes/no question.

        Args:
            question: The question to ask
            default: Optional default answer

        Returns:
            str: JSON with the user's answer (true/false)
        """
        if default is True:
            prompt_suffix = " [Y/n]"
        elif default is False:
            prompt_suffix = " [y/N]"
        else:
            prompt_suffix = " [y/n]"

        print(f"\nQ: {question}{prompt_suffix}")

        while True:
            resp = click.prompt("A", default="", show_default=False).strip().lower()

            if resp == "" and default is not None:
                answer = default
                break
            elif resp in ("y", "yes"):
                answer = True
                break
            elif resp in ("n", "no"):
                answer = False
                break
            else:
                print("Please enter 'y' or 'n'")

        return json.dumps(
            {
                "question": question,
                "answer": answer,
                "response": "yes" if answer else "no",
            }
        )

    @command(
        ["ask_choice", "select_option"],
        "Present multiple choices to the user and get their selection.",
        {
            "question": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The question to ask",
                required=True,
            ),
            "choices": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                items=JSONSchema(type=JSONSchema.Type.STRING),
                description="List of choices to present",
                required=True,
            ),
            "allow_multiple": JSONSchema(
                type=JSONSchema.Type.BOOLEAN,
                description="Allow selecting multiple choices (default: False)",
                required=False,
            ),
        },
    )
    def ask_choice(
        self, question: str, choices: list[str], allow_multiple: bool = False
    ) -> str:
        """Present choices to the user.

        Args:
            question: The question to ask
            choices: List of choices
            allow_multiple: Whether multiple selections are allowed

        Returns:
            str: JSON with selected choice(s)
        """
        print(f"\nQ: {question}")
        for i, choice in enumerate(choices, 1):
            print(f"  {i}. {choice}")

        if allow_multiple:
            print("Enter choice numbers separated by commas (e.g., '1,3,4'):")
        else:
            print("Enter choice number:")

        while True:
            resp = click.prompt("A", default="", show_default=False).strip()

            try:
                if allow_multiple:
                    indices = [int(x.strip()) for x in resp.split(",")]
                    if all(1 <= i <= len(choices) for i in indices):
                        selected = [choices[i - 1] for i in indices]
                        return json.dumps(
                            {
                                "question": question,
                                "selected": selected,
                                "indices": indices,
                            }
                        )
                else:
                    index = int(resp)
                    if 1 <= index <= len(choices):
                        selected = choices[index - 1]
                        return json.dumps(
                            {"question": question, "selected": selected, "index": index}
                        )

                print(f"Please enter a valid number between 1 and {len(choices)}")

            except ValueError:
                print("Please enter a valid number")
