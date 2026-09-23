from typing import Any

from typesafe_sdk import Choice

from backend.blocks._base import BlockCategory, BlockOutput
from backend.data.model import SchemaField

from ._base import JevBlockBase
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, TypeSafeCredentials
from ._schemas import JevInput, JevOutput
from ._test import CHOICE_ANSWER, TEST_OPTIONS, TEST_TRANSPARENCY, mock_result


class JevRouteBlock(JevBlockBase):
    class Input(JevInput):
        question: str = SchemaField(
            description="Plain-language judgment deciding the route.", min_length=1
        )
        options: dict[str, str] = SchemaField(
            description="Two to five option names and qualifying descriptions; dictionary order assigns option_1 to option_5.",
            min_length=2,
            max_length=5,
        )
        data: Any = SchemaField(
            default=None,
            advanced=False,
            description="Value forwarded unchanged on the winning route only.",
        )

    class Output(JevOutput):
        option_1: Any = SchemaField(
            description="Data when the first configured option wins."
        )
        option_2: Any = SchemaField(
            description="Data when the second configured option wins."
        )
        option_3: Any = SchemaField(
            description="Data when the third configured option wins."
        )
        option_4: Any = SchemaField(
            description="Data when the fourth configured option wins."
        )
        option_5: Any = SchemaField(
            description="Data when the fifth configured option wins."
        )
        choice: str = SchemaField(description="Winning option name returned by Jev.")
        probabilities: dict[str, float] = SchemaField(
            description="Probability for each configured option."
        )

    def __init__(self):
        super().__init__(
            id="529b586b-38e7-4ba8-a564-89542075f633",
            description="Route data using Jev's typed choice. Plain-language option descriptions define each route; only the winning pin fires.",
            categories={BlockCategory.AI, BlockCategory.LOGIC},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "state": "test",
                "question": "Is evidence present?",
                "options": TEST_OPTIONS,
                "data": {"id": 7},
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=TEST_TRANSPARENCY
            + [
                ("choice", "yes"),
                ("probabilities", CHOICE_ANSWER["probabilities"]),
                ("option_1", {"id": 7}),
            ],
            test_mock={
                "call_jev": lambda *args, **kwargs: mock_result(
                    {"judgment": CHOICE_ANSWER}
                )
            },
        )

    async def run(
        self, input_data: Input, *, credentials: TypeSafeCredentials, **kwargs
    ) -> BlockOutput:
        result = await self.call_jev(
            credentials,
            input_data.state,
            {
                "judgment": Choice(
                    instructions=input_data.question, criteria=input_data.options
                )
            },
        )
        for output in self.transparency(result):
            yield output
        if result.error:
            return
        answer = result.answers["judgment"]
        yield "choice", answer["choice"]
        yield "probabilities", answer["probabilities"]
        index = list(input_data.options).index(answer["choice"]) + 1
        yield f"option_{index}", input_data.data
