import json
from typing import Any

from typesafe_sdk import Choice

from backend.blocks._base import BlockCategory, BlockOutput
from backend.data.model import SchemaField

from ._base import JevBlockBase
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, TypeSafeCredentials
from ._schemas import JevInput, JevOutput
from ._test import TEST_TRANSPARENCY, mock_result

CANDIDATE_DESCRIPTION_LIMIT = 2000
PICK_ANSWER = {
    "type": "choice",
    "choice": "candidate_2",
    "probabilities": {"candidate_1": 0.2, "candidate_2": 0.8},
    "confidence": 0.8,
}


class JevPickBestBlock(JevBlockBase):
    class Input(JevInput):
        candidates: list[Any] = SchemaField(
            description="Candidates to compare; descriptions are compact JSON capped at 2,000 characters each.",
            min_length=2,
        )
        question: str = SchemaField(
            description="Plain-language comparison, for example: Which candidate best meets the requirements in the state?",
            min_length=1,
        )

    class Output(JevOutput):
        best: Any = SchemaField(description="Original candidate selected by Jev.")
        best_index: int = SchemaField(
            description="Zero-based index of the selected candidate."
        )
        ranked: list[Any] = SchemaField(
            description="Original candidates sorted by returned probability, highest first; ties retain input order."
        )
        probabilities: dict[str, float] = SchemaField(
            description="Probabilities keyed by candidate_1, candidate_2, and so on."
        )

    def __init__(self):
        super().__init__(
            id="fdae49f8-685a-41ef-80a0-597566b5d4d7",
            description="Choose the best candidate using Jev and rank candidates directly by its probabilities. State the comparison criteria in plain language.",
            categories={BlockCategory.AI},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "state": "test",
                "question": "Which candidate meets the requirement?",
                "candidates": ["first", "second"],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=TEST_TRANSPARENCY
            + [
                ("best", "second"),
                ("best_index", 1),
                ("ranked", ["second", "first"]),
                ("probabilities", PICK_ANSWER["probabilities"]),
            ],
            test_mock={
                "call_jev": lambda *args, **kwargs: mock_result(
                    {"judgment": PICK_ANSWER}
                )
            },
        )

    async def run(
        self, input_data: Input, *, credentials: TypeSafeCredentials, **kwargs
    ) -> BlockOutput:
        options, note = candidate_options(input_data.candidates)
        result = await self.call_jev(
            credentials,
            input_data.state,
            {"judgment": Choice(instructions=input_data.question, criteria=options)},
        )
        if note:
            result.truncated = True
            result.truncation_note = " ".join(
                filter(None, [result.truncation_note, note])
            )
        for output in self.transparency(result):
            yield output
        if result.error:
            return
        answer = result.answers["judgment"]
        labels = list(options)
        best_index = labels.index(answer["choice"])
        indices = sorted(
            range(len(labels)),
            key=lambda index: answer["probabilities"][labels[index]],
            reverse=True,
        )
        yield "best", input_data.candidates[best_index]
        yield "best_index", best_index
        yield "ranked", [input_data.candidates[index] for index in indices]
        yield "probabilities", answer["probabilities"]


def candidate_options(candidates: list[Any]) -> tuple[dict[str, str], str]:
    options: dict[str, str] = {}
    shortened: list[str] = []
    for index, candidate in enumerate(candidates):
        label = f"candidate_{index + 1}"
        text = json.dumps(
            candidate, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        options[label] = text[:CANDIDATE_DESCRIPTION_LIMIT]
        if len(text) > CANDIDATE_DESCRIPTION_LIMIT:
            shortened.append(label)
    note = (
        f"Candidate descriptions capped at {CANDIDATE_DESCRIPTION_LIMIT} characters: {', '.join(shortened)}."
        if shortened
        else ""
    )
    return options, note
