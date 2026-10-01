from backend.sdk import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    SchemaField,
)
from backend.util.exceptions import BlockExecutionError

from ._maps_code_assist_api import (
    CLIENT_SOURCE,
    MapsCodeAssistError,
    MapsPlatformDocPassage,
    call_maps_code_assist,
    to_maps_passage,
)

_TEST_CONTEXT = {
    "text": (
        "Use markers to display single locations on a map.\\n\\n```js\\n"
        "new AdvancedMarkerElement({ map, position });\\n```"
    ),
    "score": 0.77,
    "documentationUri": (
        "developers.google.com/maps/documentation/javascript/advanced-markers/add-marker"
    ),
    "apiState": "CURRENT",
}
_TEST_PASSAGE = to_maps_passage(_TEST_CONTEXT)
_TEST_INSTRUCTIONS = [
    "<identity>You are an expert AI pair programmer.</identity>",
    "The user accessing this service acknowledges the Maps Platform Terms of Service.",
]


class SearchGoogleMapsPlatformDocsBlock(Block):
    """Search Google Maps Platform documentation and sample code."""

    class Input(BlockSchemaInput):
        query: str = SchemaField(
            description=(
                "What you want to know, e.g. 'How do I add a marker with the Maps "
                "JavaScript API?'"
            )
        )
        product_filter: str = SchemaField(
            description="Narrow the search to an API or product area, e.g. 'Places API'",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        results: list[MapsPlatformDocPassage] = SchemaField(
            description="Matching passages, most relevant first"
        )
        result: MapsPlatformDocPassage = SchemaField(
            description="Each matching passage"
        )

    def __init__(self):
        super().__init__(
            id="a08bc6ba-423f-4fb1-8134-edc900f8ce3f",
            description=(
                "Search Google Maps Platform documentation and code samples and "
                "return the best-matching passages with their source links. "
                "Covers the Maps, Routes and Places APIs and SDKs, architecture "
                "guides and Google's official GitHub samples."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.DEVELOPER_TOOLS},
            input_schema=SearchGoogleMapsPlatformDocsBlock.Input,
            output_schema=SearchGoogleMapsPlatformDocsBlock.Output,
            test_input={"query": "How do I add a marker to a map?"},
            test_output=[("results", [_TEST_PASSAGE]), ("result", _TEST_PASSAGE)],
            test_mock={
                "_retrieve": lambda *args, **kwargs: {"contexts": [_TEST_CONTEXT]}
            },
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        arguments = {"llmQuery": input_data.query, "source": CLIENT_SOURCE}
        if product_filter := input_data.product_filter.strip():
            arguments["filter"] = product_filter
        try:
            result = await self._retrieve(arguments)
        except MapsCodeAssistError as e:
            raise BlockExecutionError(
                message=str(e), block_name=self.name, block_id=self.id
            ) from e

        passages = [to_maps_passage(context) for context in result.get("contexts", [])]
        yield "results", passages
        for passage in passages:
            yield "result", passage

    @staticmethod
    async def _retrieve(arguments: dict[str, str]) -> dict:
        return await call_maps_code_assist(
            "retrieve-google-maps-platform-docs", arguments
        )


class GetGoogleMapsPlatformCodingInstructionsBlock(Block):
    """Get Google's instructions for AI assistants that write Maps Platform code."""

    class Input(BlockSchemaInput):
        pass

    class Output(BlockSchemaOutput):
        instructions: str = SchemaField(
            description=(
                "Google's instructions for an AI assistant that writes Google "
                "Maps Platform code, ready to use as a system prompt"
            )
        )

    def __init__(self):
        super().__init__(
            id="c78f0f57-18c3-47f3-ac76-009032793faa",
            description=(
                "Get Google's system prompt for AI assistants that write Google "
                "Maps Platform code. It covers how to plan answers and ground "
                "them with Search Google Maps Platform Docs, which terms apply "
                "(including the EEA terms) and how to cite sources."
            ),
            categories={BlockCategory.AI, BlockCategory.DEVELOPER_TOOLS},
            input_schema=GetGoogleMapsPlatformCodingInstructionsBlock.Input,
            output_schema=GetGoogleMapsPlatformCodingInstructionsBlock.Output,
            # One empty input: the block takes none, and an empty dict alone
            # would make the self-test skip it.
            test_input=[{}],
            test_output=[("instructions", "\n\n".join(_TEST_INSTRUCTIONS))],
            test_mock={
                "_retrieve_instructions": lambda *args, **kwargs: {
                    "systemInstructions": _TEST_INSTRUCTIONS
                }
            },
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            result = await self._retrieve_instructions()
        except MapsCodeAssistError as e:
            raise BlockExecutionError(
                message=str(e), block_name=self.name, block_id=self.id
            ) from e
        sections = result.get("systemInstructions") or []
        yield "instructions", "\n\n".join(section for section in sections if section)

    @staticmethod
    async def _retrieve_instructions() -> dict:
        return await call_maps_code_assist(
            "retrieve-instructions", {"name": "instructions"}
        )
