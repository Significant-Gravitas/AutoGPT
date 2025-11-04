from typing import Any

from firecrawl import FirecrawlApp

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockCost,
    BlockCostType,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
    cost,
)

from ._config import firecrawl


def normalize_to_json_schema(schema: dict | None) -> dict | None:
    """
    Normalize a simplified schema format into valid JSON Schema format.
    
    Transforms simplified schemas like {"field": "type"} into proper JSON Schema format:
    {"type": "object", "properties": {"field": {"type": "type"}}}
    
    If the schema already appears to be a valid JSON Schema (has "type" or "properties"),
    it is returned as-is.
    
    Args:
        schema: The schema to normalize, or None
        
    Returns:
        A valid JSON Schema dict, or None if input was None
    """
    if schema is None:
        return None
    
    # If it already has "type" at the root level, assume it's already a JSON Schema
    if "type" in schema:
        return schema
    
    # If it already has "properties", assume it's already a JSON Schema
    if "properties" in schema:
        return schema
    
    # Otherwise, treat it as a simplified format and transform it
    properties = {}
    for key, value in schema.items():
        if isinstance(value, str):
            # Simple type string like "string", "number", etc.
            properties[key] = {"type": value}
        elif isinstance(value, dict):
            # Already a property definition, use as-is
            properties[key] = value
        else:
            # Fallback: treat as any type
            properties[key] = {"type": "string"}
    
    return {
        "type": "object",
        "properties": properties,
    }


@cost(BlockCost(2, BlockCostType.RUN))
class FirecrawlExtractBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = firecrawl.credentials_field()
        urls: list[str] = SchemaField(
            description="The URLs to crawl - at least one is required. Wildcards are supported. (/*)"
        )
        prompt: str | None = SchemaField(
            description="The prompt to use for the crawl", default=None, advanced=False
        )
        output_schema: dict | None = SchemaField(
            description='A JSON Schema describing the output structure. Supports both simplified format (e.g., {"field": "string"}) and full JSON Schema format (e.g., {"type": "object", "properties": {"field": {"type": "string"}}}).',
            default=None,
        )
        enable_web_search: bool = SchemaField(
            description="When true, extraction can follow links outside the specified domain.",
            default=False,
        )

    class Output(BlockSchemaOutput):
        data: dict[str, Any] = SchemaField(description="The result of the crawl")
        error: str = SchemaField(
            description="Error message if the extraction failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="d1774756-4d9e-40e6-bab1-47ec0ccd81b2",
            description="Firecrawl crawls websites to extract comprehensive data while bypassing blockers.",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        app = FirecrawlApp(api_key=credentials.api_key.get_secret_value())

        # Normalize the schema to ensure it's in valid JSON Schema format
        normalized_schema = normalize_to_json_schema(input_data.output_schema)

        extract_result = app.extract(
            urls=input_data.urls,
            prompt=input_data.prompt,
            schema=normalized_schema,
            enable_web_search=input_data.enable_web_search,
        )

        yield "data", extract_result.data
