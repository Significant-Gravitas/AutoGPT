"""
Advanced Provider Example using the SDK

This demonstrates more advanced provider configurations including:
1. API Key authentication
2. Custom API client integration
3. Error handling patterns
4. Multiple provider configurations
"""

import logging

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import advanced_service, custom_api

logger = logging.getLogger(__name__)


class AdvancedProviderBlock(Block):
    """
    Advanced example block demonstrating API key authentication
    and provider configuration features.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = advanced_service.credentials_field(
            description="Credentials for Advanced Service",
        )
        operation: str = SchemaField(
            description="Operation to perform",
            default="read",
        )
        data: str = SchemaField(
            description="Data to process",
            default="",
        )

    class Output(BlockSchema):
        result: str = SchemaField(description="Operation result")
        auth_type: str = SchemaField(description="Authentication type used")
        success: bool = SchemaField(description="Whether operation succeeded")

    def __init__(self):
        super().__init__(
            id="d0086843-4c6c-4b9a-a490-d0e7b4cb317e",
            description="Advanced provider example with multiple auth types",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=AdvancedProviderBlock.Input,
            output_schema=AdvancedProviderBlock.Output,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            logger.debug(
                "Starting AdvancedProviderBlock run with operation: %s",
                input_data.operation,
            )
            # Use API key authentication
            _ = (
                credentials.api_key.get_secret_value()
            )  # Would be used in real implementation
            logger.debug("Successfully authenticated with API key")

            result = f"Performed {input_data.operation} with API key auth"
            logger.debug("Operation completed successfully")

            yield "result", result
            yield "auth_type", "api_key"
            yield "success", True

        except Exception as e:
            logger.error("Error in AdvancedProviderBlock: %s", str(e))
            yield "result", f"Error: {str(e)}"
            yield "auth_type", "error"
            yield "success", False


class CustomAPIBlock(Block):
    """
    Example block using a provider with custom API client.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = custom_api.credentials_field(
            description="Credentials for Custom API",
        )
        endpoint: str = SchemaField(
            description="API endpoint to call",
            default="/data",
        )
        payload: str = SchemaField(
            description="Payload to send",
            default="{}",
        )

    class Output(BlockSchema):
        response: str = SchemaField(description="API response")
        status: str = SchemaField(description="Response status")

    def __init__(self):
        super().__init__(
            id="979ccdfd-db5a-4179-ad57-aeb277999d79",
            description="Example using custom API client provider",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=CustomAPIBlock.Input,
            output_schema=CustomAPIBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            logger.debug(
                "Starting CustomAPIBlock run with endpoint: %s", input_data.endpoint
            )
            # Get API client from provider
            api_client = custom_api.get_api(credentials)
            logger.debug("Successfully obtained API client")

            # Make API request
            logger.debug("Making API request with payload: %s", input_data.payload)
            response = await api_client.request(
                method="POST",
                endpoint=input_data.endpoint,
                data=input_data.payload,
            )
            logger.debug("Received API response: %s", response)

            yield "response", str(response)
            yield "status", response.get("status", "unknown")

        except Exception as e:
            logger.error("Error in CustomAPIBlock: %s", str(e))
            yield "response", f"Error: {str(e)}"
            yield "status", "error"
