# """
# Authentication module for Airtable API integration.

# This module provides credential types and test credentials for the Airtable API integration.
# It defines the structure for API key credentials used to authenticate with the Airtable API
# and provides mock credentials for testing purposes.
# """

# from typing import Literal

# from pydantic import SecretStr

# from backend.data.model import APIKeyCredentials, CredentialsMetaInput
# from backend.integrations.providers import ProviderName

# # Define the type of credentials input expected for Airtable API
# AirtableCredentialsInput = CredentialsMetaInput[
#     Literal[ProviderName.AIRTABLE], Literal["api_key"]
# ]


# # Mock credentials for testing Airtable API integration
# TEST_CREDENTIALS = APIKeyCredentials(
#     id="7a91c8f0-399f-4235-a79c-59c0e37454d5",
#     provider="airtable",
#     api_key=SecretStr("mock-airtable-api-key"),
#     title="Mock Airtable API key",
#     expires_at=None,
# )

# # Dictionary representation of test credentials for input fields
# TEST_CREDENTIALS_INPUT = {
#     "provider": TEST_CREDENTIALS.provider,
#     "id": TEST_CREDENTIALS.id,
#     "type": TEST_CREDENTIALS.type,
#     "title": TEST_CREDENTIALS.title,
# }
