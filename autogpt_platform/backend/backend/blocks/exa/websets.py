from typing import Any, Dict, List, Optional

import requests

from backend.blocks.exa._auth import (
    ExaCredentials,
    ExaCredentialsField,
    ExaCredentialsInput,
)
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField

# --- Create Webset Block ---
class ExaCreateWebsetBlock(Block):
    """Block for creating a Webset using Exa's Websets API."""
    class Input(BlockSchema):
        credentials: ExaCredentialsInput = ExaCredentialsField()
        query: str = SchemaField(description="The search query for the webset")
        count: int = SchemaField(description="Number of results to return", default=5)
        enrichments: Optional[List[Dict[str, Any]]] = SchemaField(
            description="List of enrichment dicts (optional)", default_factory=list, advanced=True
        )
        external_id: Optional[str] = SchemaField(
            description="Optional external identifier", default=None, advanced=True
        )
        metadata: Optional[Dict[str, Any]] = SchemaField(
            description="Optional metadata", default_factory=dict, advanced=True
        )

    class Output(BlockSchema):
        webset: Optional[Dict[str, Any]] = SchemaField(
            description="The created webset object (or None if error)", default=None
        )
        error: str = SchemaField(
            description="Error message if the request failed", default=""
        )

    def __init__(self):
        """Initialize the ExaCreateWebsetBlock with its configuration."""
        super().__init__(
            id="322351cc-35d7-45ec-8920-9a3c98920411",
            description="Creates a Webset using Exa's Websets API",
            categories={BlockCategory.SEARCH},
            input_schema=ExaCreateWebsetBlock.Input,
            output_schema=ExaCreateWebsetBlock.Output,
        )

    def run(self, input_data: Input, *, credentials: ExaCredentials, **kwargs) -> BlockOutput:
        """
        Execute the block to create a webset with Exa's API.
        
        Args:
            input_data: The input parameters for creating a webset
            credentials: The Exa API credentials
            
        Yields:
            Either the created webset object or an error message
        """
        url = "https://api.exa.ai/websets/v0/websets"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }
        payload = {
            "search": {
                "query": input_data.query,
                "count": input_data.count,
            }
        }
        optional_fields = {}
        if isinstance(input_data.enrichments, list) and input_data.enrichments:
            optional_fields["enrichments"] = input_data.enrichments
        if isinstance(input_data.external_id, str) and input_data.external_id:
            optional_fields["externalId"] = input_data.external_id
        if isinstance(input_data.metadata, dict) and input_data.metadata:
            optional_fields["metadata"] = input_data.metadata
        payload.update(optional_fields)
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            yield "webset", data
        except Exception as e:
            yield "error", str(e)

# --- Get Webset Block ---
class ExaGetWebsetBlock(Block):
    """Block for retrieving a Webset by ID using Exa's Websets API."""
    class Input(BlockSchema):
        credentials: ExaCredentialsInput = ExaCredentialsField()
        webset_id: str = SchemaField(description="The Webset ID or externalId")
        expand_items: bool = SchemaField(description="Expand with items", default=False, advanced=True)

    class Output(BlockSchema):
        webset: Optional[Dict[str, Any]] = SchemaField(description="The webset object (or None if error)", default=None)
        error: str = SchemaField(description="Error message if the request failed", default="")

    def __init__(self):
        """Initialize the ExaGetWebsetBlock with its configuration."""
        super().__init__(
            id="f9229293-cddf-43fc-94b3-48cbd1a44618",
            description="Retrieves a Webset by ID using Exa's Websets API",
            categories={BlockCategory.SEARCH},
            input_schema=ExaGetWebsetBlock.Input,
            output_schema=ExaGetWebsetBlock.Output,
        )

    def run(self, input_data: Input, *, credentials: ExaCredentials, **kwargs) -> BlockOutput:
        """
        Execute the block to retrieve a webset by ID from Exa's API.
        
        Args:
            input_data: The input parameters including the webset ID
            credentials: The Exa API credentials
            
        Yields:
            Either the retrieved webset object or an error message
        """
        url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }
        params = {"expand": "items"} if input_data.expand_items else None
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            yield "webset", data
        except Exception as e:
            yield "error", str(e)

# --- Delete Webset Block ---
class ExaDeleteWebsetBlock(Block):
    """Block for deleting a Webset by ID using Exa's Websets API."""
    class Input(BlockSchema):
        credentials: ExaCredentialsInput = ExaCredentialsField()
        webset_id: str = SchemaField(description="The Webset ID or externalId")

    class Output(BlockSchema):
        deleted: Optional[Dict[str, Any]] = SchemaField(description="The deleted webset object (or None if error)", default=None)
        error: str = SchemaField(description="Error message if the request failed", default="")

    def __init__(self):
        """Initialize the ExaDeleteWebsetBlock with its configuration."""
        super().__init__(
            id="a082e162-274e-4167-a467-a1839e644cbd",
            description="Deletes a Webset by ID using Exa's Websets API",
            categories={BlockCategory.SEARCH},
            input_schema=ExaDeleteWebsetBlock.Input,
            output_schema=ExaDeleteWebsetBlock.Output,
        )

    def run(self, input_data: Input, *, credentials: ExaCredentials, **kwargs) -> BlockOutput:
        """
        Execute the block to delete a webset by ID using Exa's API.
        
        Args:
            input_data: The input parameters including the webset ID
            credentials: The Exa API credentials
            
        Yields:
            Either the deleted webset object or an error message
        """
        url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }
        try:
            response = requests.delete(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            yield "deleted", data
        except Exception as e:
            yield "error", str(e)

# --- Update Webset Block ---
class ExaUpdateWebsetBlock(Block):
    """Block for updating a Webset's metadata using Exa's Websets API."""
    class Input(BlockSchema):
        credentials: ExaCredentialsInput = ExaCredentialsField()
        webset_id: str = SchemaField(description="The Webset ID or externalId")
        metadata: Dict[str, Any] = SchemaField(description="Metadata to update", default_factory=dict)

    class Output(BlockSchema):
        webset: Optional[Dict[str, Any]] = SchemaField(description="The updated webset object (or None if error)", default=None)
        error: str = SchemaField(description="Error message if the request failed", default="")

    def __init__(self):
        """Initialize the ExaUpdateWebsetBlock with its configuration."""
        super().__init__(
            id="e0c81b70-ac38-4239-8ecd-a75c1737c9ef",
            description="Updates a Webset's metadata using Exa's Websets API",
            categories={BlockCategory.SEARCH},
            input_schema=ExaUpdateWebsetBlock.Input,
            output_schema=ExaUpdateWebsetBlock.Output,
        )

    def run(self, input_data: Input, *, credentials: ExaCredentials, **kwargs) -> BlockOutput:
        """
        Execute the block to update a webset's metadata using Exa's API.
        
        Args:
            input_data: The input parameters including the webset ID and metadata
            credentials: The Exa API credentials
            
        Yields:
            Either the updated webset object or an error message
        """
        url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }
        payload = {"metadata": input_data.metadata}
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            yield "webset", data
        except Exception as e:
            yield "error", str(e)

# --- List Websets Block ---
class ExaListWebsetsBlock(Block):
    """Block for listing all Websets using Exa's Websets API with pagination support."""
    class Input(BlockSchema):
        credentials: ExaCredentialsInput = ExaCredentialsField()
        limit: int = SchemaField(description="Number of websets to return (max 100)", default=25)
        cursor: Optional[str] = SchemaField(description="Pagination cursor (optional)", default=None, advanced=True)

    class Output(BlockSchema):
        data: Optional[List[Dict[str, Any]]] = SchemaField(description="List of websets", default=None)
        has_more: Optional[bool] = SchemaField(description="Whether there are more results", default=None)
        next_cursor: Optional[str] = SchemaField(description="Cursor for next page", default=None)
        error: str = SchemaField(description="Error message if the request failed", default="")

    def __init__(self):
        """Initialize the ExaListWebsetsBlock with its configuration."""
        super().__init__(
            id="887a2dae-c9c3-4ae5-a079-fe3b52be64e4",
            description="Lists all Websets using Exa's Websets API",
            categories={BlockCategory.SEARCH},
            input_schema=ExaListWebsetsBlock.Input,
            output_schema=ExaListWebsetsBlock.Output,
        )

    def run(self, input_data: Input, *, credentials: ExaCredentials, **kwargs) -> BlockOutput:
        """
        Execute the block to list websets with pagination using Exa's API.
        
        Args:
            input_data: The input parameters including limit and optional cursor
            credentials: The Exa API credentials
            
        Yields:
            The list of websets, pagination info, or an error message
        """
        url = "https://api.exa.ai/websets/v0/websets"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }
        params: dict[str, Any] = {"limit": int(input_data.limit)}
        if isinstance(input_data.cursor, str) and input_data.cursor:
            params["cursor"] = input_data.cursor
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            yield "data", data.get("data")
            yield "has_more", data.get("hasMore")
            yield "next_cursor", data.get("nextCursor")
        except Exception as e:
            yield "error", str(e)

# --- Cancel Webset Block ---
class ExaCancelWebsetBlock(Block):
    """Block for canceling a running Webset using Exa's Websets API."""
    class Input(BlockSchema):
        credentials: ExaCredentialsInput = ExaCredentialsField()
        webset_id: str = SchemaField(description="The Webset ID or externalId")

    class Output(BlockSchema):
        webset: Optional[Dict[str, Any]] = SchemaField(description="The canceled webset object (or None if error)", default=None)
        error: str = SchemaField(description="Error message if the request failed", default="")

    def __init__(self):
        """Initialize the ExaCancelWebsetBlock with its configuration."""
        super().__init__(
            id="f7f0b19c-71e8-4c2f-bc68-904a6a61faf7",
            description="Cancels a running Webset using Exa's Websets API",
            categories={BlockCategory.SEARCH},
            input_schema=ExaCancelWebsetBlock.Input,
            output_schema=ExaCancelWebsetBlock.Output,
        )

    def run(self, input_data: Input, *, credentials: ExaCredentials, **kwargs) -> BlockOutput:
        """
        Execute the block to cancel a running webset using Exa's API.
        
        Args:
            input_data: The input parameters including the webset ID
            credentials: The Exa API credentials
            
        Yields:
            Either the canceled webset object or an error message
        """
        url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}/cancel"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }
        try:
            response = requests.post(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            yield "webset", data
        except Exception as e:
            yield "error", str(e)
