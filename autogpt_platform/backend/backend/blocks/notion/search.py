from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import OAuth2Credentials, SchemaField

from ._api import NotionClient, extract_page_title, parse_rich_text
from ._auth import (
    NOTION_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    NotionCredentialsField,
    NotionCredentialsInput,
)


class NotionSearchResult(BaseModel):
    """Typed model for Notion search results."""

    id: str
    type: str  # 'page' or 'database'
    title: str
    url: str
    created_time: Optional[str] = None
    last_edited_time: Optional[str] = None
    parent_type: Optional[str] = None  # 'page', 'database', or 'workspace'
    parent_id: Optional[str] = None
    icon: Optional[str] = None  # emoji icon if present
    is_inline: Optional[bool] = None  # for databases only


class NotionSearchBlock(Block):
    """Search across your Notion workspace for pages and databases."""

    class Input(BlockSchemaInput):
        credentials: NotionCredentialsInput = NotionCredentialsField()
        query: str = SchemaField(
            description="Search query text. Leave empty to get all accessible pages/databases.",
            default="",
        )
        filter_type: Optional[str] = SchemaField(
            description="Filter results by type: 'page' or 'database'. Leave empty for both.",
            default=None,
        )
        limit: int = SchemaField(
            description="Maximum number of results to return", default=20, ge=1, le=100
        )

    class Output(BlockSchemaOutput):
        results: List[NotionSearchResult] = SchemaField(
            description="List of search results with title, type, URL, and metadata."
        )
        result: NotionSearchResult = SchemaField(
            description="Individual search result (yields one per result found)."
        )
        result_ids: List[str] = SchemaField(
            description="List of IDs from search results for batch operations."
        )
        count: int = SchemaField(description="Number of results found.")

    def __init__(self):
        super().__init__(
            id="313515dd-9848-46ea-9cd6-3c627c892c56",
            description="Search your Notion workspace for pages and databases by text query.",
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.SEARCH},
            input_schema=NotionSearchBlock.Input,
            output_schema=NotionSearchBlock.Output,
            disabled=not NOTION_OAUTH_IS_CONFIGURED,
            test_input={
                "query": "project",
                "limit": 5,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                (
                    "results",
                    [
                        NotionSearchResult(
                            id="123",
                            type="page",
                            title="Project Plan",
                            url="https://notion.so/Project-Plan-123",
                        )
                    ],
                ),
                ("result_ids", ["123"]),
                (
                    "result",
                    NotionSearchResult(
                        id="123",
                        type="page",
                        title="Project Plan",
                        url="https://notion.so/Project-Plan-123",
                    ),
                ),
                ("count", 1),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={
                "search_workspace": lambda *args, **kwargs: (
                    [
                        NotionSearchResult(
                            id="123",
                            type="page",
                            title="Project Plan",
                            url="https://notion.so/Project-Plan-123",
                        )
                    ],
                    1,
                )
            },
        )

    @staticmethod
    async def search_workspace(
        credentials: OAuth2Credentials,
        query: str = "",
        filter_type: Optional[str] = None,
        limit: int = 20,
    ) -> tuple[List[NotionSearchResult], int]:
        """
        Search the Notion workspace.

        Returns:
            Tuple of (results_list, count)
        """
        client = NotionClient(credentials)

        # Build filter if type is specified
        filter_obj = None
        if filter_type:
            filter_obj = {"property": "object", "value": filter_type}

        # Execute search
        response = await client.search(
            query=query, filter_obj=filter_obj, page_size=limit
        )

        # Parse results
        results = []
        for item in response.get("results", []):
            result_data = {
                "id": item.get("id", ""),
                "type": item.get("object", ""),
                "url": item.get("url", ""),
                "created_time": item.get("created_time"),
                "last_edited_time": item.get("last_edited_time"),
                "title": "",  # Will be set below
            }

            # Extract title based on type
            if item.get("object") == "page":
                # For pages, get the title from properties
                result_data["title"] = extract_page_title(item)

                # Add parent info
                parent = item.get("parent", {})
                if parent.get("type") == "page_id":
                    result_data["parent_type"] = "page"
                    result_data["parent_id"] = parent.get("page_id")
                elif parent.get("type") == "database_id":
                    result_data["parent_type"] = "database"
                    result_data["parent_id"] = parent.get("database_id")
                elif parent.get("type") == "workspace":
                    result_data["parent_type"] = "workspace"

                # Add icon if present
                icon = item.get("icon")
                if icon and icon.get("type") == "emoji":
                    result_data["icon"] = icon.get("emoji")

            elif item.get("object") == "database":
                # For databases, get title from the title array
                result_data["title"] = parse_rich_text(item.get("title", []))

                # Add database-specific metadata
                result_data["is_inline"] = item.get("is_inline", False)

                # Add parent info
                parent = item.get("parent", {})
                if parent.get("type") == "page_id":
                    result_data["parent_type"] = "page"
                    result_data["parent_id"] = parent.get("page_id")
                elif parent.get("type") == "workspace":
                    result_data["parent_type"] = "workspace"

                # Add icon if present
                icon = item.get("icon")
                if icon and icon.get("type") == "emoji":
                    result_data["icon"] = icon.get("emoji")

            results.append(NotionSearchResult(**result_data))

        return results, len(results)

    async def run(
        self,
        input_data: Input,
        *,
        credentials: OAuth2Credentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            results, count = await self.search_workspace(
                credentials, input_data.query, input_data.filter_type, input_data.limit
            )

            # Yield the complete list for batch operations
            yield "results", results

            # Extract and yield IDs as a list for batch operations
            result_ids = [r.id for r in results]
            yield "result_ids", result_ids

            # Yield each individual result for single connections
            for result in results:
                yield "result", result

            yield "count", count
        except Exception as e:
            yield "error", str(e) if str(e) else "Unknown error"
