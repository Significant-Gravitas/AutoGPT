"""
Notion API helper functions and client for making authenticated requests.
"""

from typing import Any, Dict, List, Optional

from backend.data.model import OAuth2Credentials
from backend.util.request import Requests

NOTION_VERSION = "2022-06-28"


class NotionAPIException(Exception):
    """Exception raised for Notion API errors."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class NotionClient:
    """Client for interacting with the Notion API."""

    def __init__(self, credentials: OAuth2Credentials):
        self.credentials = credentials
        self.headers = {
            "Authorization": credentials.auth_header(),
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }
        self.requests = Requests()

    async def get_page(self, page_id: str) -> dict:
        """
        Fetch a page by ID.

        Args:
            page_id: The ID of the page to fetch.

        Returns:
            The page object from Notion API.
        """
        url = f"https://api.notion.com/v1/pages/{page_id}"
        response = await self.requests.get(url, headers=self.headers)

        if not response.ok:
            raise NotionAPIException(
                f"Failed to fetch page: {response.status} - {response.text()}",
                response.status,
            )

        return response.json()

    async def get_blocks(self, block_id: str, recursive: bool = True) -> List[dict]:
        """
        Fetch all blocks from a page or block.

        Args:
            block_id: The ID of the page or block to fetch children from.
            recursive: Whether to fetch nested blocks recursively.

        Returns:
            List of block objects.
        """
        blocks = []
        cursor = None

        while True:
            url = f"https://api.notion.com/v1/blocks/{block_id}/children"
            params = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor

            response = await self.requests.get(url, headers=self.headers, params=params)

            if not response.ok:
                raise NotionAPIException(
                    f"Failed to fetch blocks: {response.status} - {response.text()}",
                    response.status,
                )

            data = response.json()
            current_blocks = data.get("results", [])

            # If recursive, fetch children for blocks that have them
            if recursive:
                for block in current_blocks:
                    if block.get("has_children"):
                        block["children"] = await self.get_blocks(
                            block["id"], recursive=True
                        )

            blocks.extend(current_blocks)

            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")

        return blocks

    async def query_database(
        self,
        database_id: str,
        filter_obj: Optional[dict] = None,
        sorts: Optional[List[dict]] = None,
        page_size: int = 100,
    ) -> dict:
        """
        Query a database with optional filters and sorts.

        Args:
            database_id: The ID of the database to query.
            filter_obj: Optional filter object for the query.
            sorts: Optional list of sort objects.
            page_size: Number of results per page.

        Returns:
            Query results including pages and pagination info.
        """
        url = f"https://api.notion.com/v1/databases/{database_id}/query"

        payload: Dict[str, Any] = {"page_size": page_size}
        if filter_obj:
            payload["filter"] = filter_obj
        if sorts:
            payload["sorts"] = sorts

        response = await self.requests.post(url, headers=self.headers, json=payload)

        if not response.ok:
            raise NotionAPIException(
                f"Failed to query database: {response.status} - {response.text()}",
                response.status,
            )

        return response.json()

    async def create_page(
        self,
        parent: dict,
        properties: dict,
        children: Optional[List[dict]] = None,
        icon: Optional[dict] = None,
        cover: Optional[dict] = None,
    ) -> dict:
        """
        Create a new page.

        Args:
            parent: Parent object (page_id or database_id).
            properties: Page properties.
            children: Optional list of block children.
            icon: Optional icon object.
            cover: Optional cover object.

        Returns:
            The created page object.
        """
        url = "https://api.notion.com/v1/pages"

        payload: Dict[str, Any] = {"parent": parent, "properties": properties}

        if children:
            payload["children"] = children
        if icon:
            payload["icon"] = icon
        if cover:
            payload["cover"] = cover

        response = await self.requests.post(url, headers=self.headers, json=payload)

        if not response.ok:
            raise NotionAPIException(
                f"Failed to create page: {response.status} - {response.text()}",
                response.status,
            )

        return response.json()

    async def update_page(self, page_id: str, properties: dict) -> dict:
        """
        Update a page's properties.

        Args:
            page_id: The ID of the page to update.
            properties: Properties to update.

        Returns:
            The updated page object.
        """
        url = f"https://api.notion.com/v1/pages/{page_id}"

        response = await self.requests.patch(
            url, headers=self.headers, json={"properties": properties}
        )

        if not response.ok:
            raise NotionAPIException(
                f"Failed to update page: {response.status} - {response.text()}",
                response.status,
            )

        return response.json()

    async def append_blocks(self, block_id: str, children: List[dict]) -> dict:
        """
        Append blocks to a page or block.

        Args:
            block_id: The ID of the page or block to append to.
            children: List of block objects to append.

        Returns:
            Response with the created blocks.
        """
        url = f"https://api.notion.com/v1/blocks/{block_id}/children"

        response = await self.requests.patch(
            url, headers=self.headers, json={"children": children}
        )

        if not response.ok:
            raise NotionAPIException(
                f"Failed to append blocks: {response.status} - {response.text()}",
                response.status,
            )

        return response.json()

    async def search(
        self,
        query: str = "",
        filter_obj: Optional[dict] = None,
        sort: Optional[dict] = None,
        page_size: int = 100,
    ) -> dict:
        """
        Search for pages and databases.

        Args:
            query: Search query text.
            filter_obj: Optional filter object.
            sort: Optional sort object.
            page_size: Number of results per page.

        Returns:
            Search results.
        """
        url = "https://api.notion.com/v1/search"

        payload: Dict[str, Any] = {"page_size": page_size}
        if query:
            payload["query"] = query
        if filter_obj:
            payload["filter"] = filter_obj
        if sort:
            payload["sort"] = sort

        response = await self.requests.post(url, headers=self.headers, json=payload)

        if not response.ok:
            raise NotionAPIException(
                f"Search failed: {response.status} - {response.text()}", response.status
            )

        return response.json()


# Conversion helper functions


def parse_rich_text(rich_text_array: List[dict]) -> str:
    """
    Extract plain text from a Notion rich text array.

    Args:
        rich_text_array: Array of rich text objects from Notion.

    Returns:
        Plain text string.
    """
    if not rich_text_array:
        return ""

    text_parts = []
    for text_obj in rich_text_array:
        if "plain_text" in text_obj:
            text_parts.append(text_obj["plain_text"])

    return "".join(text_parts)


def rich_text_to_markdown(rich_text_array: List[dict]) -> str:
    """
    Convert Notion rich text array to markdown with formatting.

    Args:
        rich_text_array: Array of rich text objects from Notion.

    Returns:
        Markdown formatted string.
    """
    if not rich_text_array:
        return ""

    markdown_parts = []

    for text_obj in rich_text_array:
        text = text_obj.get("plain_text", "")
        annotations = text_obj.get("annotations", {})

        # Apply formatting based on annotations
        if annotations.get("code"):
            text = f"`{text}`"
        else:
            if annotations.get("bold"):
                text = f"**{text}**"
            if annotations.get("italic"):
                text = f"*{text}*"
            if annotations.get("strikethrough"):
                text = f"~~{text}~~"
            if annotations.get("underline"):
                text = f"<u>{text}</u>"

        # Handle links
        if text_obj.get("href"):
            text = f"[{text}]({text_obj['href']})"

        markdown_parts.append(text)

    return "".join(markdown_parts)


def block_to_markdown(block: dict, indent_level: int = 0) -> str:
    """
    Convert a single Notion block to markdown.

    Args:
        block: Block object from Notion API.
        indent_level: Current indentation level for nested blocks.

    Returns:
        Markdown string representation of the block.
    """
    block_type = block.get("type")
    indent = "  " * indent_level
    markdown_lines = []

    # Handle different block types
    if block_type == "paragraph":
        text = rich_text_to_markdown(block["paragraph"].get("rich_text", []))
        if text:
            markdown_lines.append(f"{indent}{text}")

    elif block_type == "heading_1":
        text = parse_rich_text(block["heading_1"].get("rich_text", []))
        markdown_lines.append(f"{indent}# {text}")

    elif block_type == "heading_2":
        text = parse_rich_text(block["heading_2"].get("rich_text", []))
        markdown_lines.append(f"{indent}## {text}")

    elif block_type == "heading_3":
        text = parse_rich_text(block["heading_3"].get("rich_text", []))
        markdown_lines.append(f"{indent}### {text}")

    elif block_type == "bulleted_list_item":
        text = rich_text_to_markdown(block["bulleted_list_item"].get("rich_text", []))
        markdown_lines.append(f"{indent}- {text}")

    elif block_type == "numbered_list_item":
        text = rich_text_to_markdown(block["numbered_list_item"].get("rich_text", []))
        # Note: This is simplified - proper numbering would need context
        markdown_lines.append(f"{indent}1. {text}")

    elif block_type == "to_do":
        text = rich_text_to_markdown(block["to_do"].get("rich_text", []))
        checked = "x" if block["to_do"].get("checked") else " "
        markdown_lines.append(f"{indent}- [{checked}] {text}")

    elif block_type == "toggle":
        text = rich_text_to_markdown(block["toggle"].get("rich_text", []))
        markdown_lines.append(f"{indent}<details>")
        markdown_lines.append(f"{indent}<summary>{text}</summary>")
        markdown_lines.append(f"{indent}")
        # Process children if they exist
        if block.get("children"):
            for child in block["children"]:
                child_markdown = block_to_markdown(child, indent_level + 1)
                if child_markdown:
                    markdown_lines.append(child_markdown)
        markdown_lines.append(f"{indent}</details>")

    elif block_type == "code":
        code = parse_rich_text(block["code"].get("rich_text", []))
        language = block["code"].get("language", "")
        markdown_lines.append(f"{indent}```{language}")
        markdown_lines.append(f"{indent}{code}")
        markdown_lines.append(f"{indent}```")

    elif block_type == "quote":
        text = rich_text_to_markdown(block["quote"].get("rich_text", []))
        markdown_lines.append(f"{indent}> {text}")

    elif block_type == "divider":
        markdown_lines.append(f"{indent}---")

    elif block_type == "image":
        image = block["image"]
        url = image.get("external", {}).get("url") or image.get("file", {}).get(
            "url", ""
        )
        caption = parse_rich_text(image.get("caption", []))
        alt_text = caption if caption else "Image"
        markdown_lines.append(f"{indent}![{alt_text}]({url})")
        if caption:
            markdown_lines.append(f"{indent}*{caption}*")

    elif block_type == "video":
        video = block["video"]
        url = video.get("external", {}).get("url") or video.get("file", {}).get(
            "url", ""
        )
        caption = parse_rich_text(video.get("caption", []))
        markdown_lines.append(f"{indent}[Video]({url})")
        if caption:
            markdown_lines.append(f"{indent}*{caption}*")

    elif block_type == "file":
        file = block["file"]
        url = file.get("external", {}).get("url") or file.get("file", {}).get("url", "")
        caption = parse_rich_text(file.get("caption", []))
        name = caption if caption else "File"
        markdown_lines.append(f"{indent}[{name}]({url})")

    elif block_type == "bookmark":
        url = block["bookmark"].get("url", "")
        caption = parse_rich_text(block["bookmark"].get("caption", []))
        markdown_lines.append(f"{indent}[{caption if caption else url}]({url})")

    elif block_type == "equation":
        expression = block["equation"].get("expression", "")
        markdown_lines.append(f"{indent}$${expression}$$")

    elif block_type == "callout":
        text = rich_text_to_markdown(block["callout"].get("rich_text", []))
        icon = block["callout"].get("icon", {})
        if icon.get("emoji"):
            markdown_lines.append(f"{indent}> {icon['emoji']} {text}")
        else:
            markdown_lines.append(f"{indent}> ℹ️ {text}")

    elif block_type == "child_page":
        title = block["child_page"].get("title", "Untitled")
        markdown_lines.append(f"{indent}📄 [{title}](notion://page/{block['id']})")

    elif block_type == "child_database":
        title = block["child_database"].get("title", "Untitled Database")
        markdown_lines.append(f"{indent}🗂️ [{title}](notion://database/{block['id']})")

    elif block_type == "table":
        # Tables are complex - for now just indicate there's a table
        markdown_lines.append(
            f"{indent}[Table with {block['table'].get('table_width', 0)} columns]"
        )

    elif block_type == "column_list":
        # Process columns
        if block.get("children"):
            markdown_lines.append(f"{indent}<div style='display: flex'>")
            for column in block["children"]:
                markdown_lines.append(f"{indent}<div style='flex: 1'>")
                if column.get("children"):
                    for child in column["children"]:
                        child_markdown = block_to_markdown(child, indent_level + 1)
                        if child_markdown:
                            markdown_lines.append(child_markdown)
                markdown_lines.append(f"{indent}</div>")
            markdown_lines.append(f"{indent}</div>")

    # Handle children for blocks that haven't been processed yet
    elif block.get("children") and block_type not in ["toggle", "column_list"]:
        for child in block["children"]:
            child_markdown = block_to_markdown(child, indent_level)
            if child_markdown:
                markdown_lines.append(child_markdown)

    return "\n".join(markdown_lines) if markdown_lines else ""


def blocks_to_markdown(blocks: List[dict]) -> str:
    """
    Convert a list of Notion blocks to a markdown document.

    Args:
        blocks: List of block objects from Notion API.

    Returns:
        Complete markdown document as a string.
    """
    markdown_parts = []

    for i, block in enumerate(blocks):
        markdown = block_to_markdown(block)
        if markdown:
            markdown_parts.append(markdown)
            # Add spacing between top-level blocks (except lists)
            if i < len(blocks) - 1:
                next_type = blocks[i + 1].get("type", "")
                current_type = block.get("type", "")
                # Don't add extra spacing between list items
                list_types = {"bulleted_list_item", "numbered_list_item", "to_do"}
                if not (current_type in list_types and next_type in list_types):
                    markdown_parts.append("")

    return "\n".join(markdown_parts)


def extract_page_title(page: dict) -> str:
    """
    Extract the title from a Notion page object.

    Args:
        page: Page object from Notion API.

    Returns:
        Page title as a string.
    """
    properties = page.get("properties", {})

    # Find the title property (it has type "title")
    for prop_name, prop_value in properties.items():
        if prop_value.get("type") == "title":
            return parse_rich_text(prop_value.get("title", []))

    return "Untitled"
