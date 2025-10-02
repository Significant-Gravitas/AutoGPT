import re
from pathlib import Path
from typing import Any

import regex  # Has built-in timeout support

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.util import json, text
from backend.util.file import get_exec_file_path, store_media_file
from backend.util.type import MediaFileType

formatter = text.TextFormatter()


class MatchTextPatternBlock(Block):
    class Input(BlockSchema):
        text: Any = SchemaField(description="Text to match")
        match: str = SchemaField(description="Pattern (Regex) to match")
        data: Any = SchemaField(description="Data to be forwarded to output")
        case_sensitive: bool = SchemaField(
            description="Case sensitive match", default=True
        )
        dot_all: bool = SchemaField(description="Dot matches all", default=True)

    class Output(BlockSchema):
        positive: Any = SchemaField(description="Output data if match is found")
        negative: Any = SchemaField(description="Output data if match is not found")

    def __init__(self):
        super().__init__(
            id="3060088f-6ed9-4928-9ba7-9c92823a7ccd",
            description="Matches text against a regex pattern and forwards data to positive or negative output based on the match.",
            categories={BlockCategory.TEXT},
            input_schema=MatchTextPatternBlock.Input,
            output_schema=MatchTextPatternBlock.Output,
            test_input=[
                {"text": "ABC", "match": "ab", "data": "X", "case_sensitive": False},
                {"text": "ABC", "match": "ab", "data": "Y", "case_sensitive": True},
                {"text": "Hello World!", "match": ".orld.+", "data": "Z"},
                {"text": "Hello World!", "match": "World![a-z]+", "data": "Z"},
            ],
            test_output=[
                ("positive", "X"),
                ("negative", "Y"),
                ("positive", "Z"),
                ("negative", "Z"),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        output = input_data.data or input_data.text
        flags = 0
        if not input_data.case_sensitive:
            flags = flags | re.IGNORECASE
        if input_data.dot_all:
            flags = flags | re.DOTALL

        if isinstance(input_data.text, str):
            text = input_data.text
        else:
            text = json.dumps(input_data.text)

        if re.search(input_data.match, text, flags=flags):
            yield "positive", output
        else:
            yield "negative", output


class ExtractTextInformationBlock(Block):
    class Input(BlockSchema):
        text: Any = SchemaField(description="Text to parse")
        pattern: str = SchemaField(description="Pattern (Regex) to parse")
        group: int = SchemaField(description="Group number to extract", default=0)
        case_sensitive: bool = SchemaField(
            description="Case sensitive match", default=True
        )
        dot_all: bool = SchemaField(description="Dot matches all", default=True)
        find_all: bool = SchemaField(description="Find all matches", default=False)

    class Output(BlockSchema):
        positive: str = SchemaField(description="Extracted text")
        negative: str = SchemaField(description="Original text")
        matched_results: list[str] = SchemaField(description="List of matched results")
        matched_count: int = SchemaField(description="Number of matched results")

    def __init__(self):
        super().__init__(
            id="3146e4fe-2cdd-4f29-bd12-0c9d5bb4deb0",
            description="This block extracts the text from the given text using the pattern (regex).",
            categories={BlockCategory.TEXT},
            input_schema=ExtractTextInformationBlock.Input,
            output_schema=ExtractTextInformationBlock.Output,
            test_input=[
                {"text": "Hello, World!", "pattern": "Hello, (.+)", "group": 1},
                {"text": "Hello, World!", "pattern": "Hello, (.+)", "group": 0},
                {"text": "Hello, World!", "pattern": "Hello, (.+)", "group": 2},
                {"text": "Hello, World!", "pattern": "hello,", "case_sensitive": False},
                {
                    "text": "Hello, World!! Hello, Earth!!",
                    "pattern": "Hello, (\\S+)",
                    "group": 1,
                    "find_all": False,
                },
                {
                    "text": "Hello, World!! Hello, Earth!!",
                    "pattern": "Hello, (\\S+)",
                    "group": 1,
                    "find_all": True,
                },
            ],
            test_output=[
                # Test case 1
                ("positive", "World!"),
                ("matched_results", ["World!"]),
                ("matched_count", 1),
                # Test case 2
                ("positive", "Hello, World!"),
                ("matched_results", ["Hello, World!"]),
                ("matched_count", 1),
                # Test case 3
                ("negative", "Hello, World!"),
                ("matched_results", []),
                ("matched_count", 0),
                # Test case 4
                ("positive", "Hello,"),
                ("matched_results", ["Hello,"]),
                ("matched_count", 1),
                # Test case 5
                ("positive", "World!!"),
                ("matched_results", ["World!!"]),
                ("matched_count", 1),
                # Test case 6
                ("positive", "World!!"),
                ("positive", "Earth!!"),
                ("matched_results", ["World!!", "Earth!!"]),
                ("matched_count", 2),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        # Security fix: Add limits to prevent ReDoS and memory exhaustion
        MAX_TEXT_LENGTH = 1_000_000  # 1MB character limit
        MAX_MATCHES = 1000  # Maximum number of matches to prevent memory exhaustion
        MAX_MATCH_LENGTH = 10_000  # Maximum length per match

        flags = 0
        if not input_data.case_sensitive:
            flags = flags | re.IGNORECASE
        if input_data.dot_all:
            flags = flags | re.DOTALL

        if isinstance(input_data.text, str):
            txt = input_data.text
        else:
            txt = json.dumps(input_data.text)

        # Limit text size to prevent DoS
        if len(txt) > MAX_TEXT_LENGTH:
            txt = txt[:MAX_TEXT_LENGTH]

        # Validate regex pattern to prevent dangerous patterns
        dangerous_patterns = [
            r".*\+.*\+",  # Nested quantifiers
            r".*\*.*\*",  # Nested quantifiers
            r"(?=.*\+)",  # Lookahead with quantifier
            r"(?=.*\*)",  # Lookahead with quantifier
            r"\(.+\)\+",  # Group with nested quantifier
            r"\(.+\)\*",  # Group with nested quantifier
            r"\([^)]+\+\)\+",  # Nested quantifiers like (a+)+
            r"\([^)]+\*\)\*",  # Nested quantifiers like (a*)*
        ]

        # Check if pattern is potentially dangerous
        is_dangerous = any(
            re.search(dangerous, input_data.pattern) for dangerous in dangerous_patterns
        )

        # Use regex module with timeout for dangerous patterns
        # For safe patterns, use standard re module for compatibility
        try:
            matches = []
            match_count = 0

            if is_dangerous:
                # Use regex module with timeout (5 seconds) for dangerous patterns
                # The regex module supports timeout parameter in finditer
                try:
                    for match in regex.finditer(
                        input_data.pattern, txt, flags=flags, timeout=5.0
                    ):
                        if match_count >= MAX_MATCHES:
                            break
                        if input_data.group <= len(match.groups()):
                            match_text = match.group(input_data.group)
                            # Limit match length to prevent memory exhaustion
                            if len(match_text) > MAX_MATCH_LENGTH:
                                match_text = match_text[:MAX_MATCH_LENGTH]
                            matches.append(match_text)
                            match_count += 1
                except regex.error as e:
                    # Timeout occurred or regex error
                    if "timeout" in str(e).lower():
                        # Timeout - return empty results
                        pass
                    else:
                        # Other regex error
                        raise
            else:
                # Use standard re module for non-dangerous patterns
                for match in re.finditer(input_data.pattern, txt, flags):
                    if match_count >= MAX_MATCHES:
                        break
                    if input_data.group <= len(match.groups()):
                        match_text = match.group(input_data.group)
                        # Limit match length to prevent memory exhaustion
                        if len(match_text) > MAX_MATCH_LENGTH:
                            match_text = match_text[:MAX_MATCH_LENGTH]
                        matches.append(match_text)
                        match_count += 1

            if not input_data.find_all:
                matches = matches[:1]

            for match in matches:
                yield "positive", match
            if not matches:
                yield "negative", input_data.text

            yield "matched_results", matches
            yield "matched_count", len(matches)
        except Exception:
            # Return empty results on any regex error
            yield "negative", input_data.text
            yield "matched_results", []
            yield "matched_count", 0


class FillTextTemplateBlock(Block):
    class Input(BlockSchema):
        values: dict[str, Any] = SchemaField(
            description="Values (dict) to be used in format. These values can be used by putting them in double curly braces in the format template. e.g. {{value_name}}.",
        )
        format: str = SchemaField(
            description="Template to format the text using `values`. Use Jinja2 syntax."
        )
        escape_html: bool = SchemaField(
            default=False,
            advanced=True,
            description="Whether to escape special characters in the inserted values to be HTML-safe. Enable for HTML output, disable for plain text.",
        )

    class Output(BlockSchema):
        output: str = SchemaField(description="Formatted text")

    def __init__(self):
        super().__init__(
            id="db7d8f02-2f44-4c55-ab7a-eae0941f0c30",
            description="This block formats the given texts using the format template.",
            categories={BlockCategory.TEXT},
            input_schema=FillTextTemplateBlock.Input,
            output_schema=FillTextTemplateBlock.Output,
            test_input=[
                {
                    "values": {"name": "Alice", "hello": "Hello", "world": "World!"},
                    "format": "{{hello}}, {{ world }} {{name}}",
                },
                {
                    "values": {"list": ["Hello", " World!"]},
                    "format": "{% for item in list %}{{ item }}{% endfor %}",
                },
                {
                    "values": {},
                    "format": "{% set name = 'Alice' %}Hello, World! {{ name }}",
                },
            ],
            test_output=[
                ("output", "Hello, World! Alice"),
                ("output", "Hello World!"),
                ("output", "Hello, World! Alice"),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        formatter = text.TextFormatter(autoescape=input_data.escape_html)
        yield "output", formatter.format_string(input_data.format, input_data.values)


class CombineTextsBlock(Block):
    class Input(BlockSchema):
        input: list[str] = SchemaField(description="text input to combine")
        delimiter: str = SchemaField(
            description="Delimiter to combine texts", default=""
        )

    class Output(BlockSchema):
        output: str = SchemaField(description="Combined text")

    def __init__(self):
        super().__init__(
            id="e30a4d42-7b7d-4e6a-b36e-1f9b8e3b7d85",
            description="This block combines multiple input texts into a single output text.",
            categories={BlockCategory.TEXT},
            input_schema=CombineTextsBlock.Input,
            output_schema=CombineTextsBlock.Output,
            test_input=[
                {"input": ["Hello world I like ", "cake and to go for walks"]},
                {"input": ["This is a test", "Hi!"], "delimiter": "! "},
            ],
            test_output=[
                ("output", "Hello world I like cake and to go for walks"),
                ("output", "This is a test! Hi!"),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        combined_text = input_data.delimiter.join(input_data.input)
        yield "output", combined_text


class TextSplitBlock(Block):
    class Input(BlockSchema):
        text: str = SchemaField(description="The text to split.")
        delimiter: str = SchemaField(description="The delimiter to split the text by.")
        strip: bool = SchemaField(
            description="Whether to strip the text.", default=True
        )

    class Output(BlockSchema):
        texts: list[str] = SchemaField(
            description="The text split into a list of strings."
        )

    def __init__(self):
        super().__init__(
            id="d5ea33c8-a575-477a-b42f-2fe3be5055ec",
            description="This block is used to split a text into a list of strings.",
            categories={BlockCategory.TEXT},
            input_schema=TextSplitBlock.Input,
            output_schema=TextSplitBlock.Output,
            test_input=[
                {"text": "Hello, World!", "delimiter": ","},
                {"text": "Hello, World!", "delimiter": ",", "strip": False},
            ],
            test_output=[
                ("texts", ["Hello", "World!"]),
                ("texts", ["Hello", " World!"]),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        if len(input_data.text) == 0:
            yield "texts", []
        else:
            texts = input_data.text.split(input_data.delimiter)
            if input_data.strip:
                texts = [text.strip() for text in texts]
            yield "texts", texts


class TextReplaceBlock(Block):
    class Input(BlockSchema):
        text: str = SchemaField(description="The text to replace.")
        old: str = SchemaField(description="The old text to replace.")
        new: str = SchemaField(description="The new text to replace with.")

    class Output(BlockSchema):
        output: str = SchemaField(description="The text with the replaced text.")

    def __init__(self):
        super().__init__(
            id="7e7c87ab-3469-4bcc-9abe-67705091b713",
            description="This block is used to replace a text with a new text.",
            categories={BlockCategory.TEXT},
            input_schema=TextReplaceBlock.Input,
            output_schema=TextReplaceBlock.Output,
            test_input=[
                {"text": "Hello, World!", "old": "Hello", "new": "Hi"},
            ],
            test_output=[
                ("output", "Hi, World!"),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "output", input_data.text.replace(input_data.old, input_data.new)


class FileReadBlock(Block):
    class Input(BlockSchema):
        file_input: MediaFileType = SchemaField(
            description="The file to read from (URL, data URI, or local path)"
        )
        delimiter: str = SchemaField(
            description="Delimiter to split the content into rows/chunks (e.g., '\\n' for lines)",
            default="",
            advanced=True,
        )
        size_limit: int = SchemaField(
            description="Maximum size in bytes per chunk to yield (0 for no limit)",
            default=0,
            advanced=True,
        )
        row_limit: int = SchemaField(
            description="Maximum number of rows to process (0 for no limit, requires delimiter)",
            default=0,
            advanced=True,
        )
        skip_size: int = SchemaField(
            description="Number of characters to skip from the beginning of the file",
            default=0,
            advanced=True,
        )
        skip_rows: int = SchemaField(
            description="Number of rows to skip from the beginning (requires delimiter)",
            default=0,
            advanced=True,
        )

    class Output(BlockSchema):
        content: str = SchemaField(
            description="File content, yielded as individual chunks when delimiter or size limits are applied"
        )

    def __init__(self):
        super().__init__(
            id="3735a31f-7e18-4aca-9e90-08a7120674bc",
            input_schema=FileReadBlock.Input,
            output_schema=FileReadBlock.Output,
            description="Reads a file and returns its content as a string, with optional chunking by delimiter and size limits",
            categories={BlockCategory.TEXT, BlockCategory.DATA},
            test_input={
                "file_input": "data:text/plain;base64,SGVsbG8gV29ybGQ=",
            },
            test_output=[
                ("content", "Hello World"),
            ],
        )

    async def run(
        self, input_data: Input, *, graph_exec_id: str, user_id: str, **_kwargs
    ) -> BlockOutput:
        # Store the media file properly (handles URLs, data URIs, etc.)
        stored_file_path = await store_media_file(
            user_id=user_id,
            graph_exec_id=graph_exec_id,
            file=input_data.file_input,
            return_content=False,
        )

        # Get full file path
        file_path = get_exec_file_path(graph_exec_id, stored_file_path)

        if not Path(file_path).exists():
            raise ValueError(f"File does not exist: {file_path}")

        # Read file content
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except UnicodeDecodeError:
            # Try with different encodings
            try:
                with open(file_path, "r", encoding="latin-1") as file:
                    content = file.read()
            except Exception as e:
                raise ValueError(f"Unable to read file: {e}")

        # Apply skip_size (character-level skip)
        if input_data.skip_size > 0:
            content = content[input_data.skip_size :]

        # Split content into items (by delimiter or treat as single item)
        items = (
            content.split(input_data.delimiter) if input_data.delimiter else [content]
        )

        # Apply skip_rows (item-level skip)
        if input_data.skip_rows > 0:
            items = items[input_data.skip_rows :]

        # Apply row_limit (item-level limit)
        if input_data.row_limit > 0:
            items = items[: input_data.row_limit]

        # Process each item and create chunks
        def create_chunks(text, size_limit):
            """Create chunks from text based on size_limit"""
            if size_limit <= 0:
                return [text] if text else []

            chunks = []
            for i in range(0, len(text), size_limit):
                chunk = text[i : i + size_limit]
                if chunk:  # Only add non-empty chunks
                    chunks.append(chunk)
            return chunks

        # Process items and yield as content chunks
        if items:
            full_content = (
                input_data.delimiter.join(items)
                if input_data.delimiter
                else "".join(items)
            )

            # Create chunks of the full content based on size_limit
            content_chunks = create_chunks(full_content, input_data.size_limit)
            for chunk in content_chunks:
                yield "content", chunk
        else:
            yield "content", ""
