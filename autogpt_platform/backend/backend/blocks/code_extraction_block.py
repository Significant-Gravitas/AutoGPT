import re

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class CodeExtractionBlock(Block):
    class Input(BlockSchema):
        text: str = SchemaField(
            description="Text containing code blocks to extract (e.g., AI response)",
            placeholder="Enter text containing code blocks",
        )

    class Output(BlockSchema):
        html: str = SchemaField(description="Extracted HTML code")
        css: str = SchemaField(description="Extracted CSS code")
        javascript: str = SchemaField(description="Extracted JavaScript code")
        python: str = SchemaField(description="Extracted Python code")
        sql: str = SchemaField(description="Extracted SQL code")
        java: str = SchemaField(description="Extracted Java code")
        cpp: str = SchemaField(description="Extracted C++ code")
        csharp: str = SchemaField(description="Extracted C# code")
        json: str = SchemaField(description="Extracted JSON code")
        bash: str = SchemaField(description="Extracted Bash code")
        php: str = SchemaField(description="Extracted PHP code")
        ruby: str = SchemaField(description="Extracted Ruby code")
        yaml: str = SchemaField(description="Extracted YAML code")
        markdown: str = SchemaField(description="Extracted Markdown code")
        typescript: str = SchemaField(description="Extracted TypeScript code")
        xml: str = SchemaField(description="Extracted XML code")
        remaining_text: str = SchemaField(
            description="Remaining text after code extraction"
        )

    def __init__(self):
        super().__init__(
            id="d3a7d896-3b78-4f44-8b4b-48fbf4f0bcd8",
            description="Extracts code blocks from text and identifies their programming languages",
            categories={BlockCategory.TEXT},
            input_schema=CodeExtractionBlock.Input,
            output_schema=CodeExtractionBlock.Output,
            test_input={
                "text": "Here's a Python example:\n```python\nprint('Hello World')\n```\nAnd some HTML:\n```html\n<h1>Title</h1>\n```"
            },
            test_output={
                "python": "print('Hello World')",
                "html": "<h1>Title</h1>",
                "remaining_text": "Here's a Python example:\nAnd some HTML:",
            },
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        # List of supported programming languages
        languages = [
            "html",
            "css",
            "javascript",
            "python",
            "sql",
            "java",
            "cpp",
            "csharp",
            "json",
            "bash",
            "php",
            "ruby",
            "yaml",
            "markdown",
            "typescript",
            "xml",
        ]

        # Extract code for each language
        for lang in languages:
            code = self.extract_code(input_data.text, lang)
            if code:  # Only yield if there's actual code content
                yield lang, code

        # Remove all code blocks from the text to get remaining text
        pattern = "|".join(languages)
        remaining_text = re.sub(
            rf"```({pattern})[\s\S]*?```", "", input_data.text
        ).strip()
        if remaining_text:  # Only yield if there's remaining text
            yield "remaining_text", remaining_text

    def extract_code(self, text: str, language: str) -> str:
        # Extract all code blocks enclosed in ```language``` blocks
        pattern = re.compile(rf"```{language}\n([\s\S]*?)```", re.IGNORECASE)
        matches = pattern.finditer(text)
        # Combine all code blocks for this language with newlines between them
        code_blocks = [match.group(1).strip() for match in matches]
        return "\n\n".join(code_blocks) if code_blocks else ""
