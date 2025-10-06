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
        json_code: str = SchemaField(description="Extracted JSON code")
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
            test_output=[
                ("html", "<h1>Title</h1>"),
                ("python", "print('Hello World')"),
                ("remaining_text", "Here's a Python example:\nAnd some HTML:"),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        # List of supported programming languages with mapped aliases
        language_aliases = {
            "html": ["html", "htm"],
            "css": ["css"],
            "javascript": ["javascript", "js"],
            "python": ["python", "py"],
            "sql": ["sql"],
            "java": ["java"],
            "cpp": ["cpp", "c++"],
            "csharp": ["csharp", "c#", "cs"],
            "json_code": ["json"],
            "bash": ["bash", "shell", "sh"],
            "php": ["php"],
            "ruby": ["ruby", "rb"],
            "yaml": ["yaml", "yml"],
            "markdown": ["markdown", "md"],
            "typescript": ["typescript", "ts"],
            "xml": ["xml"],
        }

        # Extract code for each language
        for canonical_name, aliases in language_aliases.items():
            code = ""
            # Try each alias for the language
            for alias in aliases:
                code_for_alias = self.extract_code(input_data.text, alias)
                if code_for_alias:
                    code = code + "\n\n" + code_for_alias if code else code_for_alias

            if code:  # Only yield if there's actual code content
                yield canonical_name, code

        # Remove all code blocks from the text to get remaining text
        pattern = (
            r"```(?:"
            + "|".join(
                re.escape(alias)
                for aliases in language_aliases.values()
                for alias in aliases
            )
            + r")[ \t]*\n[\s\S]*?```"
        )

        remaining_text = re.sub(pattern, "", input_data.text).strip()
        remaining_text = re.sub(r"\n\s*\n", "\n", remaining_text)

        if remaining_text:  # Only yield if there's remaining text
            yield "remaining_text", remaining_text

    def extract_code(self, text: str, language: str) -> str:
        # Escape special regex characters in the language string
        language = re.escape(language)
        # Extract all code blocks enclosed in ```language``` blocks
        pattern = re.compile(
            rf"```{language}[ \t]*\n(.*?)\n```", re.DOTALL | re.IGNORECASE
        )
        matches = pattern.finditer(text)
        # Combine all code blocks for this language with newlines between them
        code_blocks = [match.group(1).strip() for match in matches]
        return "\n\n".join(code_blocks) if code_blocks else ""
