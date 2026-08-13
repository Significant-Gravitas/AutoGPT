import base64
import html
import json
import logging
import re
import urllib.parse
from typing import Iterator, Literal, Optional

from pydantic import BaseModel, Field

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider, DirectiveProvider
from forge.command import Command, command
from forge.models.json_schema import JSONSchema
from forge.utils.exceptions import CommandExecutionError

logger = logging.getLogger(__name__)


class TextUtilsConfiguration(BaseModel):
    max_text_length: int = Field(
        default=100000, description="Maximum text length to process"
    )
    max_matches: int = Field(
        default=1000, description="Maximum number of regex matches to return"
    )


class TextUtilsComponent(
    DirectiveProvider, CommandProvider, ConfigurableComponent[TextUtilsConfiguration]
):
    """Provides commands for text manipulation, regex operations, and encoding."""

    config_class = TextUtilsConfiguration

    def __init__(self, config: Optional[TextUtilsConfiguration] = None):
        ConfigurableComponent.__init__(self, config)

    def get_resources(self) -> Iterator[str]:
        yield "Ability to manipulate text with regex and encoding operations."

    def get_commands(self) -> Iterator[Command]:
        yield self.regex_search
        yield self.regex_replace
        yield self.encode_text
        yield self.decode_text
        yield self.format_template

    def _parse_flags(self, flags: str | None) -> int:
        """Parse regex flag string into re flags.

        Args:
            flags: String of flags (i, m, s, x)

        Returns:
            int: Combined re flags
        """
        if not flags:
            return 0

        flag_map = {
            "i": re.IGNORECASE,
            "m": re.MULTILINE,
            "s": re.DOTALL,
            "x": re.VERBOSE,
        }

        result = 0
        for char in flags.lower():
            if char in flag_map:
                result |= flag_map[char]

        return result

    @command(
        ["regex_search", "find_pattern"],
        "Search text for matches using a regular expression pattern.",
        {
            "text": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The text to search in",
                required=True,
            ),
            "pattern": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The regex pattern to search for",
                required=True,
            ),
            "flags": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Regex flags: i=ignorecase, m=multiline, s=dotall",
                required=False,
            ),
            "return_groups": JSONSchema(
                type=JSONSchema.Type.BOOLEAN,
                description="Return capture groups instead of matches (default: False)",
                required=False,
            ),
        },
    )
    def regex_search(
        self,
        text: str,
        pattern: str,
        flags: str | None = None,
        return_groups: bool = False,
    ) -> str:
        """Search text using regex pattern.

        Args:
            text: The text to search
            pattern: The regex pattern
            flags: Optional flags string
            return_groups: Whether to return capture groups

        Returns:
            str: JSON array of matches
        """
        if len(text) > self.config.max_text_length:
            raise CommandExecutionError(
                f"Text exceeds maximum length of {self.config.max_text_length}"
            )

        try:
            regex = re.compile(pattern, self._parse_flags(flags))
        except re.error as e:
            raise CommandExecutionError(f"Invalid regex pattern: {e}")

        matches = []
        for match in regex.finditer(text):
            if len(matches) >= self.config.max_matches:
                break

            if return_groups and match.groups():
                matches.append(
                    {
                        "match": match.group(0),
                        "groups": match.groups(),
                        "start": match.start(),
                        "end": match.end(),
                    }
                )
            else:
                matches.append(
                    {
                        "match": match.group(0),
                        "start": match.start(),
                        "end": match.end(),
                    }
                )

        result = {
            "count": len(matches),
            "matches": matches,
        }

        if len(matches) >= self.config.max_matches:
            result["truncated"] = True

        return json.dumps(result, indent=2)

    @command(
        ["regex_replace", "replace_pattern"],
        "Replace text matching a regex pattern with a replacement string.",
        {
            "text": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The text to search and replace in",
                required=True,
            ),
            "pattern": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The regex pattern to match",
                required=True,
            ),
            "replacement": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The replacement string (can use \\1, \\2 for groups)",
                required=True,
            ),
            "flags": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Regex flags: i=ignorecase, m=multiline, s=dotall",
                required=False,
            ),
            "count": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="Maximum replacements (0 = all, default: 0)",
                required=False,
            ),
        },
    )
    def regex_replace(
        self,
        text: str,
        pattern: str,
        replacement: str,
        flags: str | None = None,
        count: int = 0,
    ) -> str:
        """Replace text matching regex pattern.

        Args:
            text: The text to modify
            pattern: The regex pattern
            replacement: The replacement string
            flags: Optional flags string
            count: Max replacements (0 = unlimited)

        Returns:
            str: The modified text with replacement info
        """
        if len(text) > self.config.max_text_length:
            raise CommandExecutionError(
                f"Text exceeds maximum length of {self.config.max_text_length}"
            )

        try:
            regex = re.compile(pattern, self._parse_flags(flags))
        except re.error as e:
            raise CommandExecutionError(f"Invalid regex pattern: {e}")

        # Count matches before replacement
        match_count = len(regex.findall(text))

        # Perform replacement
        result = regex.sub(replacement, text, count=count if count > 0 else 0)

        actual_replacements = min(match_count, count) if count > 0 else match_count

        return json.dumps(
            {
                "result": result,
                "replacements_made": actual_replacements,
                "pattern": pattern,
            },
            indent=2,
        )

    @command(
        ["encode_text"],
        "Encode text using various encoding schemes.",
        {
            "text": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The text to encode",
                required=True,
            ),
            "encoding": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Encoding type: base64, url, html, hex",
                required=True,
            ),
        },
    )
    def encode_text(
        self, text: str, encoding: Literal["base64", "url", "html", "hex"]
    ) -> str:
        """Encode text using specified encoding.

        Args:
            text: The text to encode
            encoding: The encoding type

        Returns:
            str: The encoded text
        """
        if encoding == "base64":
            result = base64.b64encode(text.encode("utf-8")).decode("ascii")
        elif encoding == "url":
            result = urllib.parse.quote(text, safe="")
        elif encoding == "html":
            result = html.escape(text)
        elif encoding == "hex":
            result = text.encode("utf-8").hex()
        else:
            raise CommandExecutionError(
                f"Unknown encoding: {encoding}. Supported: base64, url, html, hex"
            )

        return json.dumps(
            {"original": text, "encoding": encoding, "result": result}, indent=2
        )

    @command(
        ["decode_text"],
        "Decode text from various encoding schemes.",
        {
            "text": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The text to decode",
                required=True,
            ),
            "encoding": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Encoding type: base64, url, html, hex",
                required=True,
            ),
        },
    )
    def decode_text(
        self, text: str, encoding: Literal["base64", "url", "html", "hex"]
    ) -> str:
        """Decode text from specified encoding.

        Args:
            text: The text to decode
            encoding: The encoding type

        Returns:
            str: The decoded text
        """
        try:
            if encoding == "base64":
                result = base64.b64decode(text).decode("utf-8")
            elif encoding == "url":
                result = urllib.parse.unquote(text)
            elif encoding == "html":
                result = html.unescape(text)
            elif encoding == "hex":
                result = bytes.fromhex(text).decode("utf-8")
            else:
                raise CommandExecutionError(
                    f"Unknown encoding: {encoding}. Supported: base64, url, html, hex"
                )

            return json.dumps(
                {"original": text, "encoding": encoding, "result": result}, indent=2
            )

        except Exception as e:
            raise CommandExecutionError(f"Decoding failed: {e}")

    @command(
        ["format_template", "template_substitute"],
        "Substitute variables in a template string using {variable} syntax.",
        {
            "template": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Template with {variable} placeholders",
                required=True,
            ),
            "variables": JSONSchema(
                type=JSONSchema.Type.OBJECT,
                description="Dictionary of variable names to values",
                required=True,
            ),
        },
    )
    def format_template(self, template: str, variables: dict[str, str]) -> str:
        """Substitute variables in a template.

        Args:
            template: The template string with {placeholders}
            variables: Dictionary of variable values

        Returns:
            str: The formatted string
        """
        try:
            # Use safe substitution that only replaces found keys
            result = template
            for key, value in variables.items():
                result = result.replace("{" + key + "}", str(value))

            # Check for unfilled placeholders
            unfilled = re.findall(r"\{(\w+)\}", result)

            return json.dumps(
                {
                    "result": result,
                    "unfilled_placeholders": unfilled if unfilled else None,
                },
                indent=2,
            )

        except Exception as e:
            raise CommandExecutionError(f"Template formatting failed: {e}")
