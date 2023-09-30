import enum
from typing import Any

from pydantic import BaseModel


class ContentType(str, enum.Enum):
    """
    Enumerates the different types of content for the Knowledge class.

    Attributes:
        TEXT: Represents textual content.
        CODE: Represents code-based content.

    Example:
        >>> content_type = ContentType.TEXT
        >>> print(content_type)
        ContentType.TEXT
    """

    # TBD what these actually are.
    TEXT = "text"
    CODE = "code"


class Knowledge(BaseModel):
    """
    Represents a unit of knowledge with associated content and metadata.

    Attributes:
        content (str): The actual content of the knowledge.
        content_type (ContentType): The type of the content, whether it's TEXT or CODE.
        content_metadata (dict[str, Any]): Metadata related to the content.

    Example:
        >>> knowledge = Knowledge(content="Hello, World!", content_type=ContentType.TEXT, content_metadata={"author": "John Doe"})
        >>> print(knowledge.content)
        Hello, World!
    """

    content: str
    content_type: ContentType
    content_metadata: dict[str, Any]


class ToolResult(BaseModel):
    """
    Represents the standard response structure for an ability.

    Attributes:
        ability_name (str): Name of the executed ability.
        ability_args (dict[str, str]): Arguments passed to the ability.
        success (bool): Indicates if the execution of the ability was successful or not.
        message (str): A message related to the result of the ability's execution.
        new_knowledge (Knowledge, optional): Represents new knowledge that may have been produced.

    Example:
        >>> result = ToolResult(ability_name="print", ability_args={"text": "Hello"}, success=True, message="Printed successfully")
        >>> print(result.summary())
        print(text=Hello): Printed successfully
    """

    ability_name: str
    ability_args: dict[str, str]
    success: bool
    message: str
    new_knowledge: Knowledge = None

    def summary(self) -> str:
        """
        Generates a summary of the ability's execution result.

        Returns:
            str: A formatted string representing the ability name, its arguments, and the result message.

        Example:
            >>> result = ToolResult(ability_name="print", ability_args={"text": "Hello"}, success=True, message="Printed successfully")
            >>> print(result.summary())
            print(text=Hello): Printed successfully
        """
        kwargs = ", ".join(f"{k}={v}" for k, v in self.ability_args.items())
        return f"{self.ability_name}({kwargs}): {self.message}"
