from __future__ import annotations

from typing import Any, Literal, Optional, TypeVar

from pydantic import BaseModel
from pydantic.json_schema import (
    DEFAULT_REF_TEMPLATE,
    GenerateJsonSchema,
    JsonSchemaMode,
)

from forge.llm.providers.schema import AssistantChatMessage, AssistantFunctionCall

from .utils import ModelWithSummary


class ActionProposal(BaseModel):
    thoughts: str | ModelWithSummary
    use_tool: AssistantFunctionCall

    raw_message: AssistantChatMessage = None  # type: ignore
    """
    The message from which the action proposal was parsed. To be set by the parser.
    """

    @classmethod
    def model_json_schema(
        cls,
        by_alias: bool = True,
        ref_template: str = DEFAULT_REF_TEMPLATE,
        schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema,
        mode: JsonSchemaMode = "validation",
        **kwargs,
    ):
        """
        The schema for this ActionProposal model, excluding the 'raw_message' property.
        """
        schema = super().model_json_schema(
            by_alias=by_alias,
            ref_template=ref_template,
            schema_generator=schema_generator,
            mode=mode,
            **kwargs,
        )
        if "raw_message" in schema["properties"]:  # must check because schema is cached
            del schema["properties"]["raw_message"]
        return schema


AnyProposal = TypeVar("AnyProposal", bound=ActionProposal)


class ActionSuccessResult(BaseModel):
    outputs: Any
    status: Literal["success"] = "success"

    def __str__(self) -> str:
        outputs = str(self.outputs).replace("```", r"\```")
        multiline = "\n" in outputs
        return f"```\n{self.outputs}\n```" if multiline else str(self.outputs)


class ErrorInfo(BaseModel):
    args: tuple
    message: str
    exception_type: str
    repr: str

    @staticmethod
    def from_exception(exception: Exception) -> ErrorInfo:
        return ErrorInfo(
            args=exception.args,
            message=getattr(exception, "message", exception.args[0]),
            exception_type=exception.__class__.__name__,
            repr=repr(exception),
        )

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return self.repr


class ActionErrorResult(BaseModel):
    reason: str
    error: Optional[ErrorInfo] = None
    status: Literal["error"] = "error"

    @staticmethod
    def from_exception(exception: Exception) -> ActionErrorResult:
        return ActionErrorResult(
            reason=getattr(exception, "message", exception.args[0]),
            error=ErrorInfo.from_exception(exception),
        )

    def __str__(self) -> str:
        return f"Action failed: '{self.reason}'"


class ActionInterruptedByHuman(BaseModel):
    feedback: str
    status: Literal["interrupted_by_human"] = "interrupted_by_human"

    def __str__(self) -> str:
        return (
            'The user interrupted the action with the following feedback: "%s"'
            % self.feedback
        )


ActionResult = ActionSuccessResult | ActionErrorResult | ActionInterruptedByHuman
