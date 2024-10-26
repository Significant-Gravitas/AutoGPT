from __future__ import annotations
from enum import Enum
from typing import Any, Optional, Union, Dict, Type, Callable, ClassVar

from pydantic import Field, BaseModel, ValidationError, field_validator
from AFAAS.configs.schema import AFAASModel
from AFAAS.lib.sdk.errors import AgentException, ToolExecutionError, UnknownToolError

from inspect import signature, Parameter

class OutputTypeManager:

    _instance : OutputTypeManager = None
    _registry : dict[str, Type[OutputType]] = {}
    _data_class_registry : dict[Type[OutputType], str] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OutputTypeManager, cls).__new__(cls)
            cls._instance._registry = {}
            cls._instance._data_class_registry = {}
        return cls._instance

    def register_output_type(self, output_type: Type[OutputType]):
        if output_type.name in self._registry:
            raise ValueError(f"Output type '{output_type.name}' is already registered.")

        # Validate the `format` callable
        OutputType.validate_format_callable(output_type.format, output_type.model_fields['data'])

        self._registry[output_type.name] = output_type
        self._data_class_registry[output_type] = output_type.name

    def get_data_class(self, type_name: str) -> Type[BaseModel]:
        return self._registry.get(type_name)

    def get_type_name(self, data_class: Type[BaseModel]) -> str:
        return self._data_class_registry.get(data_class)

    def get_output_types(self)-> list[str]:
        return self._registry.keys()

def format_file_output(loopindex: int, document: FileOutputData) -> str:
    return f"{loopindex}. {document.path} ({document.doc_id}):\n{document.content}"

def format_task_output(loopindex: int, task: TaskOutputData) -> str:
    return f"{loopindex}. {task.task_goal} (id : {task.task_id})"

def format_error_output(loopindex: int, error: OutputDataError) -> str:
    return error.__str__()


class OutputType(BaseModel):
    name : ClassVar[str] = Field(description = "Name of the output type")
    description : ClassVar[str] = Field(description = "Description of the output used in afaas_task_postprocess_default_summary")
    format : ClassVar[Callable[[int, BaseModel], str]] = Field(description = "Format of the output" )
    data : BaseModel = Field(description = "Data of the output")

    @staticmethod
    def validate_format_callable(format_callable, expected_data_type: Type[BaseModel]):
        if not callable(format_callable):
            raise ValueError("`format` must be callable")

        sig = signature(format_callable)
        params = list(sig.parameters.values())

        # Check for two parameters: int and specific data type
        if len(params) != 2 or params[0].annotation != 'int' or params[1].annotation is not expected_data_type.annotation.__name__:
            raise ValueError(f"`format` callable must accept parameters (int, {expected_data_type.annotation.__name__})")

        # Check return type is str
        if sig.return_annotation != 'str':
            raise ValueError("`format` callable must return a str")


class FileOutputData(BaseModel):
    doc_id: str = Field(description="Unique identifier of the generated file.")
    name: Optional[str] = Field(description="Name of the generated file.")
    description: Optional[str] = Field(description="Description of the generated file.")
    path: str = Field(description="Location of the generated file.")
    content: str = Field(description="Content of the generated file.")

class TaskOutputData(BaseModel):
    task_id: str = Field(description="Unique identifier of the generated task.")
    task_goal: str = Field(description="Details of the task processed.")

class FileOutput(OutputType) : 
    name : ClassVar[str] = "file"
    description : ClassVar[str] = "Following documents have been created/modified :"
    format : ClassVar[Callable[[int, FileOutputData], str]]  = format_file_output
    data : FileOutputData = Field(description = "Data of the output")

class TaskOutput(OutputType) :
    name : ClassVar[str] = "task"
    description : ClassVar[str] = "The following subtasks has been created :"
    format : ClassVar[Callable[[int, FileOutputData], str]]  = format_task_output
    data : TaskOutputData = Field(description = "Data of the output")


class OutputDataError(BaseModel):
    error_code: Optional[str] = Field(description="Error code, if applicable.")
    error_message: str = Field(description="Detailed error message.")
    exception_class : Type[Exception] = Field(default = ToolExecutionError, description="Class of the exception raised.")

class ErrorOutput(OutputType) :
    name : ClassVar[str] = "error"
    description : ClassVar[str] = "Following errors have been encountered :"
    format : ClassVar[Callable[[int, FileOutputData], str]]  = format_error_output
    data : OutputDataError = Field(description = "Data of the output")


# Create a global instance of the manager
AFAAS_OUTPUT_TYPE_MANAGER = OutputTypeManager()
AFAAS_OUTPUT_TYPE_MANAGER.register_output_type(FileOutput)
AFAAS_OUTPUT_TYPE_MANAGER.register_output_type(TaskOutput)
AFAAS_OUTPUT_TYPE_MANAGER.register_output_type(ErrorOutput)

# class Output(BaseModel):
#     type: str = Field(description = "Type of output (e.g., file, userID, error, correction)")
#     data: OutputType = Field(description = "Generic structure ; data will depend of the type of output")

#     @field_validator('data')
#     def validate_data_type(cls, value, values):
#         type_name = values.get('name')
#         expected_class = AFAAS_OUTPUT_TYPE_MANAGER.get_data_class(type_name)
#         if not expected_class:
#             raise ValueError(f"No data class registered for output type '{type_name}'.")
#         if not isinstance(value, expected_class):
#             raise ValueError(f"Data for output type '{type_name}' must be an instance of {expected_class.__name__}.")
#         return value

#     def __init__(__pydantic_self__, **data):
#         type_name = data.get('type')
#         raw_data = data.get('data')
#         data_class = AFAAS_OUTPUT_TYPE_MANAGER.get_data_class(type_name)
#         if data_class and isinstance(raw_data, dict):
#             try:
#                 data['data'] = data_class(**raw_data)
#             except ValidationError as e:
#                 raise ValueError(f"Invalid data for {data_class.__name__}: {e}")
#         super().__init__(**data)

class ToolOutput(AFAASModel): 
    reasoning: Optional[str] = Field(default=None, description="Rationale behind the task, including context and dependencies.")
    action: Optional[str] = Field(default=None, description="Detailed account of steps taken or attempted by the AI to accomplish the task.")
    output: Optional[Dict[str, list[OutputType]]] = Field(default_factory=dict, description="Output of the task, including files, tasks, errors, and corrections.")


    def add_output(self, output: OutputType):
        if output.name not in AFAAS_OUTPUT_TYPE_MANAGER.get_output_types():
            raise ValueError(f"Invalid output type '{output.name}'.")
        if output.name not in self.output:
            self.output[output.name] = []
        self.output[output.name].append(output)

class ToolOutputErrorMissing(Exception):
    "Tool must return an object of type ToolOutput"
