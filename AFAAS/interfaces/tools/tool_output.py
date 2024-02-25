from __future__ import annotations
from enum import Enum
from typing import Any, Optional, Union, Dict, Type
from pydantic import Field, BaseModel, ValidationError, field_validator
from AFAAS.configs.schema import AFAASModel

class OutputTypeManager:

    _instance : OutputTypeManager = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OutputTypeManager, cls).__new__(cls)
            cls._instance._registry = {}
            cls._instance._data_class_registry = {}
        return cls._instance


    def register_output_type(self, type_name: str, data_class: Type[BaseModel]):
        if type_name in self._registry:
            raise ValueError(f"Output type '{type_name}' is already registered.")
        self._registry[type_name] = data_class
        self._data_class_registry[data_class] = type_name

    def get_data_class(self, type_name: str) -> Type[BaseModel]:
        return self._registry.get(type_name)

    def get_type_name(self, data_class: Type[BaseModel]) -> str:
        return self._data_class_registry.get(data_class)

    def get_output_types(self)-> list[str]:
        return self._registry.keys()

    # @classmethod
    # def register_output_data_type(cls, output_type: DefaultOutputType, data_class: Type[BaseModel]):
    #     if output_type in OUTPUT_DATA_REGISTRY:
    #         raise ValueError(f"Output type '{output_type}' is already registered.")
    #     OUTPUT_DATA_REGISTRY[output_type] = data_class



class DefaultOutputType(Enum):
    FILE = "file"
    TASK = "task"
    ERROR = "error"
    # CORRECTION = "correction" # Consider adding if you plan to handle corrections as a type of output


# Create a global instance of the manager
AFAAS_OUTPUT_TYPE_MANAGER = OutputTypeManager()
# Registry mapping OutputTypes to data classes
# OUTPUT_DATA_REGISTRY: Dict[DefaultOutputType, Type[BaseModel]] = {}


class FileOutputData(BaseModel):
    document_id: str = Field(description="Unique identifier of the generated file.")
    file_name: str = Field(description="Name of the generated file.")
    file_description: str = Field(description="Description of the generated file.")
    file_path: str = Field(description="Location of the generated file.")

class TaskOutputData(BaseModel):
    task_id: str = Field(description="Unique identifier of the generated task.")
    task_goal: str = Field(description="Details of the task processed.")

class ErrorOutputData(BaseModel):
    error_code: Optional[str] = Field(description="Error code, if applicable.")
    error_message: str = Field(description="Detailed error message.")



class Output(BaseModel):
    type: str = Field(description = "Type of output (e.g., file, userID, error, correction)")
    data: BaseModel = Field(description = "Generic structure ; data will depend of the type of output")

    @field_validator('data', pre=True, always=True)
    def validate_data_type(cls, value, values):
        type_name = values.get('type')
        expected_class = AFAAS_OUTPUT_TYPE_MANAGER.get_data_class(type_name)
        if not expected_class:
            raise ValueError(f"No data class registered for output type '{type_name}'.")
        if not isinstance(value, expected_class):
            raise ValueError(f"Data for output type '{type_name}' must be an instance of {expected_class.__name__}.")
        return value

    def __init__(__pydantic_self__, **data):
        type_name = data.get('type')
        raw_data = data.get('data')
        data_class = AFAAS_OUTPUT_TYPE_MANAGER.get_data_class(type_name)
        if data_class and isinstance(raw_data, dict):
            try:
                data['data'] = data_class(**raw_data)
            except ValidationError as e:
                raise ValueError(f"Invalid data for {data_class.__name__}: {e}")
        super().__init__(**data)

class ToolOutput(AFAASModel): 
    reasoning: str = Field(description = "Rationale behind the task, including context and dependencies.")
    action: str = Field(description =  "Detailed account of steps taken or attempted by the AI to accomplish the task.")
    output: Dict[str, Output]  = Field(description = "Output of the task, including files, tasks, errors, and corrections.")

    def add_output(self, output_type: str, data: BaseModel):
        if output_type not in AFAAS_OUTPUT_TYPE_MANAGER.get_output_types():
            raise ValueError(f"Invalid output type '{output_type}'.")
        output = Output(type=output_type, data=data)
        self.output[output_type] = output

