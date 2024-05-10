from typing import List, Optional
from pydantic import BaseModel, Field


class ObjectType(BaseModel):
    name: str = Field(description="The name of the object")
    code: Optional[str] = Field(description="The code of the object", default=None)
    description: Optional[str] = Field(
        description="The description of the object", default=None
    )
    Fields: List["ObjectField"] = Field(description="The fields of the object")
    is_pydantic: bool = Field(
        description="Whether the object is a pydantic model", default=True
    )
    is_implemented: bool = Field(
        description="Whether the object is implemented", default=True
    )
    is_enum: bool = Field(description="Whether the object is an enum", default=False)


class ObjectField(BaseModel):
    name: str = Field(description="The name of the field")
    description: Optional[str] = Field(
        description="The description of the field", default=None
    )
    type: str = Field(
        description="The type of the field. Can be a string like List[str] or an use "
        "any of they related types like list[User]",
    )
    value: Optional[str] = Field(description="The value of the field", default=None)
    related_types: Optional[List[ObjectType]] = Field(
        description="The related types of the field", default=[]
    )


class FunctionDef(BaseModel):
    name: str
    arg_types: list[tuple[str, str]]
    arg_descs: dict[str, str]
    return_type: str | None = None
    return_desc: str
    function_desc: str
    is_implemented: bool = False
    function_code: str = ""
    function_template: str | None = None
    is_async: bool = False

    def __generate_function_template(f) -> str:
        args_str = ", ".join([f"{name}: {type}" for name, type in f.arg_types])
        arg_desc = f"\n{' '*4}".join(
            [
                f'{name} ({type}): {f.arg_descs.get(name, "-")}'
                for name, type in f.arg_types
            ]
        )

        def_str = "async def" if "await " in f.function_code or f.is_async else "def"
        ret_type_str = f" -> {f.return_type}" if f.return_type else ""
        func_desc = f.function_desc.replace("\n", "\n    ")

        template = f"""
{def_str} {f.name}({args_str}){ret_type_str}:
    \"\"\"
    {func_desc}

    Args:
    {arg_desc}

    Returns:
    {f.return_type}{': ' + f.return_desc if f.return_desc else ''}
    \"\"\"
    pass
"""
        return "\n".join([line for line in template.split("\n")]).strip()

    def __init__(self, function_template: Optional[str] = None, **data):
        super().__init__(**data)
        self.function_template = (
            function_template or self.__generate_function_template()
        )


class ValidationResponse(BaseModel):
    function_name: str
    available_objects: dict[str, ObjectType]
    available_functions: dict[str, FunctionDef]
    
    template: str
    rawCode: str
    packages: List[str]
    imports: List[str]
    functionCode: str

    functions: List[FunctionDef]
    objects: List[ObjectType]

    def get_compiled_code(self) -> str:
        return "\n".join(self.imports) + "\n\n" + self.rawCode.strip()
