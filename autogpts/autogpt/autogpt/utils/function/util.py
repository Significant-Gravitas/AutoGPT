from typing import List, Tuple, __all__ as all_types
from autogpt.utils.function.model import FunctionDef, ObjectType, ValidationResponse

OPEN_BRACES = {"{": "Dict", "[": "List", "(": "Tuple"}
CLOSE_BRACES = {"}": "Dict", "]": "List", ")": "Tuple"}

RENAMED_TYPES = {
    "dict": "Dict",
    "list": "List",
    "tuple": "Tuple",
    "set": "Set",
    "frozenset": "FrozenSet",
    "type": "Type",
}
PYTHON_TYPES = set(all_types)


def unwrap_object_type(type: str) -> Tuple[str, List[str]]:
    """
    Get the type and children of a composite type.
    Args:
        type (str): The type to parse.
    Returns:
        str: The type.
        [str]: The children types.
    """
    type = type.replace(" ", "")
    if not type:
        return "", []

    def split_outer_level(type: str, separator: str) -> List[str]:
        brace_count = 0
        last_index = 0
        splits = []

        for i, c in enumerate(type):
            if c in OPEN_BRACES:
                brace_count += 1
            elif c in CLOSE_BRACES:
                brace_count -= 1
            elif c == separator and brace_count == 0:
                splits.append(type[last_index:i])
                last_index = i + 1

        splits.append(type[last_index:])
        return splits

    # Unwrap primitive union types
    union_split = split_outer_level(type, "|")
    if len(union_split) > 1:
        if len(union_split) == 2 and "None" in union_split:
            return "Optional", [v for v in union_split if v != "None"]
        return "Union", union_split

    # Unwrap primitive dict/list/tuple types
    if type[0] in OPEN_BRACES and type[-1] in CLOSE_BRACES:
        type_name = OPEN_BRACES[type[0]]
        type_children = split_outer_level(type[1:-1], ",")
        return type_name, type_children

    brace_pos = type.find("[")
    if brace_pos != -1 and type[-1] == "]":
        # Unwrap normal composite types
        type_name = type[:brace_pos]
        type_children = split_outer_level(type[brace_pos + 1 : -1], ",")
    else:
        # Non-composite types, no need to unwrap
        type_name = type
        type_children = []

    return RENAMED_TYPES.get(type_name, type_name), type_children


def is_type_equal(type1: str | None, type2: str | None) -> bool:
    """
    Check if two types are equal.
    This function handle composite types like list, dict, and tuple.
    group similar types like list[str], List[str], and [str] as equal.
    """
    if type1 is None and type2 is None:
        return True
    if type1 is None or type2 is None:
        return False

    evaluated_type1, children1 = unwrap_object_type(type1)
    evaluated_type2, children2 = unwrap_object_type(type2)

    # Compare the class name of the types (ignoring the module)
    # TODO(majdyz): compare the module name as well.
    t_len = min(len(evaluated_type1), len(evaluated_type2))
    if evaluated_type1.split(".")[-t_len:] != evaluated_type2.split(".")[-t_len:]:
        return False

    if len(children1) != len(children2):
        return False

    if len(children1) == len(children2) == 0:
        return True

    for c1, c2 in zip(children1, children2):
        if not is_type_equal(c1, c2):
            return False

    return True


def validate_matching_function(this: FunctionDef, that: FunctionDef):
    expected_args = that.arg_types
    expected_rets = that.return_type
    func_name = that.name
    errors = []

    if any(
        [
            x[0] != y[0] or not is_type_equal(x[1], y[1]) and x[1] != "object"
            # TODO: remove sorted and provide a stable order for one-to-many arg-types.
            for x, y in zip(sorted(expected_args), sorted(this.arg_types))
        ]
    ):
        errors.append(
            f"Function {func_name} has different arguments than expected, "
            f"expected {expected_args} but got {this.arg_types}"
        )
    if (
        not is_type_equal(expected_rets, this.return_type)
        and expected_rets != "object"
    ):
        errors.append(
            f"Function {func_name} has different return type than expected, expected "
            f"{expected_rets} but got {this.return_type}"
        )

    if errors:
        raise Exception("Signature validation errors:\n  " + "\n  ".join(errors))


def normalize_type(type: str, renamed_types: dict[str, str] = {}) -> str:
    """
    Normalize the type to a standard format.
    e.g. list[str] -> List[str], dict[str, int | float] -> Dict[str, Union[int, float]]

    Args:
        type (str): The type to normalize.
    Returns:
        str: The normalized type.
    """
    parent_type, children = unwrap_object_type(type)

    if parent_type in renamed_types:
        parent_type = renamed_types[parent_type]

    if len(children) == 0:
        return parent_type

    content_type = ", ".join([normalize_type(c, renamed_types) for c in children])
    return f"{parent_type}[{content_type}]"


def generate_object_code(obj: ObjectType) -> str:
    if not obj.name:
        return ""  # Avoid generating an empty object

    # Auto-generate a template for the object, this will not capture any class functions
    fields = f"\n{' ' * 4}".join(
        [
            f"{field.name}: {field.type} "
            f"{('= '+field.value) if field.value else ''} "
            f"{('# '+field.description) if field.description else ''}"
            for field in obj.Fields or []
        ]
    )

    parent_class = ""
    if obj.is_enum:
        parent_class = "Enum"
    elif obj.is_pydantic:
        parent_class = "BaseModel"

    doc_string = (
        f"""\"\"\"
    {obj.description}
    \"\"\""""
        if obj.description
        else ""
    )

    method_body = ("\n" + " " * 4).join(obj.code.split("\n")) + "\n" if obj.code else ""

    template = f"""
class {obj.name}({parent_class}):
    {doc_string if doc_string else ""}
    {fields if fields else ""}
    {method_body if method_body else ""}
    {"pass" if not fields and not method_body else ""}
"""
    return "\n".join(line for line in template.split("\n")).strip()


def genererate_line_error(error: str, code: str, line_number: int) -> str:
    lines = code.split("\n")
    if line_number > len(lines):
        return error

    code_line = lines[line_number - 1]
    return f"{error} -> '{code_line.strip()}'"


def generate_compiled_code(
        resp: ValidationResponse,
        add_code_stubs: bool = True
) -> str:
    """
    Regenerate imports & raw code using the available objects and functions.
    """
    resp.imports = sorted(set(resp.imports))

    def __append_comment(code_block: str, comment: str) -> str:
        """
        Append `# noqa` to the first line of the code block.
        This is to suppress flake8 warnings for redefined names.
        """
        lines = code_block.split("\n")
        lines[0] = lines[0] + " # " + comment
        return "\n".join(lines)

    def __generate_stub(name, is_enum):
        if not name:
            return ""
        elif is_enum:
            return f"class {name}(Enum):\n    pass"
        else:
            return f"class {name}(BaseModel):\n    pass"

    stub_objects = resp.available_objects if add_code_stubs else {}
    stub_functions = resp.available_functions if add_code_stubs else {}

    object_stubs_code = "\n\n".join(
        [
            __append_comment(__generate_stub(obj.name, obj.is_enum), "type: ignore")
            for obj in stub_objects.values()
        ]
        + [
            __append_comment(__generate_stub(obj.name, obj.is_enum), "type: ignore")
            for obj in resp.objects
            if obj.name not in stub_objects
        ]
    )

    objects_code = "\n\n".join(
        [
            __append_comment(generate_object_code(obj), "noqa")
            for obj in stub_objects.values()
        ]
        + [
            __append_comment(generate_object_code(obj), "noqa")
            for obj in resp.objects
            if obj.name not in stub_objects
        ]
    )

    functions_code = "\n\n".join(
        [
            __append_comment(f.function_template.strip(), "type: ignore")
            for f in stub_functions.values()
            if f.name != resp.function_name and f.function_template
        ]
        + [
            __append_comment(f.function_template.strip(), "type: ignore")
            for f in resp.functions
            if f.name not in stub_functions and f.function_template
        ]
    )

    resp.rawCode = (
        object_stubs_code.strip()
        + "\n\n"
        + objects_code.strip()
        + "\n\n"
        + functions_code.strip()
        + "\n\n"
        + resp.functionCode.strip()
    )

    return resp.get_compiled_code()
