import inspect
import logging
import re
from typing import Callable, Concatenate, Optional, TypeVar, cast

from forge.agent.protocols import CommandProvider
from forge.models.json_schema import JSONSchema

from .command import CO, Command, CommandParameter, P

logger = logging.getLogger(__name__)

_CP = TypeVar("_CP", bound=CommandProvider)


def command(
    names: Optional[list[str]] = None,
    description: Optional[str] = None,
    parameters: Optional[dict[str, JSONSchema]] = None,
) -> (
    Callable[[Callable[Concatenate[_CP, P], CO]], Command[P, CO]]
    | Callable[[Callable[P, CO]], Command[P, CO]]
):
    """
    Make a `Command` from a function or a method on a `CommandProvider`.
    All parameters are optional if the decorated function has a fully featured
    docstring. For the requirements of such a docstring,
    see `get_param_descriptions_from_docstring`.

    Args:
        names (list[str]): The names of the command.
            If not provided, the function name will be used.
        description (str): A brief description of what the command does.
            If not provided, the docstring until double line break will be used
            (or entire docstring if no double line break is found)
        parameters (dict[str, JSONSchema]): The parameters of the function
            that the command executes.
    """

    def decorator(
        func: Callable[P, CO] | Callable[Concatenate[_CP, P], CO]
    ) -> Command[P, CO]:
        # If names is not provided, use the function name
        _names = names or [func.__name__]

        # If description is not provided, use the first part of the docstring
        docstring = inspect.getdoc(func)
        if not (_description := description):
            if not docstring:
                raise ValueError(
                    "'description' is required if function has no docstring"
                )
            _description = get_clean_description_from_docstring(docstring)

        if not (_parameters := parameters):
            if not docstring:
                raise ValueError(
                    "'parameters' is required if function has no docstring"
                )

            # Combine descriptions from docstring with JSONSchemas from annotations
            param_descriptions = get_param_descriptions_from_docstring(docstring)
            _parameters = get_params_json_schemas(func)
            for param, param_schema in _parameters.items():
                if desc := param_descriptions.get(param):
                    param_schema.description = desc

        typed_parameters = [
            CommandParameter(
                name=param_name,
                spec=spec,
            )
            for param_name, spec in _parameters.items()
        ]

        # Wrap func with Command
        command = Command(
            names=_names,
            description=_description,
            # Method technically has a `self` parameter, but we can ignore that
            # since Python passes it internally.
            method=cast(Callable[P, CO], func),
            parameters=typed_parameters,
        )

        return command

    return decorator


def get_clean_description_from_docstring(docstring: str) -> str:
    """Return the part of the docstring before double line break or everything"""
    return re.sub(r"\s+", " ", docstring.split("\n\n")[0].strip())


def get_params_json_schemas(func: Callable) -> dict[str, JSONSchema]:
    """Gets the annotations of the given function and converts them to JSONSchemas"""
    result: dict[str, JSONSchema] = {}
    for name, parameter in inspect.signature(func).parameters.items():
        if name == "return":
            continue
        param_schema = result[name] = JSONSchema.from_python_type(parameter.annotation)
        if parameter.default:
            param_schema.default = parameter.default
            param_schema.required = False

    return result


def get_param_descriptions_from_docstring(docstring: str) -> dict[str, str]:
    """
    Get parameter descriptions from a docstring. Requirements for the docstring:
    - The section describing the parameters MUST start with `Params:` or `Args:`, in any
      capitalization.
    - An entry describing a parameter MUST be indented by 4 spaces.
    - An entry describing a parameter MUST start with the parameter name, an optional
      type annotation, followed by a colon `:`.
    - Continuations of parameter descriptions MUST be indented relative to the first
      line of the entry.
    - The docstring must not be indented as a whole. To get a docstring with the uniform
      indentation stripped off, use `inspect.getdoc(func)`.

    Example:
    ```python
    \"\"\"
    This is the description. This will be ignored.
    The description can span multiple lines,

    or contain any number of line breaks.

    Params:
        param1: This is a simple parameter description.
        param2 (list[str]): This parameter also has a type annotation.
        param3: This parameter has a long description. This means we will have to let it
          continue on the next line. The continuation is indented relative to the first
          line of the entry.

        param4: Extra line breaks to group parameters are allowed. This will not break
          the algorithm.

       This text is
      is indented by
     less than 4 spaces
    and is interpreted as the end of the `Params:` section.
    \"\"\"
    ```
    """
    param_descriptions: dict[str, str] = {}
    param_section = False
    last_param_name = ""
    for line in docstring.split("\n"):
        if not line.strip():  # ignore empty lines
            continue

        if line.lower().startswith(("params:", "args:")):
            param_section = True
            continue

        if param_section:
            if line.strip() and not line.startswith(" " * 4):  # end of section
                break

            line = line[4:]
            if line.startswith(" ") and last_param_name:  # continuation of description
                param_descriptions[last_param_name] += " " + line.strip()
            else:
                if _match := re.match(r"^(\w+).*?: (.*)", line):
                    param_name = _match.group(1)
                    param_desc = _match.group(2).strip()
                else:
                    logger.warning(
                        f"Invalid line in docstring's parameter section: {repr(line)}"
                    )
                    continue
                param_descriptions[param_name] = param_desc
                last_param_name = param_name
    return param_descriptions
