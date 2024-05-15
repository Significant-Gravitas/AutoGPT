import ast
import collections
import datetime
import json
import logging
import pathlib
import re
import typing
import black
import isort

from autogpt.utils.function.model import FunctionDef, ObjectType, ValidationResponse
from autogpt.utils.function.visitor import FunctionVisitor
from autogpt.utils.function.util import (
    genererate_line_error,
    generate_object_code,
    generate_compiled_code,
    validate_matching_function,
)
from autogpt.utils.function.exec import (
    exec_external_on_contents,
    ExecError,
    PROJECT_TEMP_DIR,
    DEFAULT_DEPS,
    execute_command,
    setup_if_required,
)

logger = logging.getLogger(__name__)


class CodeValidator:
    def __init__(
        self,
        function_name: str | None = None,
        available_functions: dict[str, FunctionDef] | None = None,
        available_objects: dict[str, ObjectType] | None = None,
    ):
        self.func_name: str = function_name or ""
        self.available_functions: dict[str, FunctionDef] = available_functions or {}
        self.available_objects: dict[str, ObjectType] = available_objects or {}

    async def reformat_code(
        self,
        code: str,
        packages: list[str] = [],
    ) -> str:
        """
        Reformat the code snippet
        Args:
            code (str): The code snippet to reformat
            packages (list[str]): The list of packages to validate
        Returns:
            str: The reformatted code snippet
        """
        try:
            code = (
                await self.validate_code(
                    raw_code=code,
                    packages=packages,
                    raise_validation_error=False,
                    add_code_stubs=False,
                )
            ).get_compiled_code()
        except Exception as e:
            # We move on with unfixed code if there's an error
            logger.warning(
                f"Error formatting code for route #{self.func_name}: {e}"
            )
            raise e

        for formatter in [
            lambda code: isort.code(code),
            lambda code: black.format_str(code, mode=black.FileMode()),
        ]:
            try:
                code = formatter(code)
            except Exception as e:
                # We move on with unformatted code if there's an error
                logger.warning(
                    f"Error formatting code for route #{self.func_name}: {e}"
                )

        return code

    async def validate_code(
        self,
        raw_code: str,
        packages: list[str] = [],
        raise_validation_error: bool = True,
        add_code_stubs: bool = True,
        call_cnt: int = 0,
    ) -> ValidationResponse:
        """
        Validate the code snippet for any error
        Args:
            packages (list[Package]): The list of packages to validate
            raw_code (str): The code snippet to validate
        Returns:
            ValidationResponse: The response of the validation
        Raise:
            ValidationError(e): The list of validation errors in the code snippet
        """
        validation_errors: list[str] = []

        try:
            tree = ast.parse(raw_code)
            visitor = FunctionVisitor()
            visitor.visit(tree)
            validation_errors.extend(visitor.errors)
        except Exception as e:
            # parse invalid code line and add it to the error message
            error = f"Error parsing code: {e}"

            if "async lambda" in raw_code:
                error += "\nAsync lambda is not supported in Python. "
                "Use async def instead!"

            if line := re.search(r"line (\d+)", error):
                raise Exception(
                    genererate_line_error(error, raw_code, int(line.group(1))))
            else:
                raise Exception(error)

        # Eliminate duplicate visitor.functions and visitor.objects, prefer the last one
        visitor.imports = list(set(visitor.imports))
        visitor.functions = list({f.name: f for f in visitor.functions}.values())
        visitor.objects = list(
            {
                o.name: o
                for o in visitor.objects
                if o.name not in self.available_objects
            }.values()
        )

        # Add implemented functions into the main function, only link the stub functions
        deps_funcs = [f for f in visitor.functions if f.is_implemented]
        stub_funcs = [f for f in visitor.functions if not f.is_implemented]

        objects_block = zip(
            ["\n\n" + generate_object_code(obj) for obj in visitor.objects],
            visitor.objectsIdx,
        )
        functions_block = zip(
            ["\n\n" + fun.function_code for fun in deps_funcs], visitor.functionsIdx
        )
        globals_block = zip(
            ["\n\n" + glob for glob in visitor.globals], visitor.globalsIdx
        )
        function_code = "".join(
            code
            for code, _ in sorted(
                list(objects_block) + list(functions_block) + list(globals_block),
                key=lambda x: x[1],
            )
        ).strip()

        # No need to validate main function if it's not provided
        if self.func_name:
            main_func = self.__validate_main_function(
                deps_funcs=deps_funcs,
                function_code=function_code,
                validation_errors=validation_errors,
            )
            function_template = main_func.function_template
        else:
            function_template = None

        # Validate that code is not re-declaring any existing entities.
        already_declared_entities = set(
            [
                obj.name
                for obj in visitor.objects
                if obj.name in self.available_objects.keys()
            ]
            + [
                func.name
                for func in visitor.functions
                if func.name in self.available_functions.keys()
            ]
        )
        if not already_declared_entities:
            validation_errors.append(
                "These class/function names has already been declared in the code, "
                "no need to declare them again: "
                + ", ".join(already_declared_entities)
            )

        result = ValidationResponse(
            function_name=self.func_name,
            available_objects=self.available_objects,
            available_functions=self.available_functions,
            rawCode=function_code,
            imports=visitor.imports.copy(),
            objects=[],  # Objects will be bundled in the function_code instead.
            template=function_template or "",
            functionCode=function_code,
            functions=stub_funcs,
            packages=packages,
        )

        # Execute static validators and fixers.
        # print('old compiled code import ---->', result.imports)
        old_compiled_code = generate_compiled_code(result, add_code_stubs)
        validation_errors.extend(
            await static_code_analysis(result)
        )
        new_compiled_code = result.get_compiled_code()

        # Auto-fixer works, retry validation (limit to 5 times, to avoid infinite loop)
        if old_compiled_code != new_compiled_code and call_cnt < 5:
            return await self.validate_code(
                packages=packages,
                raw_code=new_compiled_code,
                raise_validation_error=raise_validation_error,
                add_code_stubs=add_code_stubs,
                call_cnt=call_cnt + 1,
            )

        if validation_errors:
            if raise_validation_error:
                error_message = "".join("\n * " + e for e in validation_errors)
                raise Exception("Error validating code: " + error_message)
            else:
                # This should happen only on `reformat_code` call
                logger.warning("Error validating code: %s", validation_errors)

        return result

    def __validate_main_function(
        self,
        deps_funcs: list[FunctionDef],
        function_code: str,
        validation_errors: list[str],
    ) -> FunctionDef:
        """
        Validate the main function body and signature
        Returns:
            tuple[str, FunctionDef]: The function ID and the function object
        """
        # Validate that the main function is implemented.
        func_obj = next((f for f in deps_funcs if f.name == self.func_name), None)
        if not func_obj or not func_obj.is_implemented:
            raise Exception(
                f"Main Function body {self.func_name} is not implemented."
                f" Please complete the implementation of this function!"
            )
        func_obj.function_code = function_code

        # Validate that the main function is matching the expected signature.
        func_req: FunctionDef | None = self.available_functions.get(self.func_name)
        if not func_req:
            raise AssertionError(f"Function {self.func_name} does not exist on DB")
        try:
            validate_matching_function(func_obj, func_req)
        except Exception as e:
            validation_errors.append(e.__str__())

        return func_obj


# ======= Static Code Validation Helper Functions =======#


async def static_code_analysis(func: ValidationResponse) -> list[str]:
    """
    Run static code analysis on the function code and mutate the function code to
    fix any issues.
    Args:
        func (ValidationResponse):
            The function to run static code analysis on. `func` will be mutated.
    Returns:
        list[str]: The list of validation errors
    """
    validation_errors = []
    validation_errors += await __execute_ruff(func)
    validation_errors += await __execute_pyright(func)

    return validation_errors


CODE_SEPARATOR = "#------Code-Start------#"


def __pack_import_and_function_code(func: ValidationResponse) -> str:
    return "\n".join(func.imports + [CODE_SEPARATOR, func.rawCode])


def __unpack_import_and_function_code(code: str) -> tuple[list[str], str]:
    split = code.split(CODE_SEPARATOR)
    return split[0].splitlines(), split[1].strip()


async def __execute_ruff(func: ValidationResponse) -> list[str]:
    code = __pack_import_and_function_code(func)

    try:
        # Currently Disabled Rule List
        # E402 module level import not at top of file
        # F841 local variable is assigned to but never used
        code = await exec_external_on_contents(
            command_arguments=[
                "ruff",
                "check",
                "--fix",
                "--ignore",
                "F841",
                "--ignore",
                "E402",
                "--ignore",
                "F811",  # Redefinition of unused '...' from line ...
            ],
            file_contents=code,
            suffix=".py",
            raise_file_contents_on_error=True,
        )
        func.imports, func.rawCode = __unpack_import_and_function_code(code)
        return []

    except ExecError as e:
        if e.content:
            # Ruff failed, but the code is reformatted
            code = e.content
            e = str(e)

        error_messages = [
            v
            for v in str(e).split("\n")
            if v.strip()
            if re.match(r"Found \d+ errors?\.*", v) is None
        ]

        added_imports, error_messages = await __fix_missing_imports(
            error_messages, func
        )

        # Append problematic line to the error message or add it as TODO line
        validation_errors: list[str] = []
        split_pattern = r"(.+):(\d+):(\d+): (.+)"
        for error_message in error_messages:
            error_split = re.match(split_pattern, error_message)

            if not error_split:
                error = error_message
            else:
                _, line, _, error = error_split.groups()
                error = genererate_line_error(error, code, int(line))

            validation_errors.append(error)

        func.imports, func.rawCode = __unpack_import_and_function_code(code)
        func.imports.extend(added_imports)  # Avoid line-code change, do it at the end.

        return validation_errors


async def __execute_pyright(func: ValidationResponse) -> list[str]:
    code = __pack_import_and_function_code(func)
    validation_errors: list[str] = []

    # Create temporary directory under the TEMP_DIR with random name
    temp_dir = PROJECT_TEMP_DIR / (func.function_name)
    py_path = await setup_if_required(temp_dir)

    async def __execute_pyright_commands(code: str) -> list[str]:
        try:
            await execute_command(
                ["pip", "install", "-r", "requirements.txt"], temp_dir, py_path
            )
        except Exception as e:
            # Unknown deps should be reported as validation errors
            validation_errors.append(e.__str__())

        # execute pyright
        result = await execute_command(
            ["pyright", "--outputjson"], temp_dir, py_path, raise_on_error=False
        )
        if not result:
            return []

        try:
            json_response = json.loads(result)["generalDiagnostics"]
        except Exception as e:
            logger.error(f"Error parsing pyright output, error: {e} output: {result}")
            raise e

        for e in json_response:
            rule: str = e.get("rule", "")
            severity: str = e.get("severity", "")
            excluded_rules = ["reportRedeclaration"]
            if severity != "error" or any([rule.startswith(r) for r in excluded_rules]):
                continue

            e = genererate_line_error(
                error=f"{e['message']}. {e.get('rule', '')}",
                code=code,
                line_number=e["range"]["start"]["line"] + 1,
            )
            validation_errors.append(e)

        # read code from code.py. split the code into imports and raw code
        code = open(f"{temp_dir}/code.py").read()
        func.imports, func.rawCode = __unpack_import_and_function_code(code)

        return validation_errors

    packages = "\n".join(
        [str(p) for p in func.packages if p not in DEFAULT_DEPS]
    )
    (temp_dir / "requirements.txt").write_text(packages)
    (temp_dir / "code.py").write_text(code)

    return await __execute_pyright_commands(code)


async def find_module_dist_and_source(
    module: str, py_path: pathlib.Path | str
) -> typing.Tuple[pathlib.Path | None, pathlib.Path | None]:
    # Find the module in the env
    modules_path = pathlib.Path(py_path).parent / "lib" / "python3.11" / "site-packages"
    matches = modules_path.glob(f"{module}*")

    # resolve the generator to an array
    matches = list(matches)
    if not matches:
        return None, None

    # find the dist info path and the module path
    dist_info_path: typing.Optional[pathlib.Path] = None
    module_path: typing.Optional[pathlib.Path] = None

    # find the dist info path
    for match in matches:
        if re.match(f"{module}-[0-9]+.[0-9]+.[0-9]+.dist-info", match.name):
            dist_info_path = match
            break
    # Get the module path
    for match in matches:
        if module == match.name:
            module_path = match
            break

    return dist_info_path, module_path

AUTO_IMPORT_TYPES: dict[str, str] = {
    "Enum": "from enum import Enum",
    "array": "from array import array",
}
for t in typing.__all__:
    AUTO_IMPORT_TYPES[t] = f"from typing import {t}"
for t in datetime.__all__:
    AUTO_IMPORT_TYPES[t] = f"from datetime import {t}"
for t in collections.__all__:
    AUTO_IMPORT_TYPES[t] = f"from collections import {t}"


async def __fix_missing_imports(
    errors: list[str], func: ValidationResponse
) -> tuple[set[str], list[str]]:
    """
    Generate missing imports based on the errors
    Args:
        errors (list[str]): The list of errors
        func (ValidationResponse): The function to fix the imports
    Returns:
        tuple[set[str], list[str]]: The set of missing imports and the list
        of non-missing import errors
    """
    missing_imports = []
    filtered_errors = []
    for error in errors:
        pattern = r"Undefined name `(.+?)`"
        match = re.search(pattern, error)
        if not match:
            filtered_errors.append(error)
            continue

        missing = match.group(1)
        if missing in AUTO_IMPORT_TYPES:
            missing_imports.append(AUTO_IMPORT_TYPES[missing])
        elif missing in func.available_functions:
            # TODO FIX THIS!! IMPORT AUTOGPT CORRECY SERVICE.
            missing_imports.append(f"from project.{missing}_service import {missing}")
        elif missing in func.available_objects:
            # TODO FIX THIS!! IMPORT AUTOGPT CORRECY SERVICE.
            missing_imports.append(f"from project.{missing}_object import {missing}")
        else:
            filtered_errors.append(error)

    return set(missing_imports), filtered_errors
