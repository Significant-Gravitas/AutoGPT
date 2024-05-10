import ast
import re

from autogpt.utils.function.model import FunctionDef, ObjectType, ObjectField
from autogpt.utils.function.util import normalize_type, PYTHON_TYPES


class FunctionVisitor(ast.NodeVisitor):
    """
    Visits a Python AST and extracts function definitions and Pydantic class definitions

    To use this class, create an instance and call the visit method with the AST.
    as the argument The extracted function definitions and Pydantic class definitions
    can be accessed from the functions and objects attributes respectively.

    Example:
    ```
    visitor = FunctionVisitor()
    visitor.visit(ast.parse("def foo(x: int) -> int: return x"))
    print(visitor.functions)
    ```
    """

    def __init__(self):
        self.functions: list[FunctionDef] = []
        self.functionsIdx: list[int] = []
        self.objects: list[ObjectType] = []
        self.objectsIdx: list[int] = []
        self.globals: list[str] = []
        self.globalsIdx: list[int] = []
        self.imports: list[str] = []
        self.errors: list[str] = []

    def visit_Import(self, node):
        for alias in node.names:
            import_line = f"import {alias.name}"
            if alias.asname:
                import_line += f" as {alias.asname}"
            self.imports.append(import_line)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            import_line = f"from {node.module} import {alias.name}"
            if alias.asname:
                import_line += f" as {alias.asname}"
            self.imports.append(import_line)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        # treat async functions as normal functions
        self.visit_FunctionDef(node)  # type: ignore

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        args = []
        for arg in node.args.args:
            arg_type = ast.unparse(arg.annotation) if arg.annotation else "object"
            args.append((arg.arg, normalize_type(arg_type)))
        return_type = (
            normalize_type(ast.unparse(node.returns)) if node.returns else None
        )

        # Extract doc_string & function body
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            doc_string = node.body[0].value.s.strip()
            template_body = [node.body[0], ast.Pass()]
            is_implemented = not isinstance(node.body[1], ast.Pass)
        else:
            doc_string = ""
            template_body = [ast.Pass()]
            is_implemented = not isinstance(node.body[0], ast.Pass)

        # Construct function template
        original_body = node.body.copy()
        node.body = template_body  # type: ignore
        function_template = ast.unparse(node)
        node.body = original_body

        function_code = ast.unparse(node)
        if "await" in function_code and "async def" not in function_code:
            function_code = function_code.replace("def ", "async def ")
            function_template = function_template.replace("def ", "async def ")

        def split_doc(keywords: list[str], doc: str) -> tuple[str, str]:
            for keyword in keywords:
                if match := re.search(f"{keyword}\\s?:", doc):
                    return doc[: match.start()], doc[match.end() :]
            return doc, ""

        # Decompose doc_pattern into func_doc, args_doc, rets_doc, errs_doc, usage_doc
        # by splitting in reverse order
        func_doc = doc_string
        func_doc, usage_doc = split_doc(
            ["Ex", "Usage", "Usages", "Example", "Examples"], func_doc
        )
        func_doc, errs_doc = split_doc(["Error", "Errors", "Raise", "Raises"], func_doc)
        func_doc, rets_doc = split_doc(["Return", "Returns"], func_doc)
        func_doc, args_doc = split_doc(
            ["Arg", "Args", "Argument", "Arguments"], func_doc
        )

        # Extract Func
        function_desc = func_doc.strip()

        # Extract Args
        args_descs = {}
        split_pattern = r"\n(\s+.+):"
        for match in reversed(list(re.finditer(split_pattern, string=args_doc))):
            arg = match.group(1).strip().split(" ")[0]
            desc = args_doc.rsplit(match.group(1), 1)[1].strip(": ")
            args_descs[arg] = desc.strip()
            args_doc = args_doc[: match.start()]

        # Extract Returns
        return_desc = ""
        if match := re.match(split_pattern, string=rets_doc):
            return_desc = rets_doc[match.end() :].strip()

        self.functions.append(
            FunctionDef(
                name=node.name,
                arg_types=args,
                arg_descs=args_descs,
                return_type=return_type,
                return_desc=return_desc,
                is_implemented=is_implemented,
                function_desc=function_desc,
                function_template=function_template,
                function_code=function_code,
            )
        )
        self.functionsIdx.append(node.lineno)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        Visits a ClassDef node in the AST and checks if it is a Pydantic class.
        If it is a Pydantic class, adds its name to the list of Pydantic classes.
        """
        is_pydantic = any(
            [
                (isinstance(base, ast.Name) and base.id == "BaseModel")
                or (isinstance(base, ast.Attribute) and base.attr == "BaseModel")
                for base in node.bases
            ]
        )
        is_enum = any(
            [
                (isinstance(base, ast.Name) and base.id.endswith("Enum"))
                or (isinstance(base, ast.Attribute) and base.attr.endswith("Enum"))
                for base in node.bases
            ]
        )
        is_implemented = not any(isinstance(v, ast.Pass) for v in node.body)
        doc_string = ""
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            doc_string = node.body[0].value.s.strip()

        if node.name in PYTHON_TYPES:
            self.errors.append(
                f"Can't declare class with a Python built-in name "
                f"`{node.name}`. Please use a different name."
            )

        fields = []
        methods = []
        for v in node.body:
            if isinstance(v, ast.AnnAssign):
                field = ObjectField(
                    name=ast.unparse(v.target),
                    type=normalize_type(ast.unparse(v.annotation)),
                    value=ast.unparse(v.value) if v.value else None,
                )
                if field.value is None and field.type.startswith("Optional"):
                    field.value = "None"
            elif isinstance(v, ast.Assign):
                if len(v.targets) > 1:
                    self.errors.append(
                        f"Class {node.name} has multiple assignments in a single line."
                    )
                field = ObjectField(
                    name=ast.unparse(v.targets[0]),
                    type=type(ast.unparse(v.value)).__name__,
                    value=ast.unparse(v.value) if v.value else None,
                )
            elif isinstance(v, ast.Expr) and isinstance(v.value, ast.Constant):
                # skip comments and docstrings
                continue
            else:
                methods.append(ast.unparse(v))
                continue
            fields.append(field)

        self.objects.append(
            ObjectType(
                name=node.name,
                code="\n".join(methods),
                description=doc_string,
                Fields=fields,
                is_pydantic=is_pydantic,
                is_enum=is_enum,
                is_implemented=is_implemented,
            )
        )
        self.objectsIdx.append(node.lineno)

        """Some class are simply used as a type and doesn't have any new fields"""
        # if not is_implemented:
        #     raise ValidationError(
        #         f"Class {node.name} is not implemented. "
        #         f"Please complete the implementation of this class!"
        #     )

    def visit(self, node):
        if (
            isinstance(node, ast.Assign)
            or isinstance(node, ast.AnnAssign)
            or isinstance(node, ast.AugAssign)
        ) and node.col_offset == 0:
            self.globals.append(ast.unparse(node))
            self.globalsIdx.append(node.lineno)
        super().visit(node)
