"""Resolve static Python imports and references without importing the backend."""

import ast

from backend.util.db_boundary_identity import reference_identity


def collect_references(
    sources: dict[str, str],
) -> tuple[
    dict[str, list[tuple[str, int, str]]], dict[str, set[str]], set[str], set[str]
]:
    trees = {name: ast.parse(source, filename=name) for name, source in sources.items()}
    exports = {name: _export_names(tree) for name, tree in trees.items()}
    exports.update(
        {
            name.removesuffix(".__init__"): names
            for name, names in list(exports.items())
            if name.endswith(".__init__")
        }
    )
    references: dict[str, list[tuple[str, int, str]]] = {}
    aliases: dict[str, set[str]] = {}
    callables: set[str] = set()
    rpc_clients: set[str] = set()
    for module, tree in trees.items():
        visitor = _References(module, exports)
        visitor.visit(tree)
        references.update(visitor.references)
        aliases.update(visitor.exports)
        callables.update(visitor.callables)
        rpc_clients.update(visitor.rpc_clients)
    return references, aliases, callables, rpc_clients


def resolve_alias(target: str, aliases: dict[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    pending = {target}
    resolved: set[str] = set()
    while pending:
        target = pending.pop()
        if target in seen:
            continue
        seen.add(target)
        parts = target.split(".")
        for length in range(len(parts), 0, -1):
            prefix = ".".join(parts[:length])
            replacements = {
                alias
                for alias in aliases.get(prefix, set())
                if alias != prefix
                and (target == prefix or not alias.startswith(prefix + "."))
            }
            if replacements:
                pending.update(
                    ".".join([alias, *parts[length:]]) for alias in replacements
                )
                break
        else:
            resolved.add(target)
    return resolved or {target}


def _export_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(
                alias.asname or alias.name.split(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.Assign):
            names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
    return {name for name in names if not name.startswith("_")}


class _References(ast.NodeVisitor):
    def __init__(self, module: str, exports: dict[str, set[str]]):
        self.module = module
        self.scope = module
        self.bindings: dict[str, set[str]] = {}
        self.references: dict[str, list[tuple[str, int, str]]] = {module: []}
        self.calls: list[ast.Call] = []
        self.exports: dict[str, set[str]] = {}
        self.callables: set[str] = set()
        self.rpc_clients: set[str] = set()
        self.module_exports = exports

    def visit_Module(self, node: ast.Module):
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.bindings[child.name] = {f"{self.module}.{child.name}"}
        for child in node.body:
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                self.visit(child)
        for child in node.body:
            if not isinstance(child, (ast.Import, ast.ImportFrom)):
                self.visit(child)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            self._bind(name, alias.name if alias.asname else name)
            self._record_import(alias.name, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        if node.level:
            package = self.module.removesuffix(".__init__").split(".")
            if not self.module.endswith(".__init__"):
                package.pop()
            module = ".".join(
                [*package[: len(package) - node.level + 1], module]
            ).rstrip(".")
        for alias in node.names:
            names = (
                self.module_exports.get(module, set())
                if alias.name == "*"
                else {alias.name}
            )
            for name in names:
                target = f"{module}.{name}"
                self._bind(alias.asname or name, target)
                self._record_import(target, node.lineno)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        for decorator in node.decorator_list:
            self.visit(decorator)
        target = f"{self.scope}.{node.name}"
        self._bind(node.name, target)
        args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        if node.args.vararg:
            args.append(node.args.vararg)
        if node.args.kwarg:
            args.append(node.args.kwarg)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        self._visit_scope(target, node.body, {arg.arg for arg in args})

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef):
        target = f"{self.scope}.{node.name}"
        self._bind(node.name, target)
        if any(
            "backend.util.service.AppServiceClient" in self._target(base)
            for base in node.bases
        ):
            self.rpc_clients.add(target)
        for base in [*node.bases, *node.decorator_list]:
            self.visit(base)
        self._visit_scope(target, node.body, set())
        for child in node.body:
            if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and child.name in {"__init__", "__new__"}:
                self.references[target].append(
                    (f"{target}.{child.name}", child.lineno, "constructor")
                )

    def visit_Call(self, node: ast.Call):
        self.calls.append(node)
        self.generic_visit(node)
        self.calls.pop()

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id in self.bindings:
            for target in self.bindings[node.id]:
                self._record(target, node)

    def visit_Attribute(self, node: ast.Attribute):
        targets = self._target(node)
        if targets:
            for target in targets:
                self._record(target, node)
            self._visit_receiver_arguments(node.value)
        else:
            self.generic_visit(node)

    def _visit_receiver_arguments(self, node: ast.expr):
        if isinstance(node, ast.Call):
            for argument in node.args:
                self.visit(argument)
            for keyword in node.keywords:
                self.visit(keyword.value)
            self._visit_receiver_arguments(node.func)
        elif isinstance(node, ast.Attribute):
            self._visit_receiver_arguments(node.value)

    def visit_Assign(self, node: ast.Assign):
        self.visit(node.value)
        target = self._target(node.value)
        for name in node.targets:
            self._bind_assignment(name, target)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.value is None:
            return
        self.visit(node.value)
        self._bind_assignment(node.target, self._target(node.value))

    def _bind_assignment(self, name: ast.expr, target: set[str]):
        if isinstance(name, ast.Name):
            self._bind(name.id, target)
            return
        if target:
            return
        if isinstance(name, ast.Starred):
            self._bind_assignment(name.value, target)
        elif isinstance(name, (ast.Tuple, ast.List)):
            for element in name.elts:
                self._bind_assignment(element, target)

    def visit_If(self, node: ast.If):
        target = self._target(node.test)
        if target == {"typing.TYPE_CHECKING"} or (
            isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
        ):
            for statement in node.orelse:
                self.visit(statement)
            return
        self.visit(node.test)
        original = self.bindings.copy()
        for statement in node.body:
            self.visit(statement)
        body_bindings = self.bindings
        self.bindings = original.copy()
        for statement in node.orelse:
            self.visit(statement)
        self.bindings = {
            name: body_bindings.get(name, set()) | self.bindings.get(name, set())
            for name in body_bindings.keys() | self.bindings.keys()
        }

    def _visit_scope(self, target: str, body: list[ast.stmt], shadowed: set[str]):
        outer_scope, outer_bindings = self.scope, self.bindings
        self.scope = target
        self.bindings = {
            name: value
            for name, value in outer_bindings.items()
            if name not in shadowed
        }
        self.references[target] = []
        self.callables.add(target)
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.bindings[node.name] = {f"{target}.{node.name}"}
        for statement in body:
            self.visit(statement)
        self.scope, self.bindings = outer_scope, outer_bindings

    def _bind(self, name: str, target: str | set[str]):
        targets = {target} if isinstance(target, str) else target
        self.bindings[name] = targets
        if self.scope == self.module:
            self.exports[f"{self.module}.{name}"] = targets
            if self.module.endswith(".__init__"):
                self.exports[f"{self.module.removesuffix('.__init__')}.{name}"] = (
                    targets
                )

    def _record_import(self, target: str, lineno: int):
        self.references[self.scope].append((target, lineno, "import"))

    def _record(self, target: str, node: ast.expr):
        syntax = self.calls[-1] if self.calls else node
        identity = reference_identity(syntax, self._target)
        self.references[self.scope].append((target, node.lineno, identity))

    def _target(self, node: ast.expr) -> set[str]:
        if isinstance(node, ast.Name):
            return self.bindings.get(node.id, set())
        if isinstance(node, ast.Attribute):
            if node.attr == "prisma":
                return {".prisma"}
            return {f"{parent}.{node.attr}" for parent in self._target(node.value)}
        if isinstance(node, ast.Call):
            return self._target(node.func)
        if isinstance(node, ast.IfExp):
            return self._target(node.body) | self._target(node.orelse)
        return set()
