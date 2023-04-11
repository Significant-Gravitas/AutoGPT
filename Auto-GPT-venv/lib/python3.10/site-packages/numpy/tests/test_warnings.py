"""
Tests which scan for certain occurrences in the code, they may not find
all of these occurrences but should catch almost all.
"""
import pytest

from pathlib import Path
import ast
import tokenize
import numpy

class ParseCall(ast.NodeVisitor):
    def __init__(self):
        self.ls = []

    def visit_Attribute(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        self.ls.append(node.attr)

    def visit_Name(self, node):
        self.ls.append(node.id)


class FindFuncs(ast.NodeVisitor):
    def __init__(self, filename):
        super().__init__()
        self.__filename = filename

    def visit_Call(self, node):
        p = ParseCall()
        p.visit(node.func)
        ast.NodeVisitor.generic_visit(self, node)

        if p.ls[-1] == 'simplefilter' or p.ls[-1] == 'filterwarnings':
            if node.args[0].s == "ignore":
                raise AssertionError(
                    "warnings should have an appropriate stacklevel; found in "
                    "{} on line {}".format(self.__filename, node.lineno))

        if p.ls[-1] == 'warn' and (
                len(p.ls) == 1 or p.ls[-2] == 'warnings'):

            if "testing/tests/test_warnings.py" == self.__filename:
                # This file
                return

            # See if stacklevel exists:
            if len(node.args) == 3:
                return
            args = {kw.arg for kw in node.keywords}
            if "stacklevel" in args:
                return
            raise AssertionError(
                "warnings should have an appropriate stacklevel; found in "
                "{} on line {}".format(self.__filename, node.lineno))


@pytest.mark.slow
def test_warning_calls():
    # combined "ignore" and stacklevel error
    base = Path(numpy.__file__).parent

    for path in base.rglob("*.py"):
        if base / "testing" in path.parents:
            continue
        if path == base / "__init__.py":
            continue
        if path == base / "random" / "__init__.py":
            continue
        # use tokenize to auto-detect encoding on systems where no
        # default encoding is defined (e.g. LANG='C')
        with tokenize.open(str(path)) as file:
            tree = ast.parse(file.read())
            FindFuncs(path).visit(tree)
