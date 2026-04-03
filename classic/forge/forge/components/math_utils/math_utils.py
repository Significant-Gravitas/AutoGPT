import ast
import json
import logging
import math
import operator
import statistics
from typing import Any, Iterator, Optional

from pydantic import BaseModel

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider, DirectiveProvider
from forge.command import Command, command
from forge.models.json_schema import JSONSchema
from forge.utils.exceptions import CommandExecutionError

logger = logging.getLogger(__name__)


class MathUtilsConfiguration(BaseModel):
    pass  # No configuration needed for now


class SafeEvaluator(ast.NodeVisitor):
    """Safe evaluator for mathematical expressions."""

    # Allowed operators
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # Allowed functions
    FUNCTIONS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "pow": pow,
    }

    # Allowed constants
    CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
        "inf": float("inf"),
    }

    def visit(self, node: ast.AST) -> float:
        return super().visit(node)

    def generic_visit(self, node: ast.AST) -> float:
        raise CommandExecutionError(
            f"Unsupported operation: {type(node).__name__}. "
            "Only basic arithmetic, math functions, and constants are allowed."
        )

    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            return node.value
        raise CommandExecutionError(f"Invalid constant: {node.value}")

    def visit_Num(self, node: ast.Num) -> float:  # Python 3.7 compatibility
        return float(node.n)  # type: ignore[attr-defined]

    def visit_Name(self, node: ast.Name) -> float:
        if node.id in self.CONSTANTS:
            return self.CONSTANTS[node.id]
        avail = list(self.CONSTANTS.keys())
        raise CommandExecutionError(f"Unknown variable: {node.id}. Available: {avail}")

    def visit_BinOp(self, node: ast.BinOp) -> float:
        if type(node.op) not in self.OPERATORS:
            raise CommandExecutionError(
                f"Unsupported operator: {type(node.op).__name__}"
            )
        left = self.visit(node.left)
        right = self.visit(node.right)
        return self.OPERATORS[type(node.op)](left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        if type(node.op) not in self.OPERATORS:
            raise CommandExecutionError(
                f"Unsupported unary operator: {type(node.op).__name__}"
            )
        operand = self.visit(node.operand)
        return self.OPERATORS[type(node.op)](operand)

    def visit_Call(self, node: ast.Call) -> float:
        if not isinstance(node.func, ast.Name):
            raise CommandExecutionError("Only direct function calls are allowed")

        func_name = node.func.id
        if func_name not in self.FUNCTIONS:
            avail = list(self.FUNCTIONS.keys())
            raise CommandExecutionError(
                f"Unknown function: {func_name}. Available: {avail}"
            )

        args = [self.visit(arg) for arg in node.args]
        return self.FUNCTIONS[func_name](*args)

    def visit_List(self, node: ast.List) -> list:
        return [self.visit(elt) for elt in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> tuple:
        return tuple(self.visit(elt) for elt in node.elts)


class MathUtilsComponent(
    DirectiveProvider, CommandProvider, ConfigurableComponent[MathUtilsConfiguration]
):
    """Provides commands for mathematical calculations and statistics."""

    config_class = MathUtilsConfiguration

    def __init__(self, config: Optional[MathUtilsConfiguration] = None):
        ConfigurableComponent.__init__(self, config)

    def get_resources(self) -> Iterator[str]:
        yield "Ability to perform mathematical calculations and statistical analysis."

    def get_commands(self) -> Iterator[Command]:
        yield self.calculate
        yield self.statistics_calc
        yield self.convert_units

    @command(
        ["calculate", "eval_math", "compute"],
        "Evaluate math expressions. Supports operators, sqrt, sin, cos, log, etc.",
        {
            "expression": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Expression to evaluate (e.g. '2 * pi + sqrt(16)')",
                required=True,
            ),
        },
    )
    def calculate(self, expression: str) -> str:
        """Safely evaluate a mathematical expression.

        Args:
            expression: The expression to evaluate

        Returns:
            str: The result as JSON
        """
        try:
            tree = ast.parse(expression, mode="eval")
            evaluator = SafeEvaluator()
            result = evaluator.visit(tree)

            return json.dumps({"expression": expression, "result": result}, indent=2)

        except SyntaxError as e:
            raise CommandExecutionError(f"Invalid expression syntax: {e}")
        except ZeroDivisionError:
            raise CommandExecutionError("Division by zero")
        except OverflowError:
            raise CommandExecutionError("Result too large")
        except Exception as e:
            raise CommandExecutionError(f"Calculation error: {e}")

    @command(
        ["statistics", "stats_calc"],
        "Calculate statistics on a list of numbers.",
        {
            "numbers": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                items=JSONSchema(type=JSONSchema.Type.NUMBER),
                description="List of numbers to analyze",
                required=True,
            ),
            "operations": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                items=JSONSchema(type=JSONSchema.Type.STRING),
                description="Stats to compute: mean, median, mode, etc. (default: all)",
                required=False,
            ),
        },
    )
    def statistics_calc(
        self,
        numbers: list[float],
        operations: list[str] | None = None,
    ) -> str:
        """Calculate statistics on a list of numbers.

        Args:
            numbers: List of numbers
            operations: Which statistics to compute

        Returns:
            str: JSON with requested statistics
        """
        if not numbers:
            raise CommandExecutionError("Empty list provided")

        all_ops = [
            "mean",
            "median",
            "mode",
            "stdev",
            "variance",
            "min",
            "max",
            "sum",
            "count",
        ]
        ops = operations if operations else all_ops

        result = {}
        errors = []

        for op in ops:
            try:
                if op == "mean":
                    result["mean"] = statistics.mean(numbers)
                elif op == "median":
                    result["median"] = statistics.median(numbers)
                elif op == "mode":
                    try:
                        result["mode"] = statistics.mode(numbers)
                    except statistics.StatisticsError:
                        result["mode"] = None
                        errors.append("No unique mode found")
                elif op == "stdev":
                    if len(numbers) > 1:
                        result["stdev"] = statistics.stdev(numbers)
                    else:
                        result["stdev"] = 0
                elif op == "variance":
                    if len(numbers) > 1:
                        result["variance"] = statistics.variance(numbers)
                    else:
                        result["variance"] = 0
                elif op == "min":
                    result["min"] = min(numbers)
                elif op == "max":
                    result["max"] = max(numbers)
                elif op == "sum":
                    result["sum"] = sum(numbers)
                elif op == "count":
                    result["count"] = len(numbers)
                else:
                    errors.append(f"Unknown operation: {op}")
            except Exception as e:
                errors.append(f"{op}: {e}")

        output: dict[str, Any] = {"statistics": result}
        if errors:
            output["errors"] = errors

        return json.dumps(output, indent=2)

    @command(
        ["convert_units", "unit_conversion"],
        "Convert between units of measurement.",
        {
            "value": JSONSchema(
                type=JSONSchema.Type.NUMBER,
                description="The value to convert",
                required=True,
            ),
            "from_unit": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Source unit (e.g., 'km', 'miles', 'celsius', 'kg')",
                required=True,
            ),
            "to_unit": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Target unit (e.g., 'm', 'feet', 'fahrenheit', 'lbs')",
                required=True,
            ),
        },
    )
    def convert_units(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
    ) -> str:
        """Convert between units of measurement.

        Args:
            value: The value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            str: JSON with conversion result
        """
        # Normalize unit names
        from_unit = from_unit.lower().strip()
        to_unit = to_unit.lower().strip()

        # Unit conversions to base units
        # Length -> meters
        length_to_m = {
            "m": 1,
            "meter": 1,
            "meters": 1,
            "km": 1000,
            "kilometer": 1000,
            "kilometers": 1000,
            "cm": 0.01,
            "centimeter": 0.01,
            "centimeters": 0.01,
            "mm": 0.001,
            "millimeter": 0.001,
            "millimeters": 0.001,
            "mi": 1609.344,
            "mile": 1609.344,
            "miles": 1609.344,
            "yd": 0.9144,
            "yard": 0.9144,
            "yards": 0.9144,
            "ft": 0.3048,
            "foot": 0.3048,
            "feet": 0.3048,
            "in": 0.0254,
            "inch": 0.0254,
            "inches": 0.0254,
        }

        # Weight -> kilograms
        weight_to_kg = {
            "kg": 1,
            "kilogram": 1,
            "kilograms": 1,
            "g": 0.001,
            "gram": 0.001,
            "grams": 0.001,
            "mg": 0.000001,
            "milligram": 0.000001,
            "milligrams": 0.000001,
            "lb": 0.453592,
            "lbs": 0.453592,
            "pound": 0.453592,
            "pounds": 0.453592,
            "oz": 0.0283495,
            "ounce": 0.0283495,
            "ounces": 0.0283495,
        }

        # Temperature (special handling)
        temp_units = {"c", "celsius", "f", "fahrenheit", "k", "kelvin"}

        # Volume -> liters
        volume_to_l = {
            "l": 1,
            "liter": 1,
            "liters": 1,
            "litre": 1,
            "litres": 1,
            "ml": 0.001,
            "milliliter": 0.001,
            "milliliters": 0.001,
            "gal": 3.78541,
            "gallon": 3.78541,
            "gallons": 3.78541,
            "qt": 0.946353,
            "quart": 0.946353,
            "quarts": 0.946353,
            "pt": 0.473176,
            "pint": 0.473176,
            "pints": 0.473176,
            "cup": 0.236588,
            "cups": 0.236588,
            "fl oz": 0.0295735,
            "floz": 0.0295735,
        }

        # Time -> seconds
        time_to_s = {
            "s": 1,
            "sec": 1,
            "second": 1,
            "seconds": 1,
            "min": 60,
            "minute": 60,
            "minutes": 60,
            "h": 3600,
            "hr": 3600,
            "hour": 3600,
            "hours": 3600,
            "d": 86400,
            "day": 86400,
            "days": 86400,
            "week": 604800,
            "weeks": 604800,
        }

        # Data -> bytes
        data_to_bytes = {
            "b": 1,
            "byte": 1,
            "bytes": 1,
            "kb": 1024,
            "kilobyte": 1024,
            "kilobytes": 1024,
            "mb": 1024**2,
            "megabyte": 1024**2,
            "megabytes": 1024**2,
            "gb": 1024**3,
            "gigabyte": 1024**3,
            "gigabytes": 1024**3,
            "tb": 1024**4,
            "terabyte": 1024**4,
            "terabytes": 1024**4,
        }

        # Temperature conversions
        if from_unit in temp_units and to_unit in temp_units:
            # Convert to Celsius first
            if from_unit in ("c", "celsius"):
                celsius = value
            elif from_unit in ("f", "fahrenheit"):
                celsius = (value - 32) * 5 / 9
            elif from_unit in ("k", "kelvin"):
                celsius = value - 273.15
            else:
                raise CommandExecutionError(f"Unknown temperature unit: {from_unit}")

            # Convert from Celsius to target
            if to_unit in ("c", "celsius"):
                result = celsius
            elif to_unit in ("f", "fahrenheit"):
                result = celsius * 9 / 5 + 32
            elif to_unit in ("k", "kelvin"):
                result = celsius + 273.15
            else:
                raise CommandExecutionError(f"Unknown temperature unit: {to_unit}")

            return json.dumps(
                {
                    "value": value,
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "result": round(result, 6),
                },
                indent=2,
            )

        # Find matching conversion table
        for conv_table in [
            length_to_m,
            weight_to_kg,
            volume_to_l,
            time_to_s,
            data_to_bytes,
        ]:
            if from_unit in conv_table and to_unit in conv_table:
                # Convert through base unit
                base_value = value * conv_table[from_unit]
                result = base_value / conv_table[to_unit]

                return json.dumps(
                    {
                        "value": value,
                        "from_unit": from_unit,
                        "to_unit": to_unit,
                        "result": round(result, 6),
                    },
                    indent=2,
                )

        raise CommandExecutionError(
            f"Cannot convert from '{from_unit}' to '{to_unit}'. "
            "Units must be in the same category."
        )
