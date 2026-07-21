import csv
import io
import json
import logging
from typing import Any, Iterator, Literal, Optional

from pydantic import BaseModel, Field

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider, DirectiveProvider
from forge.command import Command, command
from forge.models.json_schema import JSONSchema
from forge.utils.exceptions import DataProcessingError

logger = logging.getLogger(__name__)


class DataProcessorConfiguration(BaseModel):
    max_json_depth: int = Field(
        default=10, description="Maximum nesting depth for JSON parsing"
    )
    max_csv_rows: int = Field(
        default=10000, description="Maximum rows to process in CSV operations"
    )


class DataProcessorComponent(
    DirectiveProvider,
    CommandProvider,
    ConfigurableComponent[DataProcessorConfiguration],
):
    """Provides commands to parse, transform, and query structured data."""

    config_class = DataProcessorConfiguration

    def __init__(self, config: Optional[DataProcessorConfiguration] = None):
        ConfigurableComponent.__init__(self, config)

    def get_resources(self) -> Iterator[str]:
        yield "Ability to parse and manipulate JSON and CSV data."

    def get_commands(self) -> Iterator[Command]:
        yield self.parse_json
        yield self.format_json
        yield self.query_json
        yield self.parse_csv
        yield self.filter_csv
        yield self.aggregate_csv

    @command(
        ["parse_json", "validate_json"],
        "Parse and validate a JSON string, returning a structured representation.",
        {
            "json_string": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The JSON string to parse",
                required=True,
            ),
        },
    )
    def parse_json(self, json_string: str) -> str:
        """Parse and validate a JSON string.

        Args:
            json_string: The JSON string to parse

        Returns:
            str: Parsed JSON as formatted string with type information
        """
        try:
            data = json.loads(json_string)

            # Provide type information
            result = {
                "valid": True,
                "type": type(data).__name__,
                "data": data,
            }

            if isinstance(data, list):
                result["length"] = len(data)
            elif isinstance(data, dict):
                result["keys"] = list(data.keys())

            return json.dumps(result, indent=2)

        except json.JSONDecodeError as e:
            return json.dumps(
                {
                    "valid": False,
                    "error": str(e),
                    "line": e.lineno,
                    "column": e.colno,
                },
                indent=2,
            )

    @command(
        ["format_json", "pretty_print_json"],
        "Format JSON with proper indentation for readability.",
        {
            "json_string": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The JSON string to format",
                required=True,
            ),
            "indent": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="Number of spaces for indentation (default: 2)",
                minimum=0,
                maximum=8,
                required=False,
            ),
        },
    )
    def format_json(self, json_string: str, indent: int = 2) -> str:
        """Format JSON with proper indentation.

        Args:
            json_string: The JSON string to format
            indent: Number of spaces for indentation

        Returns:
            str: Formatted JSON string
        """
        try:
            data = json.loads(json_string)
            return json.dumps(data, indent=indent, ensure_ascii=False)
        except json.JSONDecodeError as e:
            raise DataProcessingError(f"Invalid JSON: {e}")

    def _query_path(self, data: Any, path: str) -> Any:
        """Query JSON data using a dot-notation path with array support.

        Args:
            data: The data to query
            path: Path like "users[0].name" or "config.settings.enabled"

        Returns:
            The value at the path
        """
        import re

        if not path:
            return data

        # Split path into segments, handling array notation
        segments = []
        for part in path.split("."):
            # Handle array notation like "users[0]"
            array_match = re.match(r"^(\w+)\[(\d+)\]$", part)
            if array_match:
                segments.append(array_match.group(1))
                segments.append(int(array_match.group(2)))
            elif part.isdigit():
                segments.append(int(part))
            else:
                segments.append(part)

        result = data
        for segment in segments:
            try:
                if isinstance(segment, int):
                    result = result[segment]
                elif isinstance(result, dict):
                    result = result[segment]
                elif isinstance(result, list) and segment.isdigit():
                    result = result[int(segment)]
                else:
                    raise DataProcessingError(
                        f"Cannot access '{segment}' on {type(result).__name__}"
                    )
            except (KeyError, IndexError, TypeError) as e:
                raise DataProcessingError(f"Path query failed at '{segment}': {e}")

        return result

    @command(
        ["query_json", "json_path"],
        "Query JSON data using a dot-notation path (e.g., 'users[0].name').",
        {
            "json_string": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The JSON string to query",
                required=True,
            ),
            "path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Path to query (e.g., 'data.users[0].email')",
                required=True,
            ),
        },
    )
    def query_json(self, json_string: str, path: str) -> str:
        """Query JSON using dot-notation path.

        Args:
            json_string: The JSON string to query
            path: The path to query

        Returns:
            str: The value at the path as JSON
        """
        try:
            data = json.loads(json_string)
            result = self._query_path(data, path)
            return json.dumps(result, indent=2)
        except json.JSONDecodeError as e:
            raise DataProcessingError(f"Invalid JSON: {e}")

    @command(
        ["parse_csv", "csv_to_json"],
        "Parse CSV string into JSON array of objects.",
        {
            "csv_string": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The CSV string to parse",
                required=True,
            ),
            "has_header": JSONSchema(
                type=JSONSchema.Type.BOOLEAN,
                description="Whether the first row is a header (default: True)",
                required=False,
            ),
            "delimiter": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Field delimiter (default: ',')",
                required=False,
            ),
        },
    )
    def parse_csv(
        self, csv_string: str, has_header: bool = True, delimiter: str = ","
    ) -> str:
        """Parse CSV string into JSON.

        Args:
            csv_string: The CSV string to parse
            has_header: Whether first row is header
            delimiter: Field delimiter

        Returns:
            str: JSON array of objects or arrays
        """
        try:
            reader = csv.reader(io.StringIO(csv_string), delimiter=delimiter)
            rows = list(reader)

            if len(rows) > self.config.max_csv_rows:
                raise DataProcessingError(
                    f"CSV exceeds maximum of {self.config.max_csv_rows} rows"
                )

            if not rows:
                return json.dumps([])

            if has_header:
                headers = rows[0]
                data = [dict(zip(headers, row)) for row in rows[1:]]
            else:
                data = rows

            return json.dumps(data, indent=2)

        except csv.Error as e:
            raise DataProcessingError(f"CSV parsing error: {e}")

    @command(
        ["filter_csv", "csv_filter"],
        "Filter CSV rows based on a column condition.",
        {
            "csv_string": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The CSV string to filter",
                required=True,
            ),
            "column": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Column name or index to filter on",
                required=True,
            ),
            "operator": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Comparison operator (eq, ne, gt, lt, gte, lte, contains)",
                required=True,
            ),
            "value": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Value to compare against",
                required=True,
            ),
        },
    )
    def filter_csv(
        self,
        csv_string: str,
        column: str,
        operator: Literal["eq", "ne", "gt", "lt", "gte", "lte", "contains"],
        value: str,
    ) -> str:
        """Filter CSV rows based on a column condition.

        Args:
            csv_string: The CSV string to filter
            column: Column name or index
            operator: Comparison operator
            value: Value to compare against

        Returns:
            str: Filtered CSV as JSON
        """
        # Parse CSV
        data = json.loads(self.parse_csv(csv_string))

        if not data:
            return json.dumps([])

        def compare(row_value: Any, op: str, comp_value: str) -> bool:
            # Try numeric comparison
            try:
                row_num = float(row_value)
                comp_num = float(comp_value)
                if op == "eq":
                    return row_num == comp_num
                elif op == "ne":
                    return row_num != comp_num
                elif op == "gt":
                    return row_num > comp_num
                elif op == "lt":
                    return row_num < comp_num
                elif op == "gte":
                    return row_num >= comp_num
                elif op == "lte":
                    return row_num <= comp_num
            except (ValueError, TypeError):
                pass

            # String comparison
            row_str = str(row_value).lower()
            comp_str = comp_value.lower()

            if op == "eq":
                return row_str == comp_str
            elif op == "ne":
                return row_str != comp_str
            elif op == "contains":
                return comp_str in row_str
            elif op in ("gt", "lt", "gte", "lte"):
                # String comparison for non-numeric
                if op == "gt":
                    return row_str > comp_str
                elif op == "lt":
                    return row_str < comp_str
                elif op == "gte":
                    return row_str >= comp_str
                elif op == "lte":
                    return row_str <= comp_str

            return False

        filtered = []
        for row in data:
            if isinstance(row, dict):
                if column in row:
                    if compare(row[column], operator, value):
                        filtered.append(row)
            elif isinstance(row, list):
                try:
                    col_idx = int(column)
                    if col_idx < len(row):
                        if compare(row[col_idx], operator, value):
                            filtered.append(row)
                except ValueError:
                    pass

        return json.dumps(filtered, indent=2)

    @command(
        ["aggregate_csv", "csv_aggregate"],
        "Aggregate data in a CSV column (sum, avg, min, max, count).",
        {
            "csv_string": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The CSV string to aggregate",
                required=True,
            ),
            "column": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Column name to aggregate",
                required=True,
            ),
            "operation": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Aggregation operation (sum, avg, min, max, count)",
                required=True,
            ),
            "group_by": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Optional column to group by",
                required=False,
            ),
        },
    )
    def aggregate_csv(
        self,
        csv_string: str,
        column: str,
        operation: Literal["sum", "avg", "min", "max", "count"],
        group_by: str | None = None,
    ) -> str:
        """Aggregate data in a CSV column.

        Args:
            csv_string: The CSV string to aggregate
            column: Column name to aggregate
            operation: Aggregation operation
            group_by: Optional grouping column

        Returns:
            str: Aggregation result as JSON
        """
        data = json.loads(self.parse_csv(csv_string))

        if not data:
            return json.dumps({"result": None, "error": "No data"})

        def aggregate(values: list) -> float | int | None:
            # Filter to numeric values
            numeric = []
            for v in values:
                try:
                    numeric.append(float(v))
                except (ValueError, TypeError):
                    continue

            if not numeric:
                if operation == "count":
                    return len(values)
                return None

            if operation == "sum":
                return sum(numeric)
            elif operation == "avg":
                return sum(numeric) / len(numeric)
            elif operation == "min":
                return min(numeric)
            elif operation == "max":
                return max(numeric)
            elif operation == "count":
                return len(values)
            return None

        if group_by:
            # Group by operation
            groups: dict[str, list] = {}
            for row in data:
                if isinstance(row, dict):
                    key = str(row.get(group_by, ""))
                    value = row.get(column)
                    if key not in groups:
                        groups[key] = []
                    groups[key].append(value)

            result = {key: aggregate(values) for key, values in groups.items()}
            return json.dumps({"grouped_by": group_by, "results": result}, indent=2)
        else:
            # Simple aggregation
            values = []
            for row in data:
                if isinstance(row, dict):
                    values.append(row.get(column))

            return json.dumps(
                {"column": column, "operation": operation, "result": aggregate(values)},
                indent=2,
            )
