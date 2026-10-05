"""Request building and response reading for the Google Analytics report blocks.

Both report blocks call the Data API: properties.runReport for a date range and
properties.runRealtimeReport for the last minutes. They share the metrics,
dimensions, filters, sort order, row limit and the shape of the response.
"""

import re
from datetime import date
from enum import Enum
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from backend.util.exceptions import BlockInputError

# Google's limits for one report request.
MAX_METRICS = 10
MAX_DIMENSIONS = 9
# Google returns up to 250,000 rows a request; the blocks stop at 100,000.
MAX_ROWS = 100_000

FILTERS_DESCRIPTION = (
    "Only count data whose dimension values pass all of these filters, such as "
    "country exact Germany. A filter can leave out what matches instead."
)
ORDER_BY_DESCRIPTION = (
    "A metric or dimension of the report to sort the rows by. Put - in front to "
    "sort from highest to lowest, such as -activeUsers. Leave empty for Google's "
    "default order."
)
LIMIT_DESCRIPTION = (
    "The most rows to return. row_count says how many rows matched in total."
)
ROWS_DESCRIPTION = (
    "One entry per row, keyed by dimension and metric API names. Dimension values "
    "are text and metric values are numbers."
)
TOTALS_DESCRIPTION = (
    "Each metric's total for the whole report, from Google Analytics. Empty when "
    "nothing matched."
)
ROW_COUNT_DESCRIPTION = (
    "How many rows matched in total, which can be more than were returned"
)

_DAYS_AGO = re.compile(r"([0-9]+)daysago")
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


class DimensionMatchType(str, Enum):
    EXACT = "exact"
    CONTAINS = "contains"
    BEGINS_WITH = "begins_with"
    ENDS_WITH = "ends_with"
    FULL_REGEX = "full_regex"


_MATCH_TYPES: dict[DimensionMatchType, str] = {
    DimensionMatchType.EXACT: "EXACT",
    DimensionMatchType.CONTAINS: "CONTAINS",
    DimensionMatchType.BEGINS_WITH: "BEGINS_WITH",
    DimensionMatchType.ENDS_WITH: "ENDS_WITH",
    DimensionMatchType.FULL_REGEX: "FULL_REGEXP",
}


class GoogleAnalyticsDimensionFilter(BaseModel):
    """A condition that a dimension's value must meet for data to count."""

    dimension: str = Field(
        description=(
            "API name of the dimension to filter on, such as pagePath, country or "
            "sessionSource. It doesn't have to be one of the report's dimensions."
        )
    )
    match_type: DimensionMatchType = Field(
        default=DimensionMatchType.EXACT,
        description=(
            "How to compare: exact, contains, begins_with, ends_with, or full_regex "
            "(the whole value must match the regular expression)"
        ),
    )
    value: str = Field(description="The text or regular expression to match")
    case_sensitive: bool = Field(
        default=False, description="Whether upper and lower case must match too"
    )
    exclude: bool = Field(
        default=False,
        description="Leave out the data that matches, instead of keeping only it",
    )


class AnalyticsReport(BaseModel):
    """A Data API report, read into plain values."""

    rows: list[dict[str, str | int | float]]
    totals: dict[str, int | float]
    row_count: int
    time_zone: Optional[str] = None
    currency_code: Optional[str] = None


def report_body(
    *,
    metrics: list[str],
    dimensions: list[str],
    filters: list[GoogleAnalyticsDimensionFilter],
    order_by: str,
    limit: int,
    block_name: str,
    block_id: str,
) -> dict[str, Any]:
    """The request fields that regular and realtime reports share."""
    metrics, dimensions = api_names(metrics), api_names(dimensions)
    _check_field_counts(metrics, dimensions, block_name, block_id)
    body: dict[str, Any] = {"metrics": [{"name": name} for name in metrics]}
    if dimensions:
        body["dimensions"] = [{"name": name} for name in dimensions]
    if expression := dimension_filter_expression(filters, block_name, block_id):
        body["dimensionFilter"] = expression
    if sort := order_bys(order_by, metrics, dimensions, block_name, block_id):
        body["orderBys"] = sort
    body["limit"] = str(limit)  # an int64, which the API takes as a string
    body["metricAggregations"] = ["TOTAL"]
    return body


def api_names(values: Iterable[str]) -> list[str]:
    """Trim the names, split comma-separated ones, and drop blanks and repeats."""
    names = (name.strip() for value in values for name in value.split(","))
    return list(dict.fromkeys(name for name in names if name))


def report_date(value: str, field: str, block_name: str, block_id: str) -> str:
    """Check a date is in a form the Data API takes, and return it in that form.

    The API resolves today, yesterday and NdaysAgo in the property's time zone.
    """
    text = value.strip()
    lowered = text.lower()
    if lowered in ("today", "yesterday"):
        return lowered
    if match := _DAYS_AGO.fullmatch(lowered):
        return f"{int(match.group(1))}daysAgo"
    if _ISO_DATE.fullmatch(text):
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError:
            pass
    raise BlockInputError(
        message=(
            f"{field} must be a date as YYYY-MM-DD, or today, yesterday or NdaysAgo "
            f"(such as 28daysAgo), not '{value}'."
        ),
        block_name=block_name,
        block_id=block_id,
    )


def dimension_filter_expression(
    filters: list[GoogleAnalyticsDimensionFilter], block_name: str, block_id: str
) -> dict[str, Any] | None:
    """Combine the filters into one Data API FilterExpression; all must hold."""
    expressions = [_filter_expression(item, block_name, block_id) for item in filters]
    if len(expressions) > 1:
        return {"andGroup": {"expressions": expressions}}
    return expressions[0] if expressions else None


def order_bys(
    order_by: str,
    metrics: list[str],
    dimensions: list[str],
    block_name: str,
    block_id: str,
) -> list[dict[str, Any]]:
    """Turn 'name', or '-name' for descending, into a Data API orderBys list."""
    text = order_by.strip()
    if not text:
        return []
    descending = text.startswith("-")
    name = text.removeprefix("-").strip()
    if name in metrics:
        return [{"metric": {"metricName": name}, "desc": descending}]
    if name in dimensions:
        return [{"dimension": {"dimensionName": name}, "desc": descending}]
    raise BlockInputError(
        message=(
            f"Can't sort by '{name}' because it isn't one of the report's metrics or "
            f"dimensions ({', '.join(metrics + dimensions)}). Sort by one of those, "
            "or add it to the report."
        ),
        block_name=block_name,
        block_id=block_id,
    )


def parse_report(response: dict[str, Any]) -> AnalyticsReport:
    """Read a runReport or runRealtimeReport response.

    Google leaves rows and rowCount out when nothing matched, and then sends
    an empty totals entry.
    """
    dimensions = [
        header.get("name", "") for header in response.get("dimensionHeaders", [])
    ]
    metrics = [
        (header.get("name", ""), header.get("type", ""))
        for header in response.get("metricHeaders", [])
    ]
    totals = response.get("totals") or [{}]
    metadata = response.get("metadata", {})
    return AnalyticsReport(
        rows=[_row(row, dimensions, metrics) for row in response.get("rows", [])],
        totals=_metric_values(totals[0], metrics),
        row_count=int(response.get("rowCount", 0)),
        time_zone=metadata.get("timeZone"),
        currency_code=metadata.get("currencyCode"),
    )


def _check_field_counts(
    metrics: list[str], dimensions: list[str], block_name: str, block_id: str
) -> None:
    if not metrics:
        problem = "Add at least one metric, such as activeUsers or sessions."
    elif len(metrics) > MAX_METRICS:
        problem = (
            f"A report can have at most {MAX_METRICS} metrics, not {len(metrics)}."
        )
    elif len(dimensions) > MAX_DIMENSIONS:
        problem = (
            f"A report can have at most {MAX_DIMENSIONS} dimensions, "
            f"not {len(dimensions)}."
        )
    else:
        return
    raise BlockInputError(message=problem, block_name=block_name, block_id=block_id)


def _filter_expression(
    item: GoogleAnalyticsDimensionFilter, block_name: str, block_id: str
) -> dict[str, Any]:
    dimension = item.dimension.strip()
    if not dimension or not item.value:
        raise BlockInputError(
            message="Each dimension filter needs a dimension name and a value to match.",
            block_name=block_name,
            block_id=block_id,
        )
    expression = {
        "filter": {
            "fieldName": dimension,
            "stringFilter": {
                "matchType": _MATCH_TYPES[item.match_type],
                "value": item.value,
                "caseSensitive": item.case_sensitive,
            },
        }
    }
    return {"notExpression": expression} if item.exclude else expression


def _row(
    row: dict[str, Any], dimensions: list[str], metrics: list[tuple[str, str]]
) -> dict[str, str | int | float]:
    """One report row keyed by API name: dimensions as text, metrics as numbers."""
    values: dict[str, str | int | float] = {
        name: value.get("value", "")
        for name, value in zip(dimensions, row.get("dimensionValues", []))
    }
    return values | _metric_values(row, metrics)


def _metric_values(
    row: dict[str, Any], metrics: list[tuple[str, str]]
) -> dict[str, int | float]:
    """Metric values arrive as strings; whole integer metrics become ints, the
    rest floats."""
    return {
        name: _number(value.get("value", "0"), metric_type)
        for (name, metric_type), value in zip(metrics, row.get("metricValues", []))
    }


def _number(text: str, metric_type: str) -> int | float:
    # Google types each metric, not each value, so an integer metric sent as
    # "12.0" or as a fraction must not fail the whole report.
    value = float(text)
    return int(value) if metric_type == "TYPE_INTEGER" and value.is_integer() else value
