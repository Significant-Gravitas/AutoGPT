"""Unit tests for building Google Analytics report requests and reading reports."""

import json

import pytest
from googleapiclient.discovery_cache import get_static_doc

from backend.blocks.google._analytics_report import (
    AnalyticsReport,
    DimensionMatchType,
    GoogleAnalyticsDimensionFilter,
    api_names,
    dimension_filter_expression,
    order_bys,
    parse_report,
    report_body,
    report_date,
)
from backend.util.exceptions import BlockInputError

DATA_API_SCHEMAS = json.loads(get_static_doc("analyticsdata", "v1beta") or "{}")[
    "schemas"
]


def _body(**overrides) -> dict:
    fields = {
        "metrics": ["sessions"],
        "dimensions": [],
        "filters": [],
        "order_by": "",
        "limit": 100,
        "block_name": "block",
        "block_id": "id",
    } | overrides
    return report_body(**fields)


@pytest.mark.parametrize(
    "values, expected",
    [
        (["activeUsers", " sessions ", "activeUsers"], ["activeUsers", "sessions"]),
        (["activeUsers, sessions", "date"], ["activeUsers", "sessions", "date"]),
        (["", " , ", "customEvent:plan"], ["customEvent:plan"]),
        ([], []),
    ],
)
def test_api_names_trims_splits_and_drops_repeats(values, expected):
    assert api_names(values) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2026-09-01", "2026-09-01"),
        (" today ", "today"),
        ("Yesterday", "yesterday"),
        ("28daysAgo", "28daysAgo"),
        ("7DaysAgo", "7daysAgo"),
        ("007daysago", "7daysAgo"),
        ("0daysAgo", "0daysAgo"),
    ],
)
def test_report_date_passes_the_apis_own_forms_through(value: str, expected: str):
    assert report_date(value, "start_date", "block", "id") == expected


@pytest.mark.parametrize(
    "value",
    ["2026-02-30", "2026/09/01", "20260901", "2026-9-1", "last week", "", "-7daysAgo"],
)
def test_report_date_rejects_other_forms(value: str):
    with pytest.raises(
        BlockInputError, match="start_date must be a date as YYYY-MM-DD"
    ):
        report_date(value, "start_date", "block", "id")


def test_report_body_has_every_part_of_the_request():
    filters = [
        GoogleAnalyticsDimensionFilter(
            dimension=" pagePath ",
            match_type=DimensionMatchType.BEGINS_WITH,
            value="/blog/",
        ),
        GoogleAnalyticsDimensionFilter(
            dimension="country", value="Germany", case_sensitive=True, exclude=True
        ),
    ]
    body = _body(
        metrics=["activeUsers, sessions"],
        dimensions=["date", "date"],
        filters=filters,
        order_by="-sessions",
        limit=50,
    )
    assert body == {
        "metrics": [{"name": "activeUsers"}, {"name": "sessions"}],
        "dimensions": [{"name": "date"}],
        "dimensionFilter": {
            "andGroup": {
                "expressions": [
                    {
                        "filter": {
                            "fieldName": "pagePath",
                            "stringFilter": {
                                "matchType": "BEGINS_WITH",
                                "value": "/blog/",
                                "caseSensitive": False,
                            },
                        }
                    },
                    {
                        "notExpression": {
                            "filter": {
                                "fieldName": "country",
                                "stringFilter": {
                                    "matchType": "EXACT",
                                    "value": "Germany",
                                    "caseSensitive": True,
                                },
                            }
                        }
                    },
                ]
            }
        },
        "orderBys": [{"metric": {"metricName": "sessions"}, "desc": True}],
        "limit": "50",
        "metricAggregations": ["TOTAL"],
    }


def test_report_body_leaves_out_the_optional_parts():
    assert _body() == {
        "metrics": [{"name": "sessions"}],
        "limit": "100",
        "metricAggregations": ["TOTAL"],
    }


@pytest.mark.parametrize(
    "fields, problem",
    [
        ({"metrics": []}, "Add at least one metric"),
        ({"metrics": [" ", ","]}, "Add at least one metric"),
        ({"metrics": [f"m{i}" for i in range(11)]}, "at most 10 metrics, not 11"),
        ({"dimensions": [f"d{i}" for i in range(10)]}, "at most 9 dimensions, not 10"),
    ],
)
def test_report_body_checks_googles_limits(fields: dict, problem: str):
    with pytest.raises(BlockInputError, match=problem):
        _body(**fields)


def test_report_body_allows_googles_maximums():
    body = _body(
        metrics=[f"m{i}" for i in range(10)], dimensions=[f"d{i}" for i in range(9)]
    )
    assert (len(body["metrics"]), len(body["dimensions"])) == (10, 9)


@pytest.mark.parametrize(
    "match_type, api_value",
    [
        (DimensionMatchType.EXACT, "EXACT"),
        (DimensionMatchType.CONTAINS, "CONTAINS"),
        (DimensionMatchType.BEGINS_WITH, "BEGINS_WITH"),
        (DimensionMatchType.ENDS_WITH, "ENDS_WITH"),
        (DimensionMatchType.FULL_REGEX, "FULL_REGEXP"),
    ],
)
def test_match_types_map_to_the_apis_string_filter(match_type, api_value):
    expression = dimension_filter_expression(
        [
            GoogleAnalyticsDimensionFilter(
                dimension="pagePath", match_type=match_type, value="x"
            )
        ],
        "block",
        "id",
    )
    assert expression == {
        "filter": {
            "fieldName": "pagePath",
            "stringFilter": {
                "matchType": api_value,
                "value": "x",
                "caseSensitive": False,
            },
        }
    }
    allowed = DATA_API_SCHEMAS["StringFilter"]["properties"]["matchType"]["enum"]
    assert api_value in allowed


def test_one_excluded_filter_is_a_not_expression_and_none_is_no_filter():
    excluded = GoogleAnalyticsDimensionFilter(
        dimension="sessionSource", value="(direct)", exclude=True
    )
    assert dimension_filter_expression([excluded], "block", "id") == {
        "notExpression": {
            "filter": {
                "fieldName": "sessionSource",
                "stringFilter": {
                    "matchType": "EXACT",
                    "value": "(direct)",
                    "caseSensitive": False,
                },
            }
        }
    }
    assert dimension_filter_expression([], "block", "id") is None


@pytest.mark.parametrize("dimension, value", [("", "x"), ("  ", "x"), ("pagePath", "")])
def test_filters_need_a_dimension_and_a_value(dimension: str, value: str):
    item = GoogleAnalyticsDimensionFilter(dimension=dimension, value=value)
    with pytest.raises(BlockInputError, match="needs a dimension name and a value"):
        dimension_filter_expression([item], "block", "id")


@pytest.mark.parametrize(
    "order_by, expected",
    [
        ("-sessions", [{"metric": {"metricName": "sessions"}, "desc": True}]),
        (" sessions ", [{"metric": {"metricName": "sessions"}, "desc": False}]),
        ("date", [{"dimension": {"dimensionName": "date"}, "desc": False}]),
        ("- date", [{"dimension": {"dimensionName": "date"}, "desc": True}]),
        ("", []),
        ("   ", []),
    ],
)
def test_order_bys(order_by: str, expected: list):
    assert order_bys(order_by, ["sessions"], ["date"], "block", "id") == expected


def test_order_by_must_be_in_the_report():
    with pytest.raises(BlockInputError, match=r"'country'.*\(sessions, date\)"):
        order_bys("-country", ["sessions"], ["date"], "block", "id")


def test_parse_report_types_values_by_metric_type():
    report = parse_report(
        {
            "dimensionHeaders": [{"name": "date"}, {"name": "pagePath"}],
            "metricHeaders": [
                {"name": "sessions", "type": "TYPE_INTEGER"},
                {"name": "engagementRate", "type": "TYPE_FLOAT"},
                {"name": "averageSessionDuration", "type": "TYPE_SECONDS"},
                {"name": "totalRevenue", "type": "TYPE_CURRENCY"},
            ],
            "rows": [
                {
                    "dimensionValues": [{"value": "20260901"}, {}],
                    "metricValues": [
                        {"value": "120"},
                        {"value": "0.5"},
                        {"value": "65.25"},
                        {"value": "1999"},
                    ],
                }
            ],
            "totals": [
                {
                    "dimensionValues": [
                        {"value": "RESERVED_TOTAL"},
                        {"value": "RESERVED_TOTAL"},
                    ],
                    "metricValues": [
                        {"value": "300"},
                        {"value": "0.4"},
                        {"value": "61"},
                        {"value": "4250.5"},
                    ],
                }
            ],
            "rowCount": 31,
            "metadata": {"currencyCode": "EUR", "timeZone": "Europe/Berlin"},
        }
    )
    assert report == AnalyticsReport(
        rows=[
            {
                "date": "20260901",
                "pagePath": "",
                "sessions": 120,
                "engagementRate": 0.5,
                "averageSessionDuration": 65.25,
                "totalRevenue": 1999.0,
            }
        ],
        totals={
            "sessions": 300,
            "engagementRate": 0.4,
            "averageSessionDuration": 61.0,
            "totalRevenue": 4250.5,
        },
        row_count=31,
        time_zone="Europe/Berlin",
        currency_code="EUR",
    )
    row = report.rows[0]
    assert type(row["sessions"]) is int
    assert type(row["totalRevenue"]) is float


def test_parse_report_without_dimensions_has_one_row_and_its_totals():
    report = parse_report(
        {
            "metricHeaders": [{"name": "activeUsers", "type": "TYPE_INTEGER"}],
            "rows": [{"metricValues": [{"value": "880"}]}],
            "totals": [{"metricValues": [{"value": "880"}]}],
            "rowCount": 1,
        }
    )
    assert report.rows == [{"activeUsers": 880}]
    assert report.totals == {"activeUsers": 880}
    assert (report.time_zone, report.currency_code) == (None, None)


def test_parse_report_reads_an_empty_report():
    # Google leaves out rows and rowCount and sends an empty totals entry.
    report = parse_report(
        {
            "dimensionHeaders": [{"name": "country"}],
            "metricHeaders": [{"name": "sessions", "type": "TYPE_INTEGER"}],
            "totals": [{}],
            "metadata": {"currencyCode": "USD", "timeZone": "America/New_York"},
            "kind": "analyticsData#runReport",
        }
    )
    assert report == AnalyticsReport(
        rows=[],
        totals={},
        row_count=0,
        time_zone="America/New_York",
        currency_code="USD",
    )
    assert parse_report({}) == AnalyticsReport(rows=[], totals={}, row_count=0)


def test_integer_metric_type_name_matches_the_api():
    types = DATA_API_SCHEMAS["MetricHeader"]["properties"]["type"]["enum"]
    assert "TYPE_INTEGER" in types
