# Google Analytics Reports
<!-- MANUAL: file_description -->
A block for running Google Analytics 4 reports: numbers such as users, sessions, page views, key events or revenue over a date range, optionally split by dimensions such as date, page, country or traffic source. It needs read-only access to Google Analytics (the `analytics.readonly` scope) and at least the Viewer role on the property.
<!-- END MANUAL -->

## Google Analytics Run Report

### What it is
Run a Google Analytics 4 report of metrics such as users, sessions or page views over a date range, split by dimensions such as date, page or traffic source. It can filter on dimension values, sort and limit the rows, and returns totals too. Google Analytics List Dimensions and Metrics gives a property's custom names.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Google Analytics Data API `properties.runReport` endpoint for one date range. Dates can be `YYYY-MM-DD`, `today`, `yesterday` or `NdaysAgo` (such as `28daysAgo`), and Google works out relative dates in the property's time zone, which the block returns as `time_zone`. Google Analytics can take a day or two to finish processing data, so the latest days may still change.

A report has 1 to 10 metrics and up to 9 dimensions, given by API name. Each row is keyed by those names: dimension values are text (the `date` dimension comes as `YYYYMMDD`) and metric values are numbers, whole numbers for counts. Money is in `currency_code`, and standard durations such as `averageSessionDuration` are in seconds. Rows where every metric is zero are left out, as in Google Analytics. All the dimension filters must match; each compares one dimension exactly, by contains, begins with, ends with or a full regular expression, and can leave out what it matches instead. `order_by` sorts by one of the report's metrics or dimensions, with `-` in front for descending.

The block always asks Google for totals, so `totals` holds each metric's total for the whole report. Don't add up user counts across rows yourself: one person can appear in several rows. `row_count` is how many rows matched, which can be more than `limit` (at most 100,000 rows per run).

Each report uses some of the property's Data API quota: a standard property gets 200,000 tokens a day and 40,000 an hour, and one app such as AutoGPT can use 35% of the hourly tokens. Most reports cost under 10 tokens. When a quota runs out, the block says so with Google's message; try again later. A name Google doesn't know, or dimensions and metrics that can't be used together, fail with Google's message, which names the field. Google Analytics List Dimensions and Metrics lists the names a property supports.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| property_id | The Google Analytics 4 property's numeric ID, such as 123456789 (properties/123456789 works too). Find it in Google Analytics under Admin > Property details, or with Google Analytics List Properties. It isn't the G-... measurement ID. | str | Yes |
| metrics | Metric API names, 1 to 10, such as activeUsers, sessions, screenPageViews, keyEvents, engagementRate or totalRevenue. Custom metrics look like customEvent:name. | List[str] | No |
| dimensions | Dimension API names to break the numbers down by, up to 9, such as date, country, sessionSource, sessionDefaultChannelGroup or pagePath. Leave empty for a single row of totals. | List[str] | No |
| start_date | First day of the report: YYYY-MM-DD, today, yesterday or NdaysAgo (such as 28daysAgo). Relative dates follow the property's time zone. | str | No |
| end_date | Last day of the report, included: YYYY-MM-DD, today, yesterday or NdaysAgo | str | No |
| dimension_filters | Only count data whose dimension values pass all of these filters, such as country exact Germany. A filter can leave out what matches instead. | List[GoogleAnalyticsDimensionFilter] | No |
| order_by | A metric or dimension of the report to sort the rows by. Put - in front to sort from highest to lowest, such as -activeUsers. Leave empty for Google's default order. | str | No |
| limit | The most rows to return. row_count says how many rows matched in total. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| rows | One entry per row, keyed by dimension and metric API names. Dimension values are text and metric values are numbers. | List[Dict[str, str \| int \| float]] |
| row | Each row | Dict[str, str \| int \| float] |
| totals | Each metric's total for the whole report, from Google Analytics. Empty when nothing matched. | Dict[str, int \| float] |
| row_count | How many rows matched in total, which can be more than were returned | int |
| time_zone | The property's time zone, which the report's dates are in | str |
| currency_code | The currency of money metrics such as totalRevenue | str |

### Possible use case
<!-- MANUAL: use_case -->
**Weekly Traffic Summary**: Every Monday, report last week's users and sessions by `sessionDefaultChannelGroup` and email the totals to the team.

**Top Blog Posts**: Find the 10 most viewed blog pages of the last 28 days with a `pagePath` begins_with `/blog/` filter and `order_by` set to `-screenPageViews`.

**Campaign Check**: Compare sessions and key events by `sessionSource` after a launch to see which channels brought visitors who converted.
<!-- END MANUAL -->

---
