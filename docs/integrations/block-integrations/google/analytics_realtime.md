# Google Analytics Realtime
<!-- MANUAL: file_description -->
A block for Google Analytics 4 realtime reports: what is happening on a site or app in the last 30 minutes, such as how many people are active, where they are and what they are looking at. It needs read-only access to Google Analytics (the `analytics.readonly` scope) and at least the Viewer role on the property.
<!-- END MANUAL -->

## Google Analytics Run Realtime Report

### What it is
Run a Google Analytics 4 realtime report of the last 30 minutes of activity (60 on Analytics 360), such as active users by country, device or page. Realtime reports take only these dimensions: appVersion, audienceId, audienceName, audienceResourceName, city, cityId, country, countryId, deviceCategory, eventName, minutesAgo, platform, streamId, streamName, unifiedScreenName, plus user-scoped custom dimensions (customUser:...), and only these metrics: activeUsers, eventCount, keyEvents, screenPageViews.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Google Analytics Data API `properties.runRealtimeReport` endpoint with one minute range, from `minutes_ago` minutes ago up to now. A standard property can look back at most 29 minutes (the last 30 minutes, the default). A Google Analytics 360 property can look back 59. If Google rejects a longer look-back on a standard property, the block's message says it needs 360.

Realtime reports take only realtime fields. The dimensions are `appVersion`, `audienceId`, `audienceName`, `audienceResourceName`, `city`, `cityId`, `country`, `countryId`, `deviceCategory`, `eventName`, `minutesAgo`, `platform`, `streamId`, `streamName` and `unifiedScreenName` (the page title or screen name), plus user-scoped custom dimensions (`customUser:...`). The metrics are `activeUsers`, `eventCount`, `keyEvents` and `screenPageViews`. Event-scoped custom dimensions and custom metrics don't work in realtime reports.

Rows, totals, dimension filters, sorting and the row limit work as in Google Analytics Run Report. Realtime responses have no time zone or currency, so the block doesn't output them. Realtime reports use their own quota, separate from regular reports.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| property_id | The Google Analytics 4 property's numeric ID, such as 123456789 (properties/123456789 works too). Find it in Google Analytics under Admin > Property details, or with Google Analytics List Properties. It isn't the G-... measurement ID. | str | Yes |
| metrics | Realtime metric API names: activeUsers, eventCount, keyEvents, screenPageViews | List[str] | No |
| dimensions | Realtime dimension API names to break the numbers down by, up to 9, such as country, city, deviceCategory, unifiedScreenName, eventName, platform or minutesAgo. Leave empty for a single row of totals. | List[str] | No |
| minutes_ago | How many minutes back to look. 29, the most a standard property allows, covers the last 30 minutes. Google Analytics 360 properties allow up to 59. | int | No |
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

### Possible use case
<!-- MANUAL: use_case -->
**Launch Monitoring**: Check how many people are on the site right after a launch email or post goes out.

**Live Page Check**: See which pages or screens people have open right now, by `unifiedScreenName`.

**Tracking Alert**: Warn the team when active users drop to zero during business hours, which can mean the site or its tracking is down.
<!-- END MANUAL -->

---
