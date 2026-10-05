# Google Analytics
<!-- MANUAL: file_description -->
Blocks for finding the Google Analytics 4 properties the connected Google account can read, and the dimensions and metrics each property can report on. They give you the property ID and the API names that Google Analytics Run Report and Google Analytics Run Realtime Report take. Both blocks need read-only access to Google Analytics (the `analytics.readonly` scope), and the account needs at least the Viewer role on a property to see it.
<!-- END MANUAL -->

## Google Analytics List Dimensions And Metrics

### What it is
List the dimensions and metrics a Google Analytics 4 property can report on, with the API names the report blocks take. By default it lists only the property's custom ones; turn off custom_only to include the standard ones too.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Google Analytics Data API `properties.getMetadata` endpoint for the property. Google returns every dimension and metric the property can report on: several hundred standard ones that every property has, plus the custom dimensions and metrics registered on this property. By default the block returns only the custom ones. Turn off `custom_only` to get the standard ones too.

Each entry has the API name to use in the report blocks (such as `country`, `customEvent:plan` or `customUser:tier`), the name shown in Google Analytics, a description and a category. Metrics also have a `type` that says what their values are, such as `integer`, `seconds` or `currency`. A custom dimension only appears once it is registered under Admin > Custom definitions; event parameters that aren't registered can't be reported on. Realtime reports take only user-scoped custom dimensions (`customUser:...`) and no custom metrics.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| property_id | The Google Analytics 4 property's numeric ID, such as 123456789 (properties/123456789 works too). Find it in Google Analytics under Admin > Property details, or with Google Analytics List Properties. It isn't the G-... measurement ID. | str | Yes |
| custom_only | List only the property's custom dimensions and metrics. Turn off to list the standard ones too, several hundred in all. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| dimensions | Dimensions the property's reports can be split by (only its custom ones unless custom_only is off) | List[GoogleAnalyticsDimension] |
| metrics | Metrics the property's reports can show (only its custom ones unless custom_only is off) | List[GoogleAnalyticsMetric] |

### Possible use case
<!-- MANUAL: use_case -->
**Report on Custom Events**: Find the API name of a custom dimension, such as `customEvent:plan`, and split a report by it.

**Safer Agent Reports**: Let an agent check which names a property supports before it asks for a report.

**Tracking Audit**: List a property's custom definitions to review what the site sends to Google Analytics.
<!-- END MANUAL -->

---

## Google Analytics List Properties

### What it is
List the Google Analytics 4 properties the connected Google account can read, with their numeric property IDs and account names. The other Google Analytics blocks take one of these property IDs.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Google Analytics Admin API `accountSummaries.list` endpoint, 200 accounts at a time, and follows the pages until it has them all. Each account summary lists the properties in that account, and the block returns one entry per property: its numeric `property_id` (what the other Google Analytics blocks take), its resource name (`properties/...`), its name and type (`ordinary`, `subproperty` or `rollup`), and the ID and name of its account.

The list holds every property the account has a role on, Viewer or higher. A Google account with no Google Analytics access gets an empty list. Only Google Analytics 4 properties appear, since Universal Analytics was shut down in July 2024.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| properties | The properties the account can read, account by account | List[GoogleAnalyticsProperty] |
| property | Each property | GoogleAnalyticsProperty |

### Possible use case
<!-- MANUAL: use_case -->
**Find the Property ID**: Look up the numeric ID of the "example.com" property before running a report on it.

**Agency Overview**: Run the same weekly report on every client property the account can read.

**Access Check**: Confirm the connected Google account can read a property before an agent relies on it.
<!-- END MANUAL -->

---
