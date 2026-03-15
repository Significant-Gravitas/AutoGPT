# Exa Websets Enrichment
<!-- MANUAL: file_description -->
Blocks for enriching webset items with additional data using Exa's enrichment API.
<!-- END MANUAL -->

## Exa Cancel Enrichment

### What it is
Cancel a running enrichment operation

### How it works
<!-- MANUAL: how_it_works -->
This block stops a running enrichment operation on a webset. Items already enriched before cancellation retain their enrichment data, but remaining items will not be processed.

Use this when an enrichment is taking too long, producing unexpected results, or is no longer needed. The block returns the approximate number of items enriched before cancellation.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| enrichment_id | The ID of the enrichment to cancel | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| enrichment_id | The ID of the canceled enrichment | str |
| status | Status after cancellation | str |
| items_enriched_before_cancel | Approximate number of items enriched before cancellation | int |
| success | Whether the cancellation was successful | str |

### Possible use case
<!-- MANUAL: use_case -->
**Cost Control**: Stop enrichments that are exceeding budget or taking too long.

**Error Handling**: Cancel enrichments producing incorrect results to fix configuration.

**Priority Changes**: Stop lower-priority enrichments to free resources for urgent tasks.
<!-- END MANUAL -->

---

## Exa Create Enrichment

### What it is
Create enrichments to extract additional structured data from webset items

### How it works
<!-- MANUAL: how_it_works -->
This block creates an enrichment task that extracts specific data from each webset item using AI. You define what to extract via a description, and the enrichment runs against all current and future items in the webset.

Enrichments support various output formats including text, dates, numbers, and predefined options. You can apply enrichments to existing items immediately or configure them to run only on new items.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| description | What data to extract from each item | str | Yes |
| title | Short title for this enrichment (auto-generated if not provided) | str | No |
| format | Expected format of the extracted data | "text" \| "date" \| "number" \| "options" \| "email" \| "phone" | No |
| options | Available options when format is 'options' | List[str] | No |
| apply_to_existing | Apply this enrichment to existing items in the webset | bool | No |
| metadata | Metadata to attach to the enrichment | Dict[str, Any] | No |
| wait_for_completion | Wait for the enrichment to complete on existing items | bool | No |
| polling_timeout | Maximum time to wait for completion in seconds | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| enrichment_id | The unique identifier for the created enrichment | str |
| webset_id | The webset this enrichment belongs to | str |
| status | Current status of the enrichment | str |
| title | Title of the enrichment | str |
| description | Description of what data is extracted | str |
| format | Format of the extracted data | str |
| instructions | Generated instructions for the enrichment | str |
| items_enriched | Number of items enriched (if wait_for_completion was True) | int |
| completion_time | Time taken to complete in seconds (if wait_for_completion was True) | float |

### Possible use case
<!-- MANUAL: use_case -->
**Data Extraction**: Extract specific fields like founding dates, employee counts, or contact info from company profiles.

**Classification**: Categorize items into predefined buckets using the options format.

**Sentiment Analysis**: Analyze sentiment or tone from article content or reviews.
<!-- END MANUAL -->

---

## Exa Delete Enrichment

### What it is
Delete an enrichment from a webset

### How it works
<!-- MANUAL: how_it_works -->
This block removes an enrichment configuration from a webset. The enrichment will no longer be applied to new items, but existing enrichment data on items is not deleted.

Use this to clean up enrichments that are no longer needed or to remove misconfigured enrichments before creating corrected ones.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| enrichment_id | The ID of the enrichment to delete | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| enrichment_id | The ID of the deleted enrichment | str |
| success | Whether the deletion was successful | str |

### Possible use case
<!-- MANUAL: use_case -->
**Configuration Cleanup**: Remove enrichments that are no longer relevant to your data needs.

**Reconfiguration**: Delete misconfigured enrichments before creating corrected replacements.

**Cost Optimization**: Remove unnecessary enrichments to reduce processing costs on new items.
<!-- END MANUAL -->

---

## Exa Get Enrichment

### What it is
Get the status and details of a webset enrichment

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves detailed information about a specific enrichment including its configuration, current status, and processing progress.

Use this to monitor enrichment progress, verify configuration, or troubleshoot issues with enrichment results. Returns the full enrichment specification along with timestamps.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| enrichment_id | The ID of the enrichment to retrieve | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| enrichment_id | The unique identifier for the enrichment | str |
| status | Current status of the enrichment | str |
| title | Title of the enrichment | str |
| description | Description of what data is extracted | str |
| format | Format of the extracted data | str |
| options | Available options (for 'options' format) | List[str] |
| instructions | Generated instructions for the enrichment | str |
| created_at | When the enrichment was created | str |
| updated_at | When the enrichment was last updated | str |
| metadata | Metadata attached to the enrichment | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Progress Monitoring**: Check enrichment status to monitor completion of large batch operations.

**Configuration Verification**: Retrieve enrichment details to verify settings before making changes.

**Debugging**: Investigate enrichment configuration when results don't match expectations.
<!-- END MANUAL -->

---

## Exa Update Enrichment

### What it is
Update an existing enrichment configuration

### How it works
<!-- MANUAL: how_it_works -->
This block modifies an existing enrichment's configuration. You can update the description, output format, available options, or metadata without recreating the enrichment.

Changes apply to future items; existing enrichment data is not reprocessed unless you explicitly re-run the enrichment on existing items.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| enrichment_id | The ID of the enrichment to update | str | Yes |
| description | New description for what data to extract | str | No |
| format | New format for the extracted data | "text" \| "date" \| "number" \| "options" \| "email" \| "phone" | No |
| options | New options when format is 'options' | List[str] | No |
| metadata | New metadata to attach to the enrichment | Dict[str, Any] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| enrichment_id | The unique identifier for the enrichment | str |
| status | Current status of the enrichment | str |
| title | Title of the enrichment | str |
| description | Updated description | str |
| format | Updated format | str |
| success | Whether the update was successful | str |

### Possible use case
<!-- MANUAL: use_case -->
**Refinement**: Improve enrichment descriptions based on initial results to get better extractions.

**Option Updates**: Add or modify options for classification enrichments as needs evolve.

**Metadata Management**: Update enrichment metadata for organization or tracking purposes.
<!-- END MANUAL -->

---
