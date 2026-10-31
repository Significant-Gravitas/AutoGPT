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

After cancelling, the block counts how many of the webset's first 100 items have a result for this enrichment, so `items_enriched_before_cancel` undercounts on larger websets. It charges only if Exa's response reports a cost (100 credits per USD).
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
This block creates an enrichment that has Exa's AI extract one value per webset item from your description, in the chosen format (text, date, number, options, email, phone or url). Exa runs it on all existing items and on items added later. Options (1-150 labels) are only sent when the format is 'options', and more than 150 fails before calling Exa. The `title` input is not part of Exa's current API; Exa generates the title from the description and format.

With `wait_for_completion` and `apply_to_existing` both on, the block polls the enrichment (every 5 seconds, backing off to 30) until it is completed, canceled or failed, for up to `polling_timeout` seconds (default 300, max 600). `items_enriched` is then an estimate taken from the number of items the webset's searches found. On timeout the block returns the status from creation and `items_enriched` 0 instead of failing. `apply_to_existing` only controls this waiting and cannot stop Exa enriching existing items. It charges only if Exa's response reports a cost (100 credits per USD).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| description | What data to extract from each item (up to 5,000 characters) | str | Yes |
| title | Not part of Exa's current API: Exa generates the title from the description and format. Sent as-is if set. | str | No |
| format | Expected format of the extracted data: text, date, number, options (pick from a list), email, phone or url | "text" \| "date" \| "number" \| "options" \| "email" \| "phone" \| "url" | No |
| options | Labels to choose from when format is 'options' (1-150) | List[str] | No |
| apply_to_existing | Exa always enriches existing and future items; this only controls whether 'wait for completion' waits for them | bool | No |
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
This block permanently removes an enrichment from a webset, so it is no longer applied to items. The deletion cannot be undone.

Use this to clean up enrichments that are no longer needed or to remove misconfigured enrichments before creating corrected ones. It charges only if Exa's response reports a cost (100 credits per USD).
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
This block retrieves one enrichment and returns its status, title, description, format, options, the instructions Exa generated for it, metadata and timestamps. It reports status only; it has no per-item progress.

Use this to check whether an enrichment has finished, verify its configuration, or troubleshoot unexpected results. It charges only if Exa's response reports a cost (100 credits per USD).
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
This block sends a PATCH to the enrichment with only the inputs you set: description, format, options (1-150 labels, checked before calling Exa) or metadata. The title cannot be changed here.

It returns the enrichment's status, title, description and format as Exa reports them after the change. It charges only if Exa's response reports a cost (100 credits per USD).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| enrichment_id | The ID of the enrichment to update | str | Yes |
| description | New description for what data to extract (up to 5,000 characters) | str | No |
| format | New format: text, date, number, options, email, phone or url | "text" \| "date" \| "number" \| "options" \| "email" \| "phone" \| "url" | No |
| options | New labels to choose from when format is 'options' (1-150) | List[str] | No |
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
