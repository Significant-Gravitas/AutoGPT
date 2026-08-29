# Smartlead Campaign
<!-- MANUAL: file_description -->
Blocks for managing email outreach campaigns in SmartLead.
<!-- END MANUAL -->

## Add Lead To Campaign

### What it is
Add a lead to a campaign in SmartLead

### How it works
<!-- MANUAL: how_it_works -->
This block adds up to 100 leads to an existing SmartLead campaign using the SmartLead API. Each lead includes contact details and optional custom fields for personalization.

Configure upload settings to control duplicate handling and campaign status. The response includes counts for successful uploads, duplicates, and invalid entries.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| campaign_id | The ID of the campaign to add the lead to | int | Yes |
| lead_list | An array of JSON objects, each representing a lead's details. Can hold max 100 leads. | List[LeadInput] | No |
| settings | Settings for lead upload | LeadUploadSettings | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the lead was not added to the campaign | str |
| campaign_id | The ID of the campaign the lead was added to (passed through) | int |
| upload_count | The number of leads added to the campaign | int |
| already_added_to_campaign | The number of leads that were already added to the campaign | int |
| duplicate_count | The number of emails that were duplicates | int |
| invalid_email_count | The number of emails that were invalidly formatted | int |
| is_lead_limit_exhausted | Whether the lead limit was exhausted | bool |
| lead_import_stopped_count | The number of leads that were not added to the campaign because the lead import was stopped | int |

### Possible use case
<!-- MANUAL: use_case -->
**Lead Import**: Bulk import leads from CRM exports, web forms, or enrichment services.

**Campaign Automation**: Automatically add qualifying leads to outreach campaigns.

**Multi-Source Aggregation**: Consolidate leads from multiple sources into unified campaigns.
<!-- END MANUAL -->

---

## Create Campaign

### What it is
Create a campaign in SmartLead

### How it works
<!-- MANUAL: how_it_works -->
This block creates a new email campaign in SmartLead using the SmartLead API. Provide a campaign name and the block returns the created campaign's ID and metadata.

Use the campaign ID with other SmartLead blocks to add leads, configure sequences, and manage the campaign.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the campaign | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the search failed | str |
| id | The ID of the created campaign | int |
| name | The name of the created campaign | str |
| created_at | The date and time the campaign was created | str |

### Possible use case
<!-- MANUAL: use_case -->
**Dynamic Campaigns**: Create campaigns programmatically for different products or segments.

**Workflow Automation**: Spin up new outreach campaigns as part of sales or marketing workflows.

**Campaign Templating**: Create campaigns from templates with standardized configurations.
<!-- END MANUAL -->

---

## Save Campaign Sequences

### What it is
Save sequences within a campaign

### How it works
<!-- MANUAL: how_it_works -->
This block saves email sequences to an existing SmartLead campaign. Sequences define the email content, timing, and follow-up structure for the campaign's outreach.

Each sequence includes the email subject, body, and delay settings for automated follow-up emails.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| campaign_id | The ID of the campaign to save sequences for | int | Yes |
| sequences | The sequences to save | List[Sequence] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the sequences were not saved | str |
| data | Data from the API | Dict[str, Any] \| str |
| message | Message from the API | str |

### Possible use case
<!-- MANUAL: use_case -->
**Email Automation**: Define multi-step email sequences for nurturing leads.

**A/B Testing**: Create variant sequences to test different messaging approaches.

**Campaign Configuration**: Set up complete outreach flows programmatically.
<!-- END MANUAL -->

---
