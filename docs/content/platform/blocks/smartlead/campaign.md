# Add Lead To Campaign

### What it is
Add a lead to a campaign in SmartLead.

### What it does
Add a lead to a campaign in SmartLead

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Create Campaign

### What it is
Create a campaign in SmartLead.

### What it does
Create a campaign in SmartLead

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Save Campaign Sequences

### What it is
Save sequences within a campaign.

### What it does
Save sequences within a campaign

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| data | Data from the API | Dict[str, True] | str |
| message | Message from the API | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
