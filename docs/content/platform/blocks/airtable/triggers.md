# Airtable Webhook Trigger

### What it is
Starts a flow whenever Airtable emits a webhook event.

### What it does
Starts a flow whenever Airtable emits a webhook event

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| base_id | Airtable base ID | str | Yes |
| table_id_or_name | Airtable table ID or name | str | Yes |
| events | Airtable webhook event filter | AirtableEventSelector | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| payload | Airtable webhook payload | WebhookPayload |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
