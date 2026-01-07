# Exa Webset Webhook

### What it is
Receive webhook notifications for Exa webset events.

### What it does
Receive webhook notifications for Exa webset events

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The webset ID to monitor (optional, monitors all if empty) | str | No |
| event_filter | Configure which events to receive | WebsetEventFilter | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| event_type | Type of event that occurred | str |
| event_id | Unique identifier for this event | str |
| webset_id | ID of the affected webset | str |
| data | Event-specific data | Dict[str, True] |
| timestamp | When the event occurred | str |
| metadata | Additional event metadata | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
