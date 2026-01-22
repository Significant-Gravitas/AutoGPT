# HubSpot Engagement
<!-- MANUAL: file_description -->
Blocks for managing HubSpot engagements like emails and tracking metrics.
<!-- END MANUAL -->

## Hub Spot Engagement

### What it is
Manages HubSpot engagements - sends emails and tracks engagement metrics

### How it works
<!-- MANUAL: how_it_works -->
This block manages HubSpot engagements including sending emails and tracking engagement metrics. Use send_email to send emails through HubSpot, or track_engagement to retrieve engagement history for a contact.

Engagement tracking returns metrics like email opens, clicks, and other interactions within a specified timeframe.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| operation | Operation to perform (send_email, track_engagement) | str | No |
| email_data | Email data including recipient, subject, content | Dict[str, Any] | No |
| contact_id | Contact ID for engagement tracking | str | No |
| timeframe_days | Number of days to look back for engagement | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| result | Operation result | Dict[str, Any] |
| status | Operation status | str |

### Possible use case
<!-- MANUAL: use_case -->
**Automated Outreach**: Send personalized emails to contacts based on triggers or workflows.

**Engagement Scoring**: Track contact engagement to prioritize outreach efforts.

**Follow-Up Automation**: Trigger follow-up actions based on engagement metrics.
<!-- END MANUAL -->

---
