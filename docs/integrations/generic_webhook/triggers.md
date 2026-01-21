# Generic Webhook Triggers
<!-- MANUAL: file_description -->
Blocks for receiving and processing generic webhook payloads from external services.
<!-- END MANUAL -->

## Generic Webhook Trigger

### What it is
This block will output the contents of the generic input for the webhook.

### How it works
<!-- MANUAL: how_it_works -->
This block creates a webhook endpoint that receives and outputs any incoming HTTP payload. When external services send data to this webhook URL, the block triggers and outputs the complete payload as a dictionary.

Constants can be configured to pass additional static values alongside the dynamic webhook data.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| constants | The constants to be set when the block is put on the graph | Dict[str, Any] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| payload | The complete webhook payload that was received from the generic webhook. | Dict[str, Any] |
| constants | The constants to be set when the block is put on the graph | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**External Integrations**: Receive data from any third-party service that supports webhooks.

**Custom Triggers**: Create custom workflow triggers from external systems or internal tools.

**Event Processing**: Capture and process events from IoT devices, payment processors, or notification services.
<!-- END MANUAL -->

---
