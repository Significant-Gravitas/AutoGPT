# Generic Webhook Triggers
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Generic Webhook Trigger

### What it is
This block will output the contents of the generic input for the webhook.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| constants | The constants to be set when the block is put on the graph | Dict[str, Any] | No |
| secret_token | Optional. If set, the platform will only accept incoming webhook requests that include this exact value in the 'X-Webhook-Secret' header. Leave empty for unauthenticated webhooks (the URL itself is the only credential). | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| payload | The complete webhook payload that was received from the generic webhook. | Dict[str, Any] |
| constants | The constants to be set when the block is put on the graph | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
