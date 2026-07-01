# Sardis Policy
<!-- MANUAL: file_description -->
Use these blocks to check whether a proposed Sardis wallet payment would pass policy before funds move.
<!-- END MANUAL -->

## Sardis Policy Check

### What it is
Check if a payment would pass spending policy without executing it. Useful for pre-validation.

### How it works
<!-- MANUAL: how_it_works -->
This block sends a dry-run-style policy check to the Sardis API with the wallet, destination, amount, and token. Sardis evaluates the request against the wallet's configured spending rules and reports whether the payment would be allowed and how much limit would remain afterward.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| wallet_id | Sardis wallet ID (starts with wal_) | str | Yes |
| destination | Recipient address or merchant ID | str | Yes |
| amount | Payment amount to check as a decimal string (e.g. '25.00'). String type avoids IEEE 754 float rounding. | str | Yes |
| token | Token to check | "USDC" \| "USDT" \| "EURC" \| "PYUSD" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| allowed | Whether the payment would be allowed | bool |
| reason | Explanation of the policy decision | str |
| remaining_limit | Remaining spending limit after this payment (decimal string) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Pre-flight validation:** Dry-run a payment against spending rules before committing funds.

**Workflow gating:** Only proceed to the payment step if the policy engine confirms the transaction is within limits.

**Audit trail:** Log every policy decision for compliance reporting without executing any on-chain transaction.
<!-- END MANUAL -->

---
