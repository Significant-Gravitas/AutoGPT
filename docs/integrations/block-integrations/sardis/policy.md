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
| amount | Payment amount to check | float | Yes |
| token | Token to check (USDC, USDT, EURC, PYUSD) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if failed | str |
| allowed | Whether the payment would be allowed | bool |
| reason | Explanation of the policy decision | str |
| remaining_limit | Remaining spending limit after this payment | float |

### Possible use case
<!-- MANUAL: use_case -->
**Validate checkout eligibility:** Confirm a proposed payment would pass policy before showing a final checkout decision.

**Gate wallet transfers:** Check whether an operational transfer is allowed before attempting an on-chain movement of funds.

**Route around limit failures:** Estimate remaining policy headroom before deciding to split, defer, or reroute a payment.
<!-- END MANUAL -->

---
