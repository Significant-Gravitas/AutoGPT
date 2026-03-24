# Sardis Payment
<!-- MANUAL: file_description -->
Use these blocks to send policy-controlled payments from a Sardis wallet.
<!-- END MANUAL -->

## Sardis Pay

### What it is
Execute a policy-controlled payment from a Sardis wallet. Each payment is verified against spending policies before execution.

### How it works
<!-- MANUAL: how_it_works -->
This block sends a transfer request to the Sardis API using a Sardis wallet ID and API key. Sardis evaluates the wallet's spending policy before execution and returns whether the payment was approved, blocked, or rejected with an error.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| wallet_id | Sardis wallet ID (starts with wal_) | str | Yes |
| destination | Recipient address, merchant ID, or wallet ID | str | Yes |
| amount | Payment amount in token units | float | Yes |
| token | Token to use (USDC, USDT, EURC, PYUSD) | str | No |
| chain | Blockchain to use (base, polygon, ethereum, arbitrum, optimism) | str | No |
| purpose | Reason for payment (used in audit trail) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if failed | str |
| status | APPROVED, BLOCKED, or ERROR | str |
| tx_id | Transaction ID if approved | str |
| message | Status message | str |
| amount | Payment amount | float |

### Possible use case
<!-- MANUAL: use_case -->
Use this block when an agent needs to pay for an external service after it has already decided to proceed, such as funding a supplier, paying a tool invoice, or transferring stablecoins to another operational wallet.
<!-- END MANUAL -->

---
