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
| amount | Payment amount as a decimal string (e.g. '25.00'). String type avoids IEEE 754 float rounding. | str | Yes |
| token | Token to use | "USDC" \| "USDT" \| "EURC" \| "PYUSD" | No |
| chain | Blockchain to use | "base" \| "polygon" \| "ethereum" \| "arbitrum" \| "optimism" | No |
| purpose | Reason for payment (used in audit trail) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if failed | str |
| status | APPROVED, BLOCKED, or ERROR | str |
| tx_id | Transaction ID if approved | str |
| message | Status message | str |
| amount | Payment amount (decimal string) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Pay a supplier:** Execute an approved stablecoin payout to a vendor when a workflow reaches settlement.

**Settle a tool invoice:** Pay an external API or service bill from a policy-controlled operational wallet.

**Move treasury funds:** Transfer funds between wallets while preserving Sardis policy checks and auditability.
<!-- END MANUAL -->

---
