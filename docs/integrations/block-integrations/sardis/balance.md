# Sardis Balance
<!-- MANUAL: file_description -->
Use these blocks to inspect the spendable balance and remaining limits on a Sardis wallet.
<!-- END MANUAL -->

## Sardis Balance

### What it is
Check the balance and remaining spending limits of a Sardis wallet.

### How it works
<!-- MANUAL: how_it_works -->
This block calls the Sardis balance endpoint for a specific wallet and token. It returns the current token balance together with the remaining policy limit so a workflow can decide whether it has enough room to continue.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| wallet_id | Sardis wallet ID (starts with wal_) | str | Yes |
| token | Token to check | "USDC" \| "USDT" \| "EURC" \| "PYUSD" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| balance | Current balance (decimal string) | str |
| remaining_limit | Remaining spending limit (decimal string) | str |
| token | Token type | str |

### Possible use case
<!-- MANUAL: use_case -->
**Budget monitoring:** Check remaining spending limits before executing a workflow.

**Multi-agent oversight:** Monitor balances across multiple agent wallets from a single dashboard flow.

**Threshold alerts:** Trigger notifications when wallet balance drops below a critical level.
<!-- END MANUAL -->

---
