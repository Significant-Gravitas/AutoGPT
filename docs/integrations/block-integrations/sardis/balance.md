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
| token | Token to check (USDC, USDT, EURC, PYUSD) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if failed | str |
| balance | Current balance | float |
| remaining_limit | Remaining spending limit | float |
| token | Token type | str |

### Possible use case
<!-- MANUAL: use_case -->
Use this block before planning or executing paid actions so an agent can confirm that a wallet still has both sufficient funds and sufficient policy headroom.
<!-- END MANUAL -->

---
