# Spraay Balance
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Spraay Get Balance

### What it is
Check native ETH and ERC-20 balances for any wallet on Base. Paid per call via Spraay x402.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| wallet_address | Wallet address to check balances for | str | Yes |
| token_address | ERC-20 contract address on Base to check. Leave empty to list ETH plus non-zero balances of popular Base tokens. USDC on Base: 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913 | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| balances | All returned balances: symbol, address, decimals, balance (human decimal), balanceRaw (base units) | List[Dict[str, Any]] |
| balance | Requested token's balance in base units | str |
| formatted_balance | Requested token's balance with decimals applied | str |
| token_symbol | Requested token's symbol (e.g. USDC) | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
