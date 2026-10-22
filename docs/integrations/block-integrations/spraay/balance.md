# Spraay Balance
<!-- MANUAL: file_description -->
Blocks for reading wallet token balances on Base through the Spraay x402 gateway.
<!-- END MANUAL -->

## Spraay Get Balance

### What it is
Check native ETH and ERC-20 balances for any wallet on Base. Paid per call via Spraay x402.

### How it works
<!-- MANUAL: how_it_works -->
This block calls the Spraay gateway's `GET /api/v1/balances` endpoint for the wallet you give it. Balances are always read on Base. The gateway charges $0.005 per call through x402: the first request returns HTTP 402 with the payment terms, and the block signs a USDC transfer authorization with the payer wallet key stored in your Spraay credentials, then retries the request with that signature.

With no token address, `balances` lists ETH plus any non-zero balances of popular Base tokens. With a token address, the block also outputs that token's raw `balance`, its `formatted_balance` with decimals applied, and its `token_symbol`. If the gateway doesn't return the requested token, the block outputs an error instead.

The Spraay credential is a wallet private key, not an API key. Use a dedicated wallet that holds only a small USDC balance for gateway fees.
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
**Pre-flight Check**: Confirm a sender holds enough USDC before preparing a Spraay batch payment.

**Low Balance Alerts**: Check a payout wallet on a schedule and send a notification when its balance drops below a threshold.

**Treasury Reporting**: Collect the ETH and token holdings of several wallets into a regular report.
<!-- END MANUAL -->

---
