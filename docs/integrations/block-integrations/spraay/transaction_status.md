# Spraay Transaction Status
<!-- MANUAL: file_description -->
Blocks for checking the on-chain status of EVM transactions through the Spraay gateway.
<!-- END MANUAL -->

## Spraay Transaction Status

### What it is
Check the on-chain status of an EVM transaction and decode it into a plain-English summary with token transfers. Free Spraay gateway endpoint.

### How it works
<!-- MANUAL: how_it_works -->
This block calls the Spraay gateway's free `GET /api/v1/tx/decode` endpoint, which reads the transaction from a Blockscout explorer. It needs no credentials and makes no x402 payment. It works on Base, Ethereum, Arbitrum, Optimism and Polygon.

The block outputs the `status` reported by the explorer (for example `ok` or `error`), the `block_number` the transaction was included in (0 while it is still pending), the gas used and the fee in wei. It also outputs a plain-English `summary`, a decoded `transaction_type` such as transfer, swap or approval, and the list of token transfers the transaction made. If the gateway's response has no status, the block outputs an error.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| chain | Chain the transaction was sent on | "base" \| "ethereum" \| "arbitrum" \| "optimism" \| "polygon" | No |
| transaction_hash | Transaction hash (0x followed by 64 hex characters) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| status | Transaction status reported by the explorer (e.g. ok, error) | str |
| block_number | Block the transaction was included in (0 if pending) | int |
| gas_used | Gas used | str |
| fee | Fee paid, in wei | str |
| summary | Plain-English summary of the transaction | str |
| transaction_type | Decoded type (transfer, swap, approval, ...) | str |
| token_transfers | Token transfers performed by the transaction | List[Dict[str, Any]] |

### Possible use case
<!-- MANUAL: use_case -->
**Payment Confirmation**: After a Spraay batch payment is broadcast, check its hash until it is included in a block.

**Failure Handling**: Branch a workflow on `status` to retry or alert someone when a transaction fails.

**Payout Log**: Record the summary and token transfers of each completed payment in a spreadsheet or database.
<!-- END MANUAL -->

---
