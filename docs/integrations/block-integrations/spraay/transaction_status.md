# Spraay Transaction Status
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Spraay Transaction Status

### What it is
Check the on-chain status of an EVM transaction and decode it into a plain-English summary with token transfers. Free Spraay gateway endpoint.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
