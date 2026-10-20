# Spraay Batch Payment
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Spraay Batch Payment

### What it is
Prepare a non-custodial batch payment to up to 200 recipients on Base or peaq. Returns the unsigned transaction and any required token approval. Paid per call via Spraay x402.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| chain | Settlement chain for the batch | "base" \| "peaq" | No |
| token | Token symbol (Base: USDC, USDT, EURC, DAI, WETH, or ETH for native; peaq: USDC) or an ERC-20 contract address on the chain | str | No |
| recipients | Recipient wallet addresses (max 200) | List[str] | Yes |
| amounts | Amounts in RAW base units, parallel to recipients. USDC has 6 decimals, so '1500000' = 1.50 USDC | List[str] | Yes |
| sender | Payer address that will sign and broadcast the batch | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| transaction | Unsigned transaction to sign and broadcast: to, data, value (wei), chainId, gasLimit (hex) | Dict[str, Any] |
| approval_required | ERC-20 approval to grant before broadcasting: token, spender, amount (raw, pass verbatim). Empty for native ETH batches | Dict[str, Any] |
| contract | Spraay contract the transaction targets | str |
| token_symbol | Resolved token symbol | str |
| recipient_count | Number of recipients in the batch | int |
| total_amount | Total sent to recipients, as a human-decimal string | str |
| fee | Spraay protocol fee (0.3%), as a human-decimal string | str |
| total_with_fee | Total debited from the sender, as a human-decimal string | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
