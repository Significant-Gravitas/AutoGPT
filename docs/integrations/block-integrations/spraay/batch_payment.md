# Spraay Batch Payment
<!-- MANUAL: file_description -->
Blocks for preparing payments to many recipients in a single transaction through the Spraay x402 gateway. The gateway is non-custodial: it returns an unsigned transaction for your own wallet to sign and broadcast.
<!-- END MANUAL -->

## Spraay Batch Payment

### What it is
Prepare a non-custodial batch payment to up to 200 recipients on Base or peaq. Returns the unsigned transaction and any required token approval. Paid per call via Spraay x402.

### How it works
<!-- MANUAL: how_it_works -->
This block checks that `recipients` and `amounts` have the same length (up to 200 entries), then calls the Spraay gateway's `POST /api/v1/batch/execute` endpoint. The call costs $0.02, paid through x402 from the payer wallet in your Spraay credentials, which can be a different wallet from `sender`.

The gateway returns an unsigned transaction that calls the Spraay contract on the chosen chain (Base or peaq). The block does not sign or broadcast anything and never moves the batch funds itself: the `sender` wallet has to sign and send `transaction`. For ERC-20 tokens, `approval_required` describes the token approval the sender must grant to the Spraay contract first; pass its amount through unchanged. For native ETH batches it is empty.

Amounts are in raw base units (USDC has 6 decimals, so `1500000` is 1.50 USDC). Spraay adds a 0.3% protocol fee, so `total_with_fee` is what the sender is debited, while `total_amount` is what the recipients receive.
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
**Contributor Payouts**: Pay a list of contributors their monthly USDC amounts in one transaction instead of dozens.

**Rewards Distribution**: Send token rewards to community members or bounty winners from a spreadsheet of addresses and amounts.

**Review Before Sending**: Prepare the transaction in a workflow, then hand it to a person who checks and signs it in their own wallet.
<!-- END MANUAL -->

---
