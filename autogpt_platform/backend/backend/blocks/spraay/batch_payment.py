"""Spraay batch payment block for paying multiple recipients in one transaction.

The gateway is non-custodial: ``POST /api/v1/batch/execute`` returns an
unsigned transaction against the Spraay contract (plus the ERC-20 approval it
needs) for the sender to sign and broadcast. Implements BPA 1.0
(https://docs.spraay.app/bpa/1.0/).
"""

from enum import Enum
from typing import Any

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import spraay_request
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, spraay_provider

MAX_BATCH_RECIPIENTS = 200


class ChainNetwork(str, Enum):
    """Chains the Spraay gateway settles batch payments on.

    Only chains whose Spraay contract bytecode has been verified identical to
    the Base deployment are settleable. x402 fees are paid separately, in USDC
    on Base or USDG on Robinhood Chain (see ``_api.X402_PAYMENT_NETWORKS``).
    """

    BASE = "base"
    PEAQ = "peaq"


class SpraayBatchPaymentBlock(Block):
    """Prepare a batch payment to up to 200 recipients in one transaction.

    Returns the unsigned Spraay contract call and, for ERC-20 tokens, the
    approval the sender must grant first. Costs $0.02 per call via x402.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = spraay_provider.credentials_field(
            description="Spraay x402 payer wallet private key",
        )
        chain: ChainNetwork = SchemaField(
            description="Settlement chain for the batch",
            default=ChainNetwork.BASE,
        )
        token: str = SchemaField(
            description=(
                "Token symbol (Base: USDC, USDT, EURC, DAI, WETH, or ETH for "
                "native; peaq: USDC) or an ERC-20 contract address on the chain"
            ),
            default="USDC",
        )
        recipients: list[str] = SchemaField(
            description="Recipient wallet addresses (max 200)",
            min_length=1,
            max_length=MAX_BATCH_RECIPIENTS,
        )
        amounts: list[str] = SchemaField(
            description=(
                "Amounts in RAW base units, parallel to recipients. "
                "USDC has 6 decimals, so '1500000' = 1.50 USDC"
            ),
            min_length=1,
            max_length=MAX_BATCH_RECIPIENTS,
        )
        sender: str = SchemaField(
            description="Payer address that will sign and broadcast the batch",
        )

    class Output(BlockSchemaOutput):
        transaction: dict[str, Any] = SchemaField(
            description=(
                "Unsigned transaction to sign and broadcast: "
                "to, data, value (wei), chainId, gasLimit (hex)"
            ),
        )
        approval_required: dict[str, Any] = SchemaField(
            description=(
                "ERC-20 approval to grant before broadcasting: token, spender, "
                "amount (raw, pass verbatim). Empty for native ETH batches"
            ),
            default_factory=dict,
        )
        contract: str = SchemaField(
            description="Spraay contract the transaction targets",
        )
        token_symbol: str = SchemaField(description="Resolved token symbol")
        recipient_count: int = SchemaField(
            description="Number of recipients in the batch",
        )
        total_amount: str = SchemaField(
            description="Total sent to recipients, as a human-decimal string",
        )
        fee: str = SchemaField(
            description="Spraay protocol fee (0.3%), as a human-decimal string",
        )
        total_with_fee: str = SchemaField(
            description="Total debited from the sender, as a human-decimal string",
        )

    def __init__(self):
        super().__init__(
            id="9f9ce1c7-a021-4611-90bf-6f14ee7ccf84",
            description=(
                "Prepare a non-custodial batch payment to up to 200 recipients "
                "on Base or peaq. Returns the unsigned transaction and any "
                "required token approval. Paid per call via Spraay x402."
            ),
            categories={BlockCategory.BASIC},
            effect=BlockEffect.EXTERNAL,
            input_schema=SpraayBatchPaymentBlock.Input,
            output_schema=SpraayBatchPaymentBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "chain": "base",
                "token": "USDC",
                "recipients": [
                    "0x1234567890abcdef1234567890abcdef12345678",
                    "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
                ],
                "amounts": ["1000000", "2000000"],
                "sender": "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            },
            test_output=[
                (
                    "transaction",
                    {
                        "to": "0x1646452F98E36A3c9Cfc3eDD8868221E207B5eEC",
                        "data": "0xabcdef",
                        "value": "0",
                        "chainId": 8453,
                        "gasLimit": "0x38270",
                    },
                ),
                (
                    "approval_required",
                    {
                        "token": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                        "spender": "0x1646452F98E36A3c9Cfc3eDD8868221E207B5eEC",
                        "amount": "3009000",
                        "amountFormatted": "3.009",
                    },
                ),
                ("contract", "0x1646452F98E36A3c9Cfc3eDD8868221E207B5eEC"),
                ("token_symbol", "USDC"),
                ("recipient_count", 2),
                ("total_amount", "3.0"),
                ("fee", "0.009"),
                ("total_with_fee", "3.009"),
            ],
            test_mock={
                "execute_batch": lambda *args, **kwargs: {
                    "success": True,
                    "contract": "0x1646452F98E36A3c9Cfc3eDD8868221E207B5eEC",
                    "token": {
                        "symbol": "USDC",
                        "address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                        "decimals": 6,
                        "isETH": False,
                    },
                    "batch": {
                        "recipientCount": 2,
                        "totalAmount": "3.0",
                        "fee": "0.009",
                        "feePercent": "0.3%",
                        "totalWithFee": "3.009",
                    },
                    "transaction": {
                        "to": "0x1646452F98E36A3c9Cfc3eDD8868221E207B5eEC",
                        "data": "0xabcdef",
                        "value": "0",
                        "chainId": 8453,
                        "gasLimit": "0x38270",
                    },
                    "approvalRequired": {
                        "token": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                        "spender": "0x1646452F98E36A3c9Cfc3eDD8868221E207B5eEC",
                        "amount": "3009000",
                        "amountFormatted": "3.009",
                    },
                }
            },
        )

    @staticmethod
    async def execute_batch(
        private_key: str,
        chain: str,
        token: str,
        recipients: list[str],
        amounts: list[str],
        sender: str,
    ) -> dict[str, Any]:
        """Request an unsigned batch transaction from the Spraay gateway."""
        return await spraay_request(
            method="POST",
            endpoint="/api/v1/batch/execute",
            private_key=private_key,
            json_body={
                "chain": chain,
                "token": token,
                "recipients": recipients,
                "amounts": amounts,
                "sender": sender,
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        if len(input_data.recipients) != len(input_data.amounts):
            raise ValueError(
                f"Recipients ({len(input_data.recipients)}) and amounts "
                f"({len(input_data.amounts)}) lists must be the same length"
            )

        result = await self.execute_batch(
            private_key=credentials.api_key.get_secret_value(),
            chain=input_data.chain.value,
            token=input_data.token,
            recipients=input_data.recipients,
            amounts=input_data.amounts,
            sender=input_data.sender,
        )

        if "transaction" not in result:
            yield "error", "Gateway returned incomplete response: missing 'transaction'"
            return

        batch = result.get("batch", {})
        yield "transaction", result["transaction"]
        yield "approval_required", result.get("approvalRequired", {})
        yield "contract", result.get("contract", "")
        yield "token_symbol", result.get("token", {}).get("symbol", "")
        yield "recipient_count", batch.get("recipientCount", len(input_data.recipients))
        yield "total_amount", batch.get("totalAmount", "")
        yield "fee", batch.get("fee", "")
        yield "total_with_fee", batch.get("totalWithFee", "")
