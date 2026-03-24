"""Spraay batch payment block for sending crypto payments to multiple recipients."""

import uuid
from enum import Enum
from typing import Any

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import CredentialsMetaInput, SchemaField

from ._api import SpraayAPIError, spraay_request
from ._config import spraay_provider


class ChainNetwork(str, Enum):
    BASE = "base"
    ETHEREUM = "ethereum"
    ARBITRUM = "arbitrum"
    POLYGON = "polygon"
    BNB = "bnb"
    AVALANCHE = "avalanche"
    UNICHAIN = "unichain"
    PLASMA = "plasma"
    BOB = "bob"
    SOLANA = "solana"
    BITTENSOR = "bittensor"
    STACKS = "stacks"
    BITCOIN = "bitcoin"


class SpraayBatchPaymentBlock(Block):
    """
    Send batch crypto payments to multiple recipients in a single transaction.

    Spraay batches multiple transfers into one on-chain transaction, saving gas
    and time. Supports ERC-20 tokens, native tokens, and USDC across 13 chains.
    Payment is processed via the x402 protocol (USDC micropayment per API call).
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput[Any, Any] = spraay_provider.credentials_field(
            description="Spraay API credentials",
        )
        chain: ChainNetwork = SchemaField(
            description="Blockchain network to send payments on",
            default=ChainNetwork.BASE,
        )
        token_address: str = SchemaField(
            description=(
                "Contract address of the token to send. "
                "Use '0x0000000000000000000000000000000000000000' for native token (ETH, MATIC, etc.) "
                "or a valid ERC-20 contract address. "
                "USDC on Base: 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
            ),
        )
        recipients: list[str] = SchemaField(
            description="List of recipient wallet addresses",
            min_length=1,
        )
        amounts: list[str] = SchemaField(
            description=(
                "List of amounts to send (as strings, in token decimals). "
                "Must match the length of recipients. "
                "Example: ['1000000'] for 1 USDC (6 decimals)"
            ),
            min_length=1,
        )
        sender_address: str = SchemaField(
            description="Wallet address of the sender (must have approved the Spraay contract)",
        )

    class Output(BlockSchema):
        transaction_hash: str = SchemaField(
            description="On-chain transaction hash of the batch payment",
        )
        batch_id: str = SchemaField(
            description="Spraay batch ID for tracking",
        )
        status: str = SchemaField(
            description="Transaction status (pending, confirmed, failed)",
        )
        total_recipients: int = SchemaField(
            description="Number of recipients in the batch",
        )
        error: str = SchemaField(
            description="Error message if the batch payment failed",
        )

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            description=(
                "Send batch crypto payments to multiple recipients in one transaction. "
                "Supports 13 blockchain networks including Base, Ethereum, Solana, and Bitcoin. "
                "Powered by Spraay's x402 payment gateway."
            ),
            categories={BlockCategory.FINANCE},
            input_schema=SpraayBatchPaymentBlock.Input,
            output_schema=SpraayBatchPaymentBlock.Output,
            test_input={
                "credentials": {
                    "provider": "spraay",
                    "id": str(uuid.uuid4()),
                    "type": "api_key",
                },
                "chain": "base",
                "token_address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                "recipients": [
                    "0x1234567890abcdef1234567890abcdef12345678",
                    "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
                ],
                "amounts": ["1000000", "2000000"],
                "sender_address": "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            },
            test_output=[
                ("transaction_hash", "0xabc123..."),
                ("batch_id", "batch_001"),
                ("status", "confirmed"),
                ("total_recipients", 2),
            ],
            test_mock={
                "execute_batch": lambda *args, **kwargs: {
                    "transaction_hash": "0xabc123...",
                    "batch_id": "batch_001",
                    "status": "confirmed",
                    "total_recipients": 2,
                }
            },
        )

    @staticmethod
    def execute_batch(
        api_key: str,
        chain: str,
        token_address: str,
        recipients: list[str],
        amounts: list[str],
        sender_address: str,
    ) -> dict:
        return spraay_request(
            method="POST",
            endpoint="/v1/batch/send",
            api_key=api_key,
            json_body={
                "chain": chain,
                "tokenAddress": token_address,
                "recipients": recipients,
                "amounts": amounts,
                "senderAddress": sender_address,
            },
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            if len(input_data.recipients) != len(input_data.amounts):
                raise ValueError(
                    f"Recipients ({len(input_data.recipients)}) and amounts "
                    f"({len(input_data.amounts)}) lists must be the same length"
                )

            api_key = input_data.credentials.api_key.get_secret_value()

            result = self.execute_batch(
                api_key=api_key,
                chain=input_data.chain.value,
                token_address=input_data.token_address,
                recipients=input_data.recipients,
                amounts=input_data.amounts,
                sender_address=input_data.sender_address,
            )

            yield "transaction_hash", result.get("transaction_hash", "")
            yield "batch_id", result.get("batch_id", "")
            yield "status", result.get("status", "pending")
            yield "total_recipients", len(input_data.recipients)

        except SpraayAPIError as e:
            yield "error", f"Spraay API error: {e.message}"
        except ValueError as e:
            yield "error", str(e)
        except Exception as e:
            yield "error", f"Unexpected error: {str(e)}"
