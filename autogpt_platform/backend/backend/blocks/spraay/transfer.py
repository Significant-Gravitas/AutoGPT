"""Spraay single token transfer block."""

import uuid
from typing import Any

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import CredentialsMetaInput, SchemaField

from ._api import SpraayAPIError, spraay_request
from ._config import spraay_provider
from .batch_payment import ChainNetwork


class SpraayTokenTransferBlock(Block):
    """
    Send a single crypto token transfer on any supported chain.

    A simplified interface for one-to-one transfers. Under the hood, this uses
    the same Spraay batch infrastructure but optimized for single-recipient
    payments. Ideal for agent-to-agent payments, tips, bounties, or payouts.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput[Any, Any] = spraay_provider.credentials_field(
            description="Spraay API credentials",
        )
        chain: ChainNetwork = SchemaField(
            description="Blockchain network",
            default=ChainNetwork.BASE,
        )
        token_address: str = SchemaField(
            description=(
                "Token contract address. "
                "Use '0x0000000000000000000000000000000000000000' for native token. "
                "USDC on Base: 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
            ),
        )
        recipient: str = SchemaField(
            description="Recipient wallet address",
        )
        amount: str = SchemaField(
            description=(
                "Amount to send as a string in token's smallest unit. "
                "Example: '1000000' = 1 USDC (6 decimals)"
            ),
        )
        sender_address: str = SchemaField(
            description="Sender wallet address",
        )

    class Output(BlockSchema):
        transaction_hash: str = SchemaField(
            description="On-chain transaction hash",
        )
        status: str = SchemaField(
            description="Transaction status",
        )
        error: str = SchemaField(
            description="Error message if the transfer failed",
        )

    def __init__(self):
        super().__init__(
            id="b2c3d4e5-f6a7-8901-bcde-f12345678901",
            description=(
                "Send a single crypto token transfer to one recipient. "
                "Supports native tokens and ERC-20s across 13 chains. "
                "Powered by Spraay's x402 payment gateway."
            ),
            categories={BlockCategory.FINANCE},
            input_schema=SpraayTokenTransferBlock.Input,
            output_schema=SpraayTokenTransferBlock.Output,
            test_input={
                "credentials": {
                    "provider": "spraay",
                    "id": str(uuid.uuid4()),
                    "type": "api_key",
                },
                "chain": "base",
                "token_address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                "recipient": "0x1234567890abcdef1234567890abcdef12345678",
                "amount": "1000000",
                "sender_address": "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            },
            test_output=[
                ("transaction_hash", "0xdef456..."),
                ("status", "confirmed"),
            ],
            test_mock={
                "execute_transfer": lambda *args, **kwargs: {
                    "transaction_hash": "0xdef456...",
                    "status": "confirmed",
                }
            },
        )

    @staticmethod
    def execute_transfer(
        api_key: str,
        chain: str,
        token_address: str,
        recipient: str,
        amount: str,
        sender_address: str,
    ) -> dict:
        return spraay_request(
            method="POST",
            endpoint="/v1/transfer/send",
            api_key=api_key,
            json_body={
                "chain": chain,
                "tokenAddress": token_address,
                "recipient": recipient,
                "amount": amount,
                "senderAddress": sender_address,
            },
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            api_key = input_data.credentials.api_key.get_secret_value()

            result = self.execute_transfer(
                api_key=api_key,
                chain=input_data.chain.value,
                token_address=input_data.token_address,
                recipient=input_data.recipient,
                amount=input_data.amount,
                sender_address=input_data.sender_address,
            )

            yield "transaction_hash", result.get("transaction_hash", "")
            yield "status", result.get("status", "pending")

        except SpraayAPIError as e:
            yield "error", f"Spraay API error: {e.message}"
        except Exception as e:
            yield "error", f"Unexpected error: {str(e)}"
