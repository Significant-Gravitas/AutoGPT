"""Spraay token balance check block.

This module provides the SpraayGetBalanceBlock, which queries the balance
of any ERC-20 token or native token for a given wallet address on any
of the 13 supported blockchain networks.
"""

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import spraay_request
from ._config import spraay_provider
from .batch_payment import ChainNetwork


class SpraayGetBalanceBlock(Block):
    """Check a wallet's token balance on any supported chain.

    Query the balance of any ERC-20 token or native token for a given wallet
    address. Useful for pre-flight checks before sending payments, monitoring
    agent wallets, or building conditional workflows based on funds available.
    """

    class Input(BlockSchema):
        """Input schema for the balance check block."""

        credentials: CredentialsMetaInput = spraay_provider.credentials_field(
            description="Spraay API credentials",
        )
        chain: ChainNetwork = SchemaField(
            description="Blockchain network to query",
            default=ChainNetwork.BASE,
        )
        wallet_address: str = SchemaField(
            description="Wallet address to check balance for",
        )
        token_address: str = SchemaField(
            description=(
                "Token contract address. "
                "Use '0x0000000000000000000000000000000000000000' for native token. "
                "USDC on Base: 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
            ),
        )

    class Output(BlockSchema):
        """Output schema for the balance check block."""

        balance: str = SchemaField(
            description="Token balance in the token's smallest unit (wei/lamports/sats)",
        )
        formatted_balance: str = SchemaField(
            description="Human-readable balance with decimals applied",
        )
        token_symbol: str = SchemaField(
            description="Symbol of the token (e.g. USDC, ETH, SOL)",
        )
        error: str = SchemaField(
            description="Error message if the balance check failed",
        )

    def __init__(self):
        """Initialize the SpraayGetBalanceBlock with metadata and test fixtures."""
        super().__init__(
            id="b81a036d-8bf7-460c-bca8-b1242d0aa3cb",
            description=(
                "Check token balance for any wallet on 13 supported chains. "
                "Supports native tokens and ERC-20s. "
                "Powered by Spraay's x402 payment gateway."
            ),
            categories={BlockCategory.BASIC},
            input_schema=SpraayGetBalanceBlock.Input,
            output_schema=SpraayGetBalanceBlock.Output,
            test_input={
                "credentials": {
                    "provider": "spraay",
                    "id": "test-cred-id",
                    "type": "api_key",
                },
                "chain": "base",
                "wallet_address": "0x1234567890abcdef1234567890abcdef12345678",
                "token_address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            },
            test_output=[
                ("balance", "5000000"),
                ("formatted_balance", "5.0"),
                ("token_symbol", "USDC"),
            ],
            test_mock={
                "fetch_balance": lambda *args, **kwargs: {
                    "balance": "5000000",
                    "formatted_balance": "5.0",
                    "token_symbol": "USDC",
                }
            },
        )

    @staticmethod
    def fetch_balance(
        api_key: str,
        chain: str,
        wallet_address: str,
        token_address: str,
    ) -> dict:
        """Fetch the token balance for a wallet via the Spraay gateway.

        Args:
            api_key: Spraay API key for authentication.
            chain: Target blockchain network identifier.
            wallet_address: Wallet address to query the balance for.
            token_address: Contract address of the token to check.

        Returns:
            Dictionary containing balance, formatted_balance, and token_symbol.

        Raises:
            SpraayAPIError: If the gateway returns an error.
        """
        return spraay_request(
            method="GET",
            endpoint="/v1/balance",
            api_key=api_key,
            params={
                "chain": chain,
                "walletAddress": wallet_address,
                "tokenAddress": token_address,
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        """Execute the balance check block.

        Args:
            input_data: Validated input containing chain, wallet address,
                and token address.
            credentials: API key credentials injected by the framework.
            **kwargs: Additional keyword arguments passed by the executor.

        Yields:
            Output fields: balance, formatted_balance, token_symbol, or error.
        """
        api_key = credentials.api_key.get_secret_value()

        result = self.fetch_balance(
            api_key=api_key,
            chain=input_data.chain.value,
            wallet_address=input_data.wallet_address,
            token_address=input_data.token_address,
        )

        yield "balance", result.get("balance", "0")
        yield "formatted_balance", result.get("formatted_balance", "0")
        yield "token_symbol", result.get("token_symbol", "")
