from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Type

import prisma.errors
from autogpt_libs.supabase_integration_credentials_store.store import (
    anthropic_credentials,
    did_credentials,
    groq_credentials,
    ideogram_credentials,
    openai_credentials,
    replicate_credentials,
    revid_credentials,
)
from prisma import Json
from prisma.enums import UserBlockCreditType
from prisma.models import UserBlockCredit
from pydantic import BaseModel

from backend.blocks.ai_shortform_video_block import AIShortformVideoCreatorBlock
from backend.blocks.ideogram import IdeogramModelBlock
from backend.blocks.jina.search import SearchTheWebBlock
from backend.blocks.llm import (
    MODEL_METADATA,
    AIConversationBlock,
    AIStructuredResponseGeneratorBlock,
    AITextGeneratorBlock,
    AITextSummarizerBlock,
    LlmModel,
)
from backend.blocks.replicate_flux_advanced import ReplicateFluxAdvancedModelBlock
from backend.blocks.search import ExtractWebsiteContentBlock
from backend.blocks.talking_head import CreateTalkingAvatarVideoBlock
from backend.data.block import Block, BlockInput, get_block
from backend.util.settings import Config


class BlockCostType(str, Enum):
    RUN = "run"  # cost X credits per run
    BYTE = "byte"  # cost X credits per byte
    SECOND = "second"  # cost X credits per second


class BlockCost(BaseModel):
    cost_amount: int
    cost_filter: BlockInput
    cost_type: BlockCostType

    def __init__(
        self,
        cost_amount: int,
        cost_type: BlockCostType = BlockCostType.RUN,
        cost_filter: Optional[BlockInput] = None,
        **data: Any,
    ) -> None:
        super().__init__(
            cost_amount=cost_amount,
            cost_filter=cost_filter or {},
            cost_type=cost_type,
            **data,
        )


llm_cost = (
    [
        BlockCost(
            cost_type=BlockCostType.RUN,
            cost_filter={
                "model": model,
                "api_key": None,  # Running LLM with user own API key is free.
            },
            cost_amount=metadata.cost_factor,
        )
        for model, metadata in MODEL_METADATA.items()
    ]
    + [
        BlockCost(
            cost_type=BlockCostType.RUN,
            cost_filter={
                "model": model,
                "credentials": {
                    "id": anthropic_credentials.id,
                    "provider": anthropic_credentials.provider,
                    "type": anthropic_credentials.type,
                },
            },
            cost_amount=metadata.cost_factor,
        )
        for model, metadata in MODEL_METADATA.items()
        if metadata.provider == "anthropic"
    ]
    + [
        BlockCost(
            cost_type=BlockCostType.RUN,
            cost_filter={
                "model": model,
                "credentials": {
                    "id": openai_credentials.id,
                    "provider": openai_credentials.provider,
                    "type": openai_credentials.type,
                },
            },
            cost_amount=metadata.cost_factor,
        )
        for model, metadata in MODEL_METADATA.items()
        if metadata.provider == "openai"
    ]
    + [
        BlockCost(
            cost_type=BlockCostType.RUN,
            cost_filter={
                "model": model,
                "credentials": {"id": groq_credentials.id},
            },
            cost_amount=metadata.cost_factor,
        )
        for model, metadata in MODEL_METADATA.items()
        if metadata.provider == "groq"
    ]
    + [
        BlockCost(
            # Default cost is running LlmModel.GPT4O.
            cost_amount=MODEL_METADATA[LlmModel.GPT4O].cost_factor,
            cost_filter={"api_key": None},
        ),
    ]
)

BLOCK_COSTS: dict[Type[Block], list[BlockCost]] = {
    AIConversationBlock: llm_cost,
    AITextGeneratorBlock: llm_cost,
    AIStructuredResponseGeneratorBlock: llm_cost,
    AITextSummarizerBlock: llm_cost,
    CreateTalkingAvatarVideoBlock: [
        BlockCost(
            cost_amount=15,
            cost_filter={
                "credentials": {
                    "id": did_credentials.id,
                    "provider": did_credentials.provider,
                    "type": did_credentials.type,
                }
            },
        )
    ],
    SearchTheWebBlock: [BlockCost(cost_amount=1)],
    ExtractWebsiteContentBlock: [
        BlockCost(cost_amount=1, cost_filter={"raw_content": False})
    ],
    IdeogramModelBlock: [
        BlockCost(
            cost_amount=1,
            cost_filter={
                "credentials": {
                    "id": ideogram_credentials.id,
                    "provider": ideogram_credentials.provider,
                    "type": ideogram_credentials.type,
                }
            },
        )
    ],
    AIShortformVideoCreatorBlock: [
        BlockCost(
            cost_amount=10,
            cost_filter={
                "credentials": {
                    "id": revid_credentials.id,
                    "provider": revid_credentials.provider,
                    "type": revid_credentials.type,
                }
            },
        )
    ],
    ReplicateFluxAdvancedModelBlock: [
        BlockCost(
            cost_amount=10,
            cost_filter={
                "credentials": {
                    "id": replicate_credentials.id,
                    "provider": replicate_credentials.provider,
                    "type": replicate_credentials.type,
                }
            },
        )
    ],
}


class UserCreditBase(ABC):
    def __init__(self, num_user_credits_refill: int):
        self.num_user_credits_refill = num_user_credits_refill

    @abstractmethod
    async def get_or_refill_credit(self, user_id: str) -> int:
        """
        Get the current credit for the user and refill if no transaction has been made in the current cycle.

        Returns:
            int: The current credit for the user.
        """
        pass

    @abstractmethod
    async def spend_credits(
        self,
        user_id: str,
        user_credit: int,
        block_id: str,
        input_data: BlockInput,
        data_size: float,
        run_time: float,
    ) -> int:
        """
        Spend the credits for the user based on the block usage.

        Args:
            user_id (str): The user ID.
            user_credit (int): The current credit for the user.
            block_id (str): The block ID.
            input_data (BlockInput): The input data for the block.
            data_size (float): The size of the data being processed.
            run_time (float): The time taken to run the block.

        Returns:
            int: amount of credit spent
        """
        pass

    @abstractmethod
    async def top_up_credits(self, user_id: str, amount: int):
        """
        Top up the credits for the user.

        Args:
            user_id (str): The user ID.
            amount (int): The amount to top up.
        """
        pass


class UserCredit(UserCreditBase):
    async def get_or_refill_credit(self, user_id: str) -> int:
        cur_time = self.time_now()
        cur_month = cur_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        nxt_month = cur_month.replace(month=cur_month.month + 1)

        user_credit = await UserBlockCredit.prisma().group_by(
            by=["userId"],
            sum={"amount": True},
            where={
                "userId": user_id,
                "createdAt": {"gte": cur_month, "lt": nxt_month},
                "isActive": True,
            },
        )

        if user_credit:
            credit_sum = user_credit[0].get("_sum") or {}
            return credit_sum.get("amount", 0)

        key = f"MONTHLY-CREDIT-TOP-UP-{cur_month}"

        try:
            await UserBlockCredit.prisma().create(
                data={
                    "amount": self.num_user_credits_refill,
                    "type": UserBlockCreditType.TOP_UP,
                    "userId": user_id,
                    "transactionKey": key,
                    "createdAt": self.time_now(),
                }
            )
        except prisma.errors.UniqueViolationError:
            pass  # Already refilled this month

        return self.num_user_credits_refill

    @staticmethod
    def time_now():
        return datetime.now(timezone.utc)

    @staticmethod
    def _block_usage_cost(
        block: Block,
        input_data: BlockInput,
        data_size: float,
        run_time: float,
    ) -> tuple[int, BlockInput]:
        block_costs = BLOCK_COSTS.get(type(block))
        if not block_costs:
            return 0, {}

        for block_cost in block_costs:
            if all(
                # None, [], {}, "", are considered the same value.
                input_data.get(k) == b or (not input_data.get(k) and not b)
                for k, b in block_cost.cost_filter.items()
            ):
                if block_cost.cost_type == BlockCostType.RUN:
                    return block_cost.cost_amount, block_cost.cost_filter

                if block_cost.cost_type == BlockCostType.SECOND:
                    return (
                        int(run_time * block_cost.cost_amount),
                        block_cost.cost_filter,
                    )

                if block_cost.cost_type == BlockCostType.BYTE:
                    return (
                        int(data_size * block_cost.cost_amount),
                        block_cost.cost_filter,
                    )

        return 0, {}

    async def spend_credits(
        self,
        user_id: str,
        user_credit: int,
        block_id: str,
        input_data: BlockInput,
        data_size: float,
        run_time: float,
        validate_balance: bool = True,
    ) -> int:
        block = get_block(block_id)
        if not block:
            raise ValueError(f"Block not found: {block_id}")

        cost, matching_filter = self._block_usage_cost(
            block=block, input_data=input_data, data_size=data_size, run_time=run_time
        )
        if cost <= 0:
            return 0

        if validate_balance and user_credit < cost:
            raise ValueError(f"Insufficient credit: {user_credit} < {cost}")

        await UserBlockCredit.prisma().create(
            data={
                "userId": user_id,
                "amount": -cost,
                "type": UserBlockCreditType.USAGE,
                "blockId": block.id,
                "metadata": Json(
                    {
                        "block": block.name,
                        "input": matching_filter,
                    }
                ),
                "createdAt": self.time_now(),
            }
        )
        return cost

    async def top_up_credits(self, user_id: str, amount: int):
        await UserBlockCredit.prisma().create(
            data={
                "userId": user_id,
                "amount": amount,
                "type": UserBlockCreditType.TOP_UP,
                "createdAt": self.time_now(),
            }
        )


class DisabledUserCredit(UserCreditBase):
    async def get_or_refill_credit(self, *args, **kwargs) -> int:
        return 0

    async def spend_credits(self, *args, **kwargs) -> int:
        return 0

    async def top_up_credits(self, *args, **kwargs):
        pass


def get_user_credit_model() -> UserCreditBase:
    config = Config()
    if config.enable_credit.lower() == "true":
        return UserCredit(config.num_user_credits_refill)
    else:
        return DisabledUserCredit(0)


def get_block_costs() -> dict[str, list[BlockCost]]:
    return {block().id: costs for block, costs in BLOCK_COSTS.items()}
