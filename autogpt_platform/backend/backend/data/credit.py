from abc import ABC, abstractmethod
from datetime import datetime, timezone

from prisma import Json
from prisma.enums import CreditTransactionType
from prisma.errors import UniqueViolationError
from prisma.models import CreditTransaction

from backend.data.block import Block, BlockInput, get_block
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.cost import BlockCost, BlockCostType
from backend.util.settings import Config

config = Config()


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
        nxt_month = (
            cur_month.replace(month=cur_month.month + 1)
            if cur_month.month < 12
            else cur_month.replace(year=cur_month.year + 1, month=1)
        )

        user_credit = await CreditTransaction.prisma().group_by(
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
            await CreditTransaction.prisma().create(
                data={
                    "amount": self.num_user_credits_refill,
                    "type": CreditTransactionType.TOP_UP,
                    "userId": user_id,
                    "transactionKey": key,
                    "createdAt": self.time_now(),
                }
            )
        except UniqueViolationError:
            pass  # Already refilled this month

        return self.num_user_credits_refill

    @staticmethod
    def time_now():
        return datetime.now(timezone.utc)

    def _block_usage_cost(
        self,
        block: Block,
        input_data: BlockInput,
        data_size: float,
        run_time: float,
    ) -> tuple[int, BlockInput]:
        block_costs = BLOCK_COSTS.get(type(block))
        if not block_costs:
            return 0, {}

        for block_cost in block_costs:
            if not self._is_cost_filter_match(block_cost.cost_filter, input_data):
                continue

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

    def _is_cost_filter_match(
        self, cost_filter: BlockInput, input_data: BlockInput
    ) -> bool:
        """
        Filter rules:
          - If costFilter is an object, then check if costFilter is the subset of inputValues
          - Otherwise, check if costFilter is equal to inputValues.
          - Undefined, null, and empty string are considered as equal.
        """
        if not isinstance(cost_filter, dict) or not isinstance(input_data, dict):
            return cost_filter == input_data

        return all(
            (not input_data.get(k) and not v)
            or (input_data.get(k) and self._is_cost_filter_match(v, input_data[k]))
            for k, v in cost_filter.items()
        )

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

        await CreditTransaction.prisma().create(
            data={
                "userId": user_id,
                "amount": -cost,
                "type": CreditTransactionType.USAGE,
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
        await CreditTransaction.prisma().create(
            data={
                "userId": user_id,
                "amount": amount,
                "type": CreditTransactionType.TOP_UP,
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
    if config.enable_credit.lower() == "true":
        return UserCredit(config.num_user_credits_refill)
    else:
        return DisabledUserCredit(0)


def get_block_costs() -> dict[str, list[BlockCost]]:
    return {block().id: costs for block, costs in BLOCK_COSTS.items()}
