from abc import ABC, abstractmethod
from datetime import datetime, timezone

from prisma import Json
from prisma.enums import CreditTransactionType
from prisma.errors import UniqueViolationError
from prisma.models import CreditTransaction
from prisma.types import CreditTransactionCreateInput

from backend.data import db
from backend.data.block import Block, BlockInput, get_block
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.cost import BlockCost, BlockCostType
from backend.util.settings import Config

config = Config()


class UserCreditBase(ABC):
    def __init__(self, num_user_credits_refill: int):
        self.num_user_credits_refill = num_user_credits_refill

    @abstractmethod
    async def get_balance(self, user_id: str) -> int:
        """
        Get the current credit for the user.

        Returns:
            int: The current credit for the user.
        """
        pass

    @abstractmethod
    async def spend_credits(
        self,
        user_id: str,
        block_id: str,
        input_data: BlockInput,
        data_size: float,
        run_time: float,
    ) -> int:
        """
        Spend the credits for the user based on the block usage.

        Args:
            user_id (str): The user ID.
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

    @staticmethod
    async def _get_balance(user_id: str, end_time: datetime = datetime.max) -> int:
        # Find the latest captured balance snapshot.
        snapshot = await CreditTransaction.prisma().find_first(
            where={
                "userId": user_id,
                "createdAt": {"lt": end_time},
                "isActive": True,
                "runningBalance": {"not": None},  # type: ignore
            },
            order={"createdAt": "desc"},
        )
        snapshot_balance = (snapshot.runningBalance or 0) if snapshot else 0
        begin_time = snapshot.createdAt if snapshot else datetime.min

        # Sum all transactions after the last captured balance snapshot.
        transactions = await CreditTransaction.prisma().group_by(
            by=["userId"],
            sum={"amount": True},
            where={
                "userId": user_id,
                "createdAt": {"gt": begin_time, "lt": end_time},
                "isActive": True,
            },
        )
        transaction_balance = (
            transactions[0].get("_sum", {}).get("amount", 0) if transactions else 0
        )

        return snapshot_balance + transaction_balance

    async def _add_transaction(
        self,
        user_id: str,
        amount: int,
        transaction_type: CreditTransactionType,
        transaction_key: str | None = None,
        block_id: str | None = None,
        metadata: Json = Json({}),
    ):
        async with db.transaction() as tx:

            # Acquire lock on latest balance snapshot
            await tx.execute_raw(
                f"""SELECT *
                FROM platform."CreditTransaction"
                WHERE "userId" = '{user_id}'
                AND "isActive" = true
                ORDER BY "createdAt" DESC
                FOR UPDATE
            """
            )

            # Lock latest balance snapshot
            user_balance = await self._get_balance(user_id)
            if amount < 0 and user_balance < abs(amount):
                raise ValueError(
                    f"Insufficient balance for user {user_id}, balance: {user_balance}, amount: {amount}"
                )

            # Create the transaction
            transaction_data: CreditTransactionCreateInput = {
                "userId": user_id,
                "amount": amount,
                "runningBalance": user_balance + amount,
                "type": transaction_type,
                "blockId": block_id,
                "metadata": metadata,
            }
            if transaction_key:
                transaction_data["transactionKey"] = transaction_key
            await CreditTransaction.prisma().create(data=transaction_data)

            return user_balance + amount


class UserCredit(UserCreditBase):

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
        block_id: str,
        input_data: BlockInput,
        data_size: float,
        run_time: float,
    ) -> int:
        block = get_block(block_id)
        if not block:
            raise ValueError(f"Block not found: {block_id}")

        cost, matching_filter = self._block_usage_cost(
            block=block, input_data=input_data, data_size=data_size, run_time=run_time
        )
        if cost == 0:
            return 0

        await self._add_transaction(
            user_id=user_id,
            amount=-cost,
            transaction_type=CreditTransactionType.USAGE,
            block_id=block.id,
            metadata=Json(
                {
                    "block": block.name,
                    "input": matching_filter,
                }
            ),
        )

        return cost

    async def top_up_credits(self, user_id: str, amount: int):
        await self._add_transaction(
            user_id=user_id,
            amount=amount,
            transaction_type=CreditTransactionType.TOP_UP,
        )

    async def get_balance(self, user_id: str) -> int:
        return await self._get_balance(user_id)


class BetaUserCredit(UserCredit):

    @staticmethod
    def time_now():
        return datetime.now(timezone.utc)

    async def get_balance(self, user_id: str) -> int:
        cur_time = self.time_now()
        cur_month = cur_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        nxt_month = (
            cur_month.replace(month=cur_month.month + 1)
            if cur_month.month < 12
            else cur_month.replace(year=cur_month.year + 1, month=1)
        )

        curr_month_balance = await self._get_balance(user_id, nxt_month)
        prev_month_balance = await self._get_balance(user_id, cur_month)
        balance = curr_month_balance - prev_month_balance
        if balance != 0:
            return balance

        try:
            await self._add_transaction(
                user_id=user_id,
                amount=self.num_user_credits_refill,
                transaction_type=CreditTransactionType.TOP_UP,
                transaction_key=f"MONTHLY-CREDIT-TOP-UP-{cur_month}",
            )
        except UniqueViolationError:
            pass  # Already refilled this month

        return self.num_user_credits_refill


class DisabledUserCredit(UserCreditBase):
    async def get_balance(self, *args, **kwargs) -> int:
        return 0

    async def spend_credits(self, *args, **kwargs) -> int:
        return 0

    async def top_up_credits(self, *args, **kwargs):
        pass


def get_user_credit_model() -> UserCreditBase:
    if not config.enable_credit:
        return DisabledUserCredit(0)

    if config.enable_beta_monthly_credit:
        return BetaUserCredit(config.num_user_credits_refill)

    return UserCredit(config.num_user_credits_refill)


def get_block_costs() -> dict[str, list[BlockCost]]:
    return {block().id: costs for block, costs in BLOCK_COSTS.items()}
