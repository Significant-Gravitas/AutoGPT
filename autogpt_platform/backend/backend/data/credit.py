import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone

import stripe
from prisma import Json
from prisma.enums import CreditTransactionType
from prisma.errors import UniqueViolationError
from prisma.models import CreditTransaction, User
from prisma.types import CreditTransactionCreateInput, CreditTransactionWhereInput
from pydantic import BaseModel

from backend.data import db
from backend.data.block import Block, BlockInput, get_block
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.cost import BlockCost, BlockCostType
from backend.data.execution import NodeExecutionEntry
from backend.data.model import AutoTopUpConfig, TransactionHistory, UserTransaction
from backend.data.user import get_user_by_id
from backend.util.settings import Settings

settings = Settings()
stripe.api_key = settings.secrets.stripe_api_key
logger = logging.getLogger(__name__)


class UserCreditBase(ABC):
    @abstractmethod
    async def get_credits(self, user_id: str) -> int:
        """
        Get the current credits for the user.

        Returns:
            int: The current credits for the user.
        """
        pass

    @abstractmethod
    async def get_transaction_history(
        self,
        user_id: str,
        transaction_time: datetime,
        transaction_count_limit: int,
    ) -> TransactionHistory:
        """
        Get the credit transactions for the user.

        Args:
            user_id (str): The user ID.
            transaction_time (datetime): The upper bound of the transaction time.
            transaction_count_limit (int): The transaction count limit.

        Returns:
            TransactionHistory: The credit transactions for the user.
        """
        pass

    @abstractmethod
    async def spend_credits(
        self,
        entry: NodeExecutionEntry,
        data_size: float,
        run_time: float,
    ) -> int:
        """
        Spend the credits for the user based on the block usage.

        Args:
            entry (NodeExecutionEntry): The node execution identifiers & data.
            data_size (float): The size of the data being processed.
            run_time (float): The time taken to run the block.

        Returns:
            int: amount of credit spent
        """
        pass

    @abstractmethod
    async def top_up_credits(self, user_id: str, amount: int):
        """
        Top up the credits for the user immediately.

        Args:
            user_id (str): The user ID.
            amount (int): The amount to top up.
        """
        pass

    @abstractmethod
    async def top_up_intent(self, user_id: str, amount: int) -> str:
        """
        Create a payment intent to top up the credits for the user.

        Args:
            user_id (str): The user ID.
            amount (int): The amount of credits to top up.

        Returns:
            str: The redirect url to the payment page.
        """
        pass

    @abstractmethod
    async def fulfill_checkout(
        self, *, session_id: str | None = None, user_id: str | None = None
    ):
        """
        Fulfill the Stripe checkout session.

        Args:
            session_id (str | None): The checkout session ID. Will try to fulfill most recent if None.
            user_id (str | None): The user ID must be provided if session_id is None.
        """
        pass

    @staticmethod
    def time_now() -> datetime:
        return datetime.now(timezone.utc)

    # ====== Transaction Helper Methods ====== #
    # Any modifications to the transaction table should only be done through these methods #

    async def _get_credits(self, user_id: str) -> tuple[int, datetime]:
        """
        Returns the current balance of the user & the latest balance snapshot time.
        """
        top_time = self.time_now()
        snapshot = await CreditTransaction.prisma().find_first(
            where={
                "userId": user_id,
                "createdAt": {"lte": top_time},
                "isActive": True,
                "runningBalance": {"not": None},  # type: ignore
            },
            order={"createdAt": "desc"},
        )
        datetime_min = datetime.min.replace(tzinfo=timezone.utc)
        snapshot_balance = snapshot.runningBalance or 0 if snapshot else 0
        snapshot_time = snapshot.createdAt if snapshot else datetime_min

        # Get transactions after the snapshot, this should not exist, but just in case.
        transactions = await CreditTransaction.prisma().group_by(
            by=["userId"],
            sum={"amount": True},
            max={"createdAt": True},
            where={
                "userId": user_id,
                "createdAt": {
                    "gt": snapshot_time,
                    "lte": top_time,
                },
                "isActive": True,
            },
        )
        transaction_balance = (
            int(transactions[0].get("_sum", {}).get("amount", 0) + snapshot_balance)
            if transactions
            else snapshot_balance
        )
        transaction_time = (
            datetime.fromisoformat(
                str(transactions[0].get("_max", {}).get("createdAt", datetime_min))
            )
            if transactions
            else snapshot_time
        )
        return transaction_balance, transaction_time

    async def _enable_transaction(
        self, transaction_key: str, user_id: str, metadata: Json
    ):

        transaction = await CreditTransaction.prisma().find_first_or_raise(
            where={"transactionKey": transaction_key, "userId": user_id}
        )

        if transaction.isActive:
            return

        async with db.locked_transaction(f"usr_trx_{user_id}"):
            user_balance, _ = await self._get_credits(user_id)
            await CreditTransaction.prisma().update(
                where={
                    "creditTransactionIdentifier": {
                        "transactionKey": transaction_key,
                        "userId": user_id,
                    }
                },
                data={
                    "isActive": True,
                    "runningBalance": user_balance + transaction.amount,
                    "createdAt": self.time_now(),
                    "metadata": metadata,
                },
            )

    async def _add_transaction(
        self,
        user_id: str,
        amount: int,
        transaction_type: CreditTransactionType,
        is_active: bool = True,
        transaction_key: str | None = None,
        ceiling_balance: int | None = None,
        metadata: Json = Json({}),
    ) -> tuple[int, str]:
        """
        Add a new transaction for the user.
        This is the only method that should be used to add a new transaction.

        Args:
            user_id (str): The user ID.
            amount (int): The amount of credits to add.
            transaction_type (CreditTransactionType): The type of transaction.
            is_active (bool): Whether the transaction is active or needs to be manually activated through _enable_transaction.
            transaction_key (str | None): The transaction key. Avoids adding transaction if the key already exists.
            ceiling_balance (int | None): The ceiling balance. Avoids adding more credits if the balance is already above the ceiling.
            metadata (Json): The metadata of the transaction.

        Returns:
            tuple[int, str]: The new balance & the transaction key.
        """
        async with db.locked_transaction(f"usr_trx_{user_id}"):
            # Get latest balance snapshot
            user_balance, _ = await self._get_credits(user_id)

            if ceiling_balance and user_balance >= ceiling_balance:
                raise ValueError(
                    f"You already have enough balance for user {user_id}, balance: {user_balance}, ceiling: {ceiling_balance}"
                )

            if amount < 0 and user_balance < abs(amount):
                raise ValueError(
                    f"Insufficient balance of ${user_balance/100} to run the block that costs ${abs(amount)/100}"
                )

            # Create the transaction
            transaction_data: CreditTransactionCreateInput = {
                "userId": user_id,
                "amount": amount,
                "runningBalance": user_balance + amount,
                "type": transaction_type,
                "metadata": metadata,
                "isActive": is_active,
                "createdAt": self.time_now(),
            }
            if transaction_key:
                transaction_data["transactionKey"] = transaction_key
            tx = await CreditTransaction.prisma().create(data=transaction_data)
            return user_balance + amount, tx.transactionKey


class UsageTransactionMetadata(BaseModel):
    graph_exec_id: str | None = None
    graph_id: str | None = None
    node_id: str | None = None
    node_exec_id: str | None = None
    block_id: str | None = None
    block: str | None = None
    input: BlockInput | None = None


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
          - If cost_filter is an object, then check if cost_filter is the subset of input_data
          - Otherwise, check if cost_filter is equal to input_data.
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
        entry: NodeExecutionEntry,
        data_size: float,
        run_time: float,
    ) -> int:
        block = get_block(entry.block_id)
        if not block:
            raise ValueError(f"Block not found: {entry.block_id}")

        cost, matching_filter = self._block_usage_cost(
            block=block, input_data=entry.data, data_size=data_size, run_time=run_time
        )
        if cost == 0:
            return 0

        balance, _ = await self._add_transaction(
            user_id=entry.user_id,
            amount=-cost,
            transaction_type=CreditTransactionType.USAGE,
            metadata=Json(
                UsageTransactionMetadata(
                    graph_exec_id=entry.graph_exec_id,
                    graph_id=entry.graph_id,
                    node_id=entry.node_id,
                    node_exec_id=entry.node_exec_id,
                    block_id=entry.block_id,
                    block=block.name,
                    input=matching_filter,
                ).model_dump()
            ),
        )
        user_id = entry.user_id

        # Auto top-up if balance is below threshold.
        auto_top_up = await get_auto_top_up(user_id)
        if auto_top_up.threshold and balance < auto_top_up.threshold:
            try:
                await self._top_up_credits(
                    user_id=user_id,
                    amount=auto_top_up.amount,
                    # Avoid multiple auto top-ups within the same graph execution.
                    key=f"AUTO-TOP-UP-{user_id}-{entry.graph_exec_id}",
                    ceiling_balance=auto_top_up.threshold,
                )
            except Exception as e:
                # Failed top-up is not critical, we can move on.
                logger.error(
                    f"Auto top-up failed for user {user_id}, balance: {balance}, amount: {auto_top_up.amount}, error: {e}"
                )

        return cost

    async def top_up_credits(self, user_id: str, amount: int):
        await self._top_up_credits(user_id, amount)

    async def _top_up_credits(
        self,
        user_id: str,
        amount: int,
        key: str | None = None,
        ceiling_balance: int | None = None,
    ):
        if amount < 0:
            raise ValueError(f"Top up amount must not be negative: {amount}")

        if key is not None and (
            await CreditTransaction.prisma().find_first(
                where={"transactionKey": key, "userId": user_id}
            )
        ):
            raise ValueError(f"Transaction key {key} already exists for user {user_id}")

        _, transaction_key = await self._add_transaction(
            user_id=user_id,
            amount=amount,
            transaction_type=CreditTransactionType.TOP_UP,
            is_active=False,
            transaction_key=key,
            ceiling_balance=ceiling_balance,
        )

        customer_id = await get_stripe_customer_id(user_id)

        payment_methods = stripe.PaymentMethod.list(customer=customer_id, type="card")
        if not payment_methods:
            raise ValueError("No payment method found, please add it on the platform.")

        for payment_method in payment_methods:
            if amount == 0:
                setup_intent = stripe.SetupIntent.create(
                    customer=customer_id,
                    usage="off_session",
                    confirm=True,
                    payment_method=payment_method.id,
                    automatic_payment_methods={
                        "enabled": True,
                        "allow_redirects": "never",
                    },
                )
                if setup_intent.status == "succeeded":
                    return

            else:
                payment_intent = stripe.PaymentIntent.create(
                    amount=amount,
                    currency="usd",
                    description="AutoGPT Platform Credits",
                    customer=customer_id,
                    off_session=True,
                    confirm=True,
                    payment_method=payment_method.id,
                    automatic_payment_methods={
                        "enabled": True,
                        "allow_redirects": "never",
                    },
                )
                if payment_intent.status == "succeeded":
                    await self._enable_transaction(
                        transaction_key=transaction_key,
                        user_id=user_id,
                        metadata=Json({"payment_intent": payment_intent}),
                    )
                    return

        raise ValueError(
            f"Out of {len(payment_methods)} payment methods tried, none is supported"
        )

    async def top_up_intent(self, user_id: str, amount: int) -> str:
        if amount < 500 or amount % 100 != 0:
            raise ValueError(
                f"Top up amount must be at least 500 credits and multiple of 100 but is {amount}"
            )

        # Create checkout session
        # https://docs.stripe.com/checkout/quickstart?client=react
        # unit_amount param is always in the smallest currency unit (so cents for usd)
        # which is equal to amount of credits
        checkout_session = stripe.checkout.Session.create(
            customer=await get_stripe_customer_id(user_id),
            line_items=[
                {
                    "price_data": {
                        "currency": "usd",
                        "product_data": {
                            "name": "AutoGPT Platform Credits",
                        },
                        "unit_amount": amount,
                    },
                    "quantity": 1,
                }
            ],
            mode="payment",
            ui_mode="hosted",
            payment_intent_data={"setup_future_usage": "off_session"},
            saved_payment_method_options={"payment_method_save": "enabled"},
            success_url=settings.config.frontend_base_url
            + "/marketplace/credits?topup=success",
            cancel_url=settings.config.frontend_base_url
            + "/marketplace/credits?topup=cancel",
            allow_promotion_codes=True,
        )

        await self._add_transaction(
            user_id=user_id,
            amount=amount,
            transaction_type=CreditTransactionType.TOP_UP,
            transaction_key=checkout_session.id,
            is_active=False,
            metadata=Json({"checkout_session": checkout_session}),
        )

        return checkout_session.url or ""

    # https://docs.stripe.com/checkout/fulfillment
    async def fulfill_checkout(
        self, *, session_id: str | None = None, user_id: str | None = None
    ):
        if (not session_id and not user_id) or (session_id and user_id):
            raise ValueError("Either session_id or user_id must be provided")

        # Retrieve CreditTransaction
        find_filter: CreditTransactionWhereInput = {
            "type": CreditTransactionType.TOP_UP,
            "isActive": False,
        }
        if session_id:
            find_filter["transactionKey"] = session_id
        if user_id:
            find_filter["userId"] = user_id

        # Find the most recent inactive top-up transaction
        credit_transaction = await CreditTransaction.prisma().find_first(
            where=find_filter,
            order={"createdAt": "desc"},
        )

        # This can be called multiple times for one id, so ignore if already fulfilled
        if not credit_transaction:
            return

        # Retrieve the Checkout Session from the API
        checkout_session = stripe.checkout.Session.retrieve(
            credit_transaction.transactionKey
        )

        # Check the Checkout Session's payment_status property
        # to determine if fulfillment should be performed
        if checkout_session.payment_status in ["paid", "no_payment_required"]:
            await self._enable_transaction(
                transaction_key=credit_transaction.transactionKey,
                user_id=credit_transaction.userId,
                metadata=Json({"checkout_session": checkout_session}),
            )

    async def get_credits(self, user_id: str) -> int:
        balance, _ = await self._get_credits(user_id)
        return balance

    async def get_transaction_history(
        self,
        user_id: str,
        transaction_time: datetime,
        transaction_count_limit: int,
    ) -> TransactionHistory:
        transactions = await CreditTransaction.prisma().find_many(
            where={
                "userId": user_id,
                "createdAt": {"lt": transaction_time},
                "isActive": True,
            },
            order={"createdAt": "desc"},
            take=transaction_count_limit,
        )

        grouped_transactions: dict[str, UserTransaction] = defaultdict(
            lambda: UserTransaction()
        )
        tx_time = None
        for t in transactions:
            metadata = (
                UsageTransactionMetadata.model_validate(t.metadata)
                if t.metadata
                else UsageTransactionMetadata()
            )
            tx_time = t.createdAt.replace(tzinfo=None)

            if t.type == CreditTransactionType.USAGE and metadata.graph_exec_id:
                gt = grouped_transactions[metadata.graph_exec_id]
                gid = metadata.graph_id[:8] if metadata.graph_id else "UNKNOWN"
                gt.description = f"Graph #{gid} Execution"

                gt.usage_node_count += 1
                gt.usage_start_time = min(gt.usage_start_time, tx_time)
                gt.usage_execution_id = metadata.graph_exec_id
                gt.usage_graph_id = metadata.graph_id
            else:
                gt = grouped_transactions[t.transactionKey]
                gt.description = f"{t.type} Transaction"

            gt.amount += t.amount
            gt.transaction_type = t.type

            if tx_time > gt.transaction_time:
                gt.transaction_time = tx_time
                gt.balance = t.runningBalance or 0

        return TransactionHistory(
            transactions=list(grouped_transactions.values()),
            next_transaction_time=(
                tx_time if len(transactions) == transaction_count_limit else None
            ),
        )


class BetaUserCredit(UserCredit):
    """
    This is a temporary class to handle the test user utilizing monthly credit refill.
    TODO: Remove this class & its feature toggle.
    """

    def __init__(self, num_user_credits_refill: int):
        self.num_user_credits_refill = num_user_credits_refill

    async def get_credits(self, user_id: str) -> int:
        cur_time = self.time_now().date()
        balance, snapshot_time = await self._get_credits(user_id)
        if (snapshot_time.year, snapshot_time.month) == (cur_time.year, cur_time.month):
            return balance

        try:
            balance, _ = await self._add_transaction(
                user_id=user_id,
                amount=max(self.num_user_credits_refill - balance, 0),
                transaction_type=CreditTransactionType.TOP_UP,
                transaction_key=f"MONTHLY-CREDIT-TOP-UP-{cur_time}",
            )
            return balance
        except UniqueViolationError:
            # Already refilled this month
            return (await self._get_credits(user_id))[0]


class DisabledUserCredit(UserCreditBase):
    async def get_credits(self, *args, **kwargs) -> int:
        return 0

    async def get_transaction_history(self, *args, **kwargs) -> TransactionHistory:
        return TransactionHistory(transactions=[], next_transaction_time=None)

    async def spend_credits(self, *args, **kwargs) -> int:
        return 0

    async def top_up_credits(self, *args, **kwargs):
        pass

    async def top_up_intent(self, *args, **kwargs) -> str:
        return ""

    async def fulfill_checkout(self, *args, **kwargs):
        pass


def get_user_credit_model() -> UserCreditBase:
    if not settings.config.enable_credit:
        return DisabledUserCredit()

    if settings.config.enable_beta_monthly_credit:
        return BetaUserCredit(settings.config.num_user_credits_refill)

    return UserCredit()


def get_block_costs() -> dict[str, list[BlockCost]]:
    return {block().id: costs for block, costs in BLOCK_COSTS.items()}


async def get_stripe_customer_id(user_id: str) -> str:
    user = await get_user_by_id(user_id)

    if user.stripeCustomerId:
        return user.stripeCustomerId

    customer = stripe.Customer.create(name=user.name or "", email=user.email)
    await User.prisma().update(
        where={"id": user_id}, data={"stripeCustomerId": customer.id}
    )
    return customer.id


async def set_auto_top_up(user_id: str, config: AutoTopUpConfig):
    await User.prisma().update(
        where={"id": user_id},
        data={"topUpConfig": Json(config.model_dump())},
    )


async def get_auto_top_up(user_id: str) -> AutoTopUpConfig:
    user = await get_user_by_id(user_id)

    if not user.topUpConfig:
        return AutoTopUpConfig(threshold=0, amount=0)

    return AutoTopUpConfig.model_validate(user.topUpConfig)
