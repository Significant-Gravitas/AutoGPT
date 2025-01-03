from abc import ABC, abstractmethod
from datetime import datetime, timezone

import stripe
from prisma import Json
from prisma.enums import CreditTransactionType
from prisma.errors import UniqueViolationError
from prisma.models import CreditTransaction, User

from backend.data.block import Block, BlockInput, get_block
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.cost import BlockCost, BlockCostType
from backend.data.user import get_user_by_id
from backend.util.settings import Settings

settings = Settings()
stripe.api_key = settings.secrets.stripe_api_key


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
    async def fulfill_checkout(self, *, session_id: str | None = None, user_id: str | None = None):
        """
        Fulfill the Stripe checkout session.

        Args:
            session_id (str | None): The checkout session ID. Will try to fulfill most recent if None.
            user_id (str | None): The user ID must be provided if session_id is None.
        """
        pass


class UserCredit(UserCreditBase):
    def __init__(self):
        self.num_user_credits_refill = settings.config.num_user_credits_refill

    async def get_credits(self, user_id: str) -> int:
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
        if amount < 0:
            raise ValueError(f"Top up amount must not be negative: {amount}")

        await CreditTransaction.prisma().create(
            data={
                "userId": user_id,
                "amount": amount,
                "isActive": True,
                "type": CreditTransactionType.TOP_UP,
                "createdAt": self.time_now(),
            }
        )

    async def top_up_intent(self, user_id: str, amount: int) -> str:
        user = await get_user_by_id(user_id)

        if not user:
            raise ValueError(f"User not found: {user_id}")

        # Create customer if not exists
        if not user.stripeCustomerId:
            customer = stripe.Customer.create(name=user.name or "", email=user.email)
            await User.prisma().update(
                where={"id": user_id}, data={"stripeCustomerId": customer.id}
            )
            user.stripeCustomerId = customer.id

        # Create checkout session
        # https://docs.stripe.com/checkout/quickstart?client=react
        # unit_amount param is always in the smallest currency unit (so cents for usd)
        # which equals to amount of credits
        checkout_session = stripe.checkout.Session.create(
            customer=user.stripeCustomerId,
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
            success_url=settings.config.platform_base_url
            + "/store/credits?topup=success",
            cancel_url=settings.config.platform_base_url
            + "/store/credits?topup=cancel",
        )

        # Create pending transaction
        await CreditTransaction.prisma().create(
            data={
                "transactionKey": checkout_session.id,
                "userId": user_id,
                "amount": amount,
                "type": CreditTransactionType.TOP_UP,
                "isActive": False,
                "metadata": Json({"checkout_session": checkout_session}),
            }
        )

        return checkout_session.url or ""

    # https://docs.stripe.com/checkout/fulfillment
    async def fulfill_checkout(self, *, session_id: str | None = None, user_id: str | None = None):
        if (not session_id and not user_id) or (session_id and user_id):
            raise ValueError("Either session_id or user_id must be provided")
        
        # Retrieve CreditTransaction
        credit_transaction = await CreditTransaction.prisma().find_first_or_raise(
            where={
                "OR": [
                    {"transactionKey": session_id} if session_id is not None else {"transactionKey": ""},
                    {"userId": user_id} if user_id is not None else {"userId": ""}
                ]
            },
            order={
                "createdAt": "desc"
            }
        )

        # This can be called multiple times for one id, so ignore if already fulfilled
        if credit_transaction.isActive:
            return

        # Retrieve the Checkout Session from the API
        checkout_session = stripe.checkout.Session.retrieve(credit_transaction.transactionKey)

        # Check the Checkout Session's payment_status property
        # to determine if fulfillment should be peformed
        if checkout_session.payment_status in ["paid", "no_payment_required"]:
            # Activate the CreditTransaction
            await CreditTransaction.prisma().update(
                where={
                    "creditTransactionIdentifier": {
                        "transactionKey": credit_transaction.transactionKey,
                        "userId": credit_transaction.userId,
                    }
                },
                data={
                    "isActive": True,
                    "createdAt": self.time_now(),
                    "metadata": Json({"checkout_session": checkout_session}),
                },
            )


class DisabledUserCredit(UserCreditBase):
    async def get_credits(self, *args, **kwargs) -> int:
        return 0

    async def spend_credits(self, *args, **kwargs) -> int:
        return 0

    async def top_up_credits(self, *args, **kwargs):
        pass

    async def top_up_intent(self, *args, **kwargs) -> str:
        return ""

    async def fulfill_checkout(self, *args, **kwargs):
        pass


def get_user_credit_model() -> UserCreditBase:
    # return UserCredit()
    if settings.config.enable_credit.lower() == "true":
        return UserCredit()
    else:
        return DisabledUserCredit()


def get_block_costs() -> dict[str, list[BlockCost]]:
    return {block().id: costs for block, costs in BLOCK_COSTS.items()}
