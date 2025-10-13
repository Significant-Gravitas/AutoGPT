import logging
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

import stripe
from prisma import Json
from prisma.enums import (
    CreditRefundRequestStatus,
    CreditTransactionType,
    NotificationType,
    OnboardingStep,
)
from prisma.errors import UniqueViolationError
from prisma.models import CreditRefundRequest, CreditTransaction, User
from prisma.types import (
    CreditRefundRequestCreateInput,
    CreditTransactionCreateInput,
    CreditTransactionWhereInput,
)
from pydantic import BaseModel

from backend.data import db
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.includes import MAX_CREDIT_REFUND_REQUESTS_FETCH
from backend.data.model import (
    AutoTopUpConfig,
    RefundRequest,
    TopUpType,
    TransactionHistory,
    UserTransaction,
)
from backend.data.notifications import NotificationEventModel, RefundRequestData
from backend.data.user import get_user_by_id, get_user_email_by_id
from backend.notifications.notifications import queue_notification_async
from backend.server.v2.admin.model import UserHistoryResponse
from backend.util.exceptions import InsufficientBalanceError
from backend.util.json import SafeJson
from backend.util.models import Pagination
from backend.util.retry import func_retry
from backend.util.settings import Settings

if TYPE_CHECKING:
    from backend.data.block import Block, BlockCost

settings = Settings()
stripe.api_key = settings.secrets.stripe_api_key
logger = logging.getLogger(__name__)
base_url = settings.config.frontend_base_url or settings.config.platform_base_url

# PostgreSQL INT type maximum value
POSTGRES_INT_MAX = 2147483647


def format_credits_as_dollars(credits: int) -> str:
    """Format credits (stored as cents) as dollar amount string."""
    return f"${credits/100:.2f}"


class UsageTransactionMetadata(BaseModel):
    graph_exec_id: str | None = None
    graph_id: str | None = None
    node_id: str | None = None
    node_exec_id: str | None = None
    block_id: str | None = None
    block: str | None = None
    input: dict[str, Any] | None = None
    reason: str | None = None


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
        transaction_count_limit: int,
        transaction_time_ceiling: datetime | None = None,
        transaction_type: str | None = None,
    ) -> TransactionHistory:
        """
        Get the credit transactions for the user.

        Args:
            user_id (str): The user ID.
            transaction_count_limit (int): The transaction count limit.
            transaction_time_ceiling (datetime): The upper bound of the transaction time.
            transaction_type (str): The transaction type filter.

        Returns:
            TransactionHistory: The credit transactions for the user.
        """
        pass

    @abstractmethod
    async def get_refund_requests(self, user_id: str) -> list[RefundRequest]:
        """
        Get the refund requests for the user.

        Args:
            user_id (str): The user ID.

        Returns:
            list[RefundRequest]: The refund requests for the user.
        """
        pass

    @abstractmethod
    async def spend_credits(
        self,
        user_id: str,
        cost: int,
        metadata: UsageTransactionMetadata,
    ) -> int:
        """
        Spend the credits for the user based on the cost.

        Args:
            user_id (str): The user ID.
            cost (int): The cost to spend.
            metadata (UsageTransactionMetadata): The metadata of the transaction.

        Returns:
            int: The remaining balance.
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

    @abstractmethod
    async def onboarding_reward(self, user_id: str, credits: int, step: OnboardingStep):
        """
        Reward the user with credits for completing an onboarding step.
        Won't reward if the user has already received credits for the step.

        Args:
            user_id (str): The user ID.
            step (OnboardingStep): The onboarding step.
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
    async def top_up_refund(
        self, user_id: str, transaction_key: str, metadata: dict[str, str]
    ) -> int:
        """
        Refund the top-up transaction for the user.

        Args:
            user_id (str): The user ID.
            transaction_key (str): The top-up transaction key to refund.
            metadata (dict[str, str]): The metadata of the refund.

        Returns:
            int: The amount refunded.
        """
        pass

    @abstractmethod
    async def deduct_credits(
        self,
        request: stripe.Refund | stripe.Dispute,
    ):
        """
        Deduct the credits for the user based on the dispute or refund of the top-up.

        Args:
            request (stripe.Refund | stripe.Dispute): The refund or dispute request.
        """
        pass

    @abstractmethod
    async def handle_dispute(self, dispute: stripe.Dispute):
        """
        Handle the dispute for the user based on the dispute request.

        Args:
            dispute (stripe.Dispute): The dispute request.
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
    async def create_billing_portal_session(user_id: str) -> str:
        session = stripe.billing_portal.Session.create(
            customer=await get_stripe_customer_id(user_id),
            return_url=base_url + "/profile/credits",
        )
        return session.url

    @staticmethod
    def time_now() -> datetime:
        return datetime.now(timezone.utc)

    # ====== Transaction Helper Methods ====== #
    # Any modifications to the transaction table should only be done through these methods #

    async def _get_credits(self, user_id: str) -> tuple[int, datetime]:
        """
        Returns the current balance of the user & the latest balance update time.
        Now uses the balance column for better performance.
        """
        user = await User.prisma().find_unique(where={"id": user_id})
        if not user:
            return 0, datetime.min.replace(tzinfo=timezone.utc)

        # Get the last transaction time for BetaUserCredit monthly refill check
        last_transaction = await CreditTransaction.prisma().find_first(
            where={
                "userId": user_id,
                "isActive": True,
            },
            order={"createdAt": "desc"},
        )

        transaction_time = (
            last_transaction.createdAt if last_transaction else user.updatedAt
        )
        return user.balance or 0, transaction_time

    @func_retry
    async def _enable_transaction(
        self,
        transaction_key: str,
        user_id: str,
        metadata: Json,
        new_transaction_key: str | None = None,
    ):
        """Enable a transaction and update user balance atomically.

        Used for Stripe payment processing to activate transactions after successful payment.
        Returns the transaction key if successful, raises error otherwise.
        """
        result = await db.prisma.query_raw(
            """
            WITH tx_activate AS (
                UPDATE "CreditTransaction"
                SET 
                    "isActive" = true,
                    "transactionKey" = COALESCE($1, "transactionKey"),
                    "metadata" = $2,
                    "createdAt" = CURRENT_TIMESTAMP
                WHERE 
                    "transactionKey" = $3
                    AND "userId" = $4
                    AND "isActive" = false
                RETURNING amount, "userId", COALESCE($1, "transactionKey") as new_key
            ),
            balance_update AS (
                UPDATE "User"
                SET 
                    balance = CASE 
                        WHEN balance + tx_activate.amount > $5 THEN $5
                        ELSE balance + tx_activate.amount
                    END,
                    "updatedAt" = CURRENT_TIMESTAMP
                FROM tx_activate
                WHERE "User".id = tx_activate."userId"
                RETURNING balance, "updatedAt", tx_activate.new_key
            )
            UPDATE "CreditTransaction"
            SET 
                "runningBalance" = balance_update.balance,
                "createdAt" = balance_update."updatedAt"
            FROM balance_update
            WHERE 
                "CreditTransaction"."transactionKey" = balance_update.new_key
                AND "CreditTransaction"."userId" = $4
            RETURNING "CreditTransaction"."transactionKey";
            """,
            new_transaction_key,
            metadata,
            transaction_key,
            user_id,
            POSTGRES_INT_MAX,
            POSTGRES_INT_MAX,
        )

        if not result:
            # Check if transaction exists and its status
            tx = await CreditTransaction.prisma().find_first(
                where={"transactionKey": transaction_key, "userId": user_id}
            )
            if not tx:
                raise ValueError(
                    f"Transaction {transaction_key} not found for user {user_id}"
                )
            elif tx.isActive:
                logger.warning(f"Transaction {transaction_key} is already active")
                return tx.transactionKey
            else:
                raise ValueError(f"Failed to activate transaction {transaction_key}")

        return result[0]["transactionKey"]

    # Used by BetaUserCredit for monthly credit refills
    # Also provides compatibility for legacy code paths
    async def _add_transaction(
        self,
        user_id: str,
        amount: int,
        transaction_type: CreditTransactionType,
        is_active: bool = True,
        transaction_key: str | None = None,
        ceiling_balance: int | None = None,
        fail_insufficient_credits: bool = True,
        metadata: Json = SafeJson({}),
    ) -> tuple[int, str]:
        """Used by BetaUserCredit for monthly credit refills and legacy compatibility."""
        # Validate input
        if amount > POSTGRES_INT_MAX or amount < -POSTGRES_INT_MAX:
            raise ValueError(f"Invalid amount: {amount}")

        # Check ceiling balance if specified
        if ceiling_balance and amount > 0:
            user = await User.prisma().find_unique(where={"id": user_id})
            if user and user.balance >= ceiling_balance:
                raise ValueError(
                    f"You already have enough balance of {format_credits_as_dollars(user.balance)}, top-up is not required when you already have at least {format_credits_as_dollars(ceiling_balance)}"
                )

        # For spending, check balance and use atomic operation
        if amount < 0 and fail_insufficient_credits:
            # Use atomic spend operation to prevent race conditions
            result = await db.prisma.query_raw(
                """
                WITH balance_update AS (
                    UPDATE "User" 
                    SET balance = balance + $1,
                        "updatedAt" = CURRENT_TIMESTAMP
                    WHERE id = $2 AND balance + $1 >= 0
                    RETURNING id, balance, "updatedAt"
                ),
                transaction_insert AS (
                    INSERT INTO "CreditTransaction" (
                        "transactionKey", "userId", "amount", "type",
                        "runningBalance", "isActive", "metadata", "createdAt"
                    )
                    SELECT 
                        COALESCE($3, gen_random_uuid()::text),
                        balance_update.id,
                        $1::int,
                        $4::"CreditTransactionType",
                        balance_update.balance,
                        $5::boolean,
                        $6::jsonb,
                        balance_update."updatedAt"
                    FROM balance_update
                    RETURNING "runningBalance", "transactionKey"
                )
                SELECT "runningBalance" as balance, "transactionKey" FROM transaction_insert;
                """,
                amount,
                user_id,
                transaction_key,
                transaction_type.value,
                is_active,
                metadata,
            )

            if not result:
                user = await User.prisma().find_unique(where={"id": user_id})
                if not user:
                    raise ValueError(f"User {user_id} not found")
                raise InsufficientBalanceError(
                    message=f"Insufficient balance of {format_credits_as_dollars(user.balance)}, where this will cost {format_credits_as_dollars(abs(amount))}",
                    user_id=user_id,
                    balance=user.balance,
                    amount=amount,
                )

            return result[0]["balance"], result[0]["transactionKey"]

        # For non-spending operations (top-ups, grants), use atomic operation
        result = await db.prisma.query_raw(
            """
            WITH balance_update AS (
                UPDATE "User"
                SET balance = balance + $2, "updatedAt" = CURRENT_TIMESTAMP
                WHERE id = $1
                RETURNING balance, "updatedAt"
            ),
            transaction_insert AS (
                INSERT INTO "CreditTransaction" (
                    "userId", "amount", "type", "runningBalance", 
                    "metadata", "isActive", "createdAt", "transactionKey"
                )
                SELECT 
                    $1, $2, $3::"CreditTransactionType", bu.balance, 
                    $4::jsonb, $5, bu."updatedAt", COALESCE($6, gen_random_uuid()::text)
                FROM balance_update bu
                RETURNING "runningBalance", "transactionKey"
            )
            SELECT "runningBalance" as balance, "transactionKey" FROM transaction_insert;
            """,
            user_id,
            amount,
            transaction_type.value,
            metadata,
            is_active,
            transaction_key,
        )

        if result:
            return result[0]["balance"], result[0]["transactionKey"]

        # If no result, create user with balance 0 and generate transaction
        await User.prisma().upsert(
            where={"id": user_id},
            data={
                "create": {
                    "id": user_id,
                    "email": f"{user_id}@example.com",
                    "name": user_id,
                    "balance": 0,
                },
                "update": {},
            },
        )
        # Return 0 balance with a generated transaction key
        return 0, transaction_key or str(uuid.uuid4())


class UserCredit(UserCreditBase):

    async def _send_refund_notification(
        self,
        notification_request: RefundRequestData,
        notification_type: NotificationType,
    ):
        await queue_notification_async(
            NotificationEventModel(
                user_id=notification_request.user_id,
                type=notification_type,
                data=notification_request,
            )
        )

    async def spend_credits(
        self,
        user_id: str,
        cost: int,
        metadata: UsageTransactionMetadata,
    ) -> int:
        if cost == 0:
            return 0

        # Validate input to prevent integer overflow
        if cost < 0 or cost > POSTGRES_INT_MAX:
            raise ValueError(f"Invalid cost amount: {cost}")

        # CRITICAL: Use a single atomic operation to update balance and create transaction
        # This prevents any possibility of partial updates
        # Use db module's connection to avoid connection pool issues
        result = await db.prisma.query_raw(
            """
            WITH balance_update AS (
                UPDATE "User" 
                SET balance = balance - $1, 
                    "updatedAt" = CURRENT_TIMESTAMP
                WHERE id = $2 AND balance >= $1
                RETURNING id, balance, "updatedAt"
            ),
            transaction_insert AS (
                INSERT INTO "CreditTransaction" (
                    "transactionKey", "userId", "amount", "type",
                    "runningBalance", "isActive", "metadata", "createdAt"
                )
                SELECT 
                    gen_random_uuid()::text,
                    balance_update.id,
                    -$1::int,
                    $3::"CreditTransactionType",
                    balance_update.balance,
                    true,
                    $4::jsonb,
                    balance_update."updatedAt"
                FROM balance_update
                RETURNING "runningBalance"
            )
            SELECT "runningBalance" as balance FROM transaction_insert;
            """,
            cost,
            user_id,
            CreditTransactionType.USAGE.value,
            SafeJson(metadata.model_dump()),
        )

        if not result:
            # Either user doesn't exist or insufficient balance
            user = await User.prisma().find_unique(where={"id": user_id})

            if not user:
                raise ValueError(f"User {user_id} not found")

            raise InsufficientBalanceError(
                message=f"Insufficient balance of {format_credits_as_dollars(user.balance)}, where this will cost {format_credits_as_dollars(cost)}",
                user_id=user_id,
                balance=user.balance,
                amount=-cost,
            )

        new_balance = result[0]["balance"]

        # Auto top-up if balance is below threshold.
        auto_top_up = await get_auto_top_up(user_id)
        if auto_top_up.threshold and new_balance < auto_top_up.threshold:
            try:
                await self._top_up_credits(
                    user_id=user_id,
                    amount=auto_top_up.amount,
                    # Avoid multiple auto top-ups within the same graph execution.
                    key=f"AUTO-TOP-UP-{user_id}-{metadata.graph_exec_id}",
                    ceiling_balance=auto_top_up.threshold,
                    top_up_type=TopUpType.AUTO,
                )
            except Exception as e:
                # Failed top-up is not critical, we can move on.
                logger.error(
                    f"Auto top-up failed for user {user_id}, balance: {new_balance}, amount: {auto_top_up.amount}, error: {e}"
                )

        return new_balance

    async def top_up_credits(
        self,
        user_id: str,
        amount: int,
        top_up_type: TopUpType = TopUpType.UNCATEGORIZED,
    ):
        # Validate input to prevent integer overflow
        if amount <= 0 or amount > POSTGRES_INT_MAX:
            raise ValueError(f"Invalid top-up amount: {amount}")

        # Use single atomic operation for consistency
        result = await db.prisma.query_raw(
            """
            WITH balance_check AS (
                SELECT id, balance
                FROM "User"
                WHERE id = $1
                FOR UPDATE
            ),
            balance_update AS (
                UPDATE "User" 
                SET balance = CASE 
                    WHEN balance + $2::int > $5 THEN $5
                    ELSE balance + $2::int
                END,
                "updatedAt" = CURRENT_TIMESTAMP
                WHERE id = $1
                RETURNING id, balance, "updatedAt"
            ),
            transaction_insert AS (
                INSERT INTO "CreditTransaction" (
                    "transactionKey", "userId", "amount", "type",
                    "runningBalance", "isActive", "metadata", "createdAt"
                )
                SELECT 
                    gen_random_uuid()::text,
                    balance_update.id,
                    $2::int,
                    $3::"CreditTransactionType",
                    balance_update.balance,
                    true,
                    $4::jsonb,
                    balance_update."updatedAt"
                FROM balance_update
                RETURNING "runningBalance"
            )
            SELECT "runningBalance" as balance FROM transaction_insert;
            """,
            user_id,
            amount,
            CreditTransactionType.TOP_UP.value,
            SafeJson(
                {
                    "reason": f"Manual top up credits for {user_id}",
                    "type": top_up_type.value,
                }
            ),
            POSTGRES_INT_MAX,
            POSTGRES_INT_MAX,
        )

        if not result:
            raise ValueError(f"User {user_id} not found")

        return result[0]["balance"]

    async def onboarding_reward(self, user_id: str, credits: int, step: OnboardingStep):
        # Validate input to prevent integer overflow
        if credits <= 0 or credits > POSTGRES_INT_MAX:
            raise ValueError(f"Invalid reward amount: {credits}")

        transaction_key = f"REWARD-{user_id}-{step.value}"

        # Single atomic operation that creates transaction and updates balance
        # Uses INSERT ON CONFLICT to handle duplicates atomically
        result = await db.prisma.query_raw(
            """
            WITH reward_insert AS (
                INSERT INTO "CreditTransaction" (
                    "transactionKey", "userId", "amount", "type",
                    "runningBalance", "isActive", "metadata", "createdAt"
                )
                SELECT 
                    $1, $2, $3, $4::"CreditTransactionType",
                    u.balance + $3, true, $5::jsonb, CURRENT_TIMESTAMP
                FROM "User" u
                WHERE u.id = $2
                ON CONFLICT ("transactionKey", "userId") DO NOTHING
                RETURNING "userId", "runningBalance"
            ),
            balance_update AS (
                UPDATE "User"
                SET 
                    balance = CASE
                        WHEN balance + $3 > 2147483647 THEN 2147483647
                        ELSE balance + $3
                    END,
                    "updatedAt" = CURRENT_TIMESTAMP
                WHERE id = $2 
                    AND EXISTS (SELECT 1 FROM reward_insert)
                RETURNING balance
            )
            SELECT 
                COALESCE(bu.balance, ri."runningBalance") as balance
            FROM reward_insert ri
            FULL OUTER JOIN balance_update bu ON true
            LIMIT 1;
            """,
            transaction_key,
            user_id,
            credits,
            CreditTransactionType.GRANT.value,
            SafeJson(
                {"reason": f"Reward for completing {step.value} onboarding step."}
            ),
        )

        # Result will be empty if duplicate (already rewarded)
        return result[0]["balance"] if result else None

    async def top_up_refund(
        self, user_id: str, transaction_key: str, metadata: dict[str, str]
    ) -> int:
        transaction = await CreditTransaction.prisma().find_first_or_raise(
            where={
                "transactionKey": transaction_key,
                "userId": user_id,
                "isActive": True,
                "type": CreditTransactionType.TOP_UP,
            }
        )
        balance = await self.get_credits(user_id)
        amount = transaction.amount
        refund_key_format = settings.config.refund_request_time_key_format
        refund_key = f"{transaction.createdAt.strftime(refund_key_format)}-{user_id}"

        try:
            refund_request = await CreditRefundRequest.prisma().create(
                data=CreditRefundRequestCreateInput(
                    id=refund_key,
                    transactionKey=transaction_key,
                    userId=user_id,
                    amount=amount,
                    reason=metadata.get("reason", ""),
                    status=CreditRefundRequestStatus.PENDING,
                    result="The refund request is under review.",
                )
            )
        except UniqueViolationError:
            raise ValueError(
                "Unable to request a refund for this transaction, the request of the top-up transaction within the same week has already been made."
            )

        if amount - balance > settings.config.refund_credit_tolerance_threshold:
            user_data = await get_user_by_id(user_id)
            await self._send_refund_notification(
                RefundRequestData(
                    user_id=user_id,
                    user_name=user_data.name or "AutoGPT Platform User",
                    user_email=user_data.email,
                    transaction_id=transaction_key,
                    refund_request_id=refund_request.id,
                    reason=refund_request.reason,
                    amount=amount,
                    balance=balance,
                ),
                NotificationType.REFUND_REQUEST,
            )
            return 0  # Register the refund request for manual approval.

        # Auto refund the top-up.
        refund = stripe.Refund.create(payment_intent=transaction_key, metadata=metadata)
        return refund.amount

    async def deduct_credits(self, request: stripe.Refund | stripe.Dispute):
        if isinstance(request, stripe.Refund) and request.status != "succeeded":
            logger.warning(
                f"Skip processing refund #{request.id} with status {request.status}"
            )
            return

        if isinstance(request, stripe.Dispute) and request.status != "lost":
            logger.warning(
                f"Skip processing dispute #{request.id} with status {request.status}"
            )
            return

        transaction = await CreditTransaction.prisma().find_first_or_raise(
            where={
                "transactionKey": str(request.payment_intent),
                "isActive": True,
                "type": CreditTransactionType.TOP_UP,
            }
        )
        if request.amount <= 0 or request.amount > transaction.amount:
            raise AssertionError(
                f"Invalid amount to deduct {format_credits_as_dollars(request.amount)} from {format_credits_as_dollars(transaction.amount)} top-up"
            )

        # Atomic balance deduction (allow going negative for refunds)
        result = await User.prisma().update(
            where={"id": transaction.userId},
            data={"balance": {"decrement": request.amount}},
        )

        if not result:
            raise ValueError(f"User {transaction.userId} not found")

        # Record the refund transaction
        await CreditTransaction.prisma().create(
            data={
                "userId": transaction.userId,
                "amount": -request.amount,
                "runningBalance": result.balance,
                "type": CreditTransactionType.REFUND,
                "transactionKey": request.id,
                "metadata": SafeJson(request),
                "isActive": True,
                "createdAt": result.updatedAt,
            }
        )

        balance = result.balance

        # Update the result of the refund request if it exists.
        await CreditRefundRequest.prisma().update_many(
            where={
                "userId": transaction.userId,
                "transactionKey": transaction.transactionKey,
            },
            data={
                "amount": request.amount,
                "status": CreditRefundRequestStatus.APPROVED,
                "result": "The refund request has been approved, the amount will be credited back to your account.",
            },
        )

        user_data = await get_user_by_id(transaction.userId)
        await self._send_refund_notification(
            RefundRequestData(
                user_id=user_data.id,
                user_name=user_data.name or "AutoGPT Platform User",
                user_email=user_data.email,
                transaction_id=transaction.transactionKey,
                refund_request_id=request.id,
                reason=str(request.reason or "-"),
                amount=transaction.amount,
                balance=balance,
            ),
            NotificationType.REFUND_PROCESSED,
        )

    async def handle_dispute(self, dispute: stripe.Dispute):
        transaction = await CreditTransaction.prisma().find_first_or_raise(
            where={
                "transactionKey": str(dispute.payment_intent),
                "isActive": True,
                "type": CreditTransactionType.TOP_UP,
            }
        )
        user_id = transaction.userId
        amount = dispute.amount
        balance = await self.get_credits(user_id)

        # If the user has enough balance, just let them win the dispute.
        if balance - amount >= settings.config.refund_credit_tolerance_threshold:
            logger.warning(
                f"Accepting dispute from {user_id} for {format_credits_as_dollars(amount)}"
            )
            dispute.close()
            return

        logger.warning(
            f"Adding extra info for dispute from {user_id} for {format_credits_as_dollars(amount)}"
        )
        # Retrieve recent transaction history to support our evidence.
        # This provides a concise timeline that shows service usage and proper credit application.
        transaction_history = await self.get_transaction_history(
            user_id, transaction_count_limit=None
        )
        user = await get_user_by_id(user_id)

        # Build a comprehensive explanation message that includes:
        # - Confirmation that the top-up transaction was processed and credits were applied.
        # - A summary of recent transaction history.
        # - An explanation that the funds were used to render the agreed service.
        evidence_text = (
            f"The top-up transaction of ${transaction.amount / 100:.2f} was processed successfully, and the corresponding credits "
            "were applied to the user’s account. Our records confirm that the funds were utilized for the intended services. "
            "Below is a summary of recent transaction activity:\n"
        )
        for tx in transaction_history.transactions:
            if tx.transaction_key == transaction.transactionKey:
                additional_comment = (
                    " [This top-up transaction is the subject of the dispute]."
                )
            else:
                additional_comment = ""

            evidence_text += (
                f"- {tx.description}: Amount ${tx.amount / 100:.2f} on {tx.transaction_time.isoformat()}, "
                f"resulting balance ${tx.running_balance / 100:.2f} {additional_comment}\n"
            )
        evidence_text += (
            "\nThis evidence demonstrates that the transaction was authorized and that the charged amount was used to render the service as agreed."
            "\nAdditionally, we provide an automated refund functionality, so the user could have used it if they were not satisfied with the service. "
        )
        evidence: stripe.Dispute.ModifyParamsEvidence = {
            "product_description": "AutoGPT Platform Credits",
            "customer_email_address": user.email,
            "uncategorized_text": evidence_text[:20000],
        }
        stripe.Dispute.modify(dispute.id, evidence=evidence)

    async def _top_up_credits(
        self,
        user_id: str,
        amount: int,
        key: str | None = None,
        ceiling_balance: int | None = None,
        top_up_type: TopUpType = TopUpType.UNCATEGORIZED,
        metadata: dict | None = None,
    ):
        # Validate amount to prevent integer overflow
        if amount < 0 or amount > POSTGRES_INT_MAX:
            raise ValueError(f"Invalid top up amount: {amount}")

        # init metadata, without sharing it with the world
        metadata = metadata or {}
        if not metadata.get("reason"):
            match top_up_type:
                case TopUpType.MANUAL:
                    metadata["reason"] = {"reason": f"Top up credits for {user_id}"}
                case TopUpType.AUTO:
                    metadata["reason"] = {
                        "reason": f"Auto top up credits for {user_id}"
                    }
                case _:
                    metadata["reason"] = {
                        "reason": f"Top up reason unknown for {user_id}"
                    }

        if amount == 0:
            transaction_type = CreditTransactionType.CARD_CHECK
        else:
            transaction_type = CreditTransactionType.TOP_UP

        # Create inactive transaction record for Stripe processing
        transaction_data: CreditTransactionCreateInput = {
            "userId": user_id,
            "amount": amount,
            "type": transaction_type,
            "metadata": SafeJson(metadata),
            "isActive": False,
            "createdAt": self.time_now(),
        }
        if key:
            transaction_data["transactionKey"] = key

        # Check for duplicate key
        if key and await CreditTransaction.prisma().find_first(
            where={"transactionKey": key, "userId": user_id}
        ):
            raise ValueError(f"Transaction key {key} already exists for user {user_id}")

        # Check ceiling balance if specified
        if ceiling_balance and amount > 0:
            user = await User.prisma().find_unique(where={"id": user_id})
            current_balance = user.balance if user else 0
            if current_balance >= ceiling_balance:
                raise ValueError(
                    f"You already have enough balance of {format_credits_as_dollars(current_balance)}, top-up is not required when you already have at least {format_credits_as_dollars(ceiling_balance)}"
                )

        tx = await CreditTransaction.prisma().create(data=transaction_data)
        transaction_key = tx.transactionKey

        customer_id = await get_stripe_customer_id(user_id)

        payment_methods = stripe.PaymentMethod.list(customer=customer_id, type="card")
        if not payment_methods:
            raise ValueError("No payment method found, please add it on the platform.")

        successful_transaction = None
        new_transaction_key = None
        for payment_method in payment_methods:
            if transaction_type == CreditTransactionType.CARD_CHECK:
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
                    successful_transaction = SafeJson({"setup_intent": setup_intent})
                    new_transaction_key = setup_intent.id
                    break
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
                    successful_transaction = SafeJson(
                        {"payment_intent": payment_intent}
                    )
                    new_transaction_key = payment_intent.id
                    break

        if not successful_transaction:
            raise ValueError(
                f"Out of {len(payment_methods)} payment methods tried, none is supported"
            )

        await self._enable_transaction(
            transaction_key=transaction_key,
            new_transaction_key=new_transaction_key,
            user_id=user_id,
            metadata=successful_transaction,
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
            success_url=base_url + "/profile/credits?topup=success",
            cancel_url=base_url + "/profile/credits?topup=cancel",
            allow_promotion_codes=True,
        )

        # Create inactive transaction for Stripe checkout
        # This will be activated later when payment succeeds
        await CreditTransaction.prisma().create(
            data={
                "userId": user_id,
                "amount": amount,
                "type": CreditTransactionType.TOP_UP,
                "transactionKey": checkout_session.id,
                "isActive": False,
                "metadata": SafeJson(checkout_session),
                "createdAt": self.time_now(),
            }
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
            "amount": {"gt": 0},
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

        # If the transaction is not a checkout session, then skip the fulfillment
        if not credit_transaction.transactionKey.startswith("cs_"):
            return

        # Retrieve the Checkout Session from the API
        checkout_session = stripe.checkout.Session.retrieve(
            credit_transaction.transactionKey,
            expand=["payment_intent"],
        )

        # Check the Checkout Session's payment_status property
        # to determine if fulfillment should be performed
        if checkout_session.payment_status in ["paid", "no_payment_required"]:
            if payment_intent := checkout_session.payment_intent:
                assert isinstance(payment_intent, stripe.PaymentIntent)
                new_transaction_key = payment_intent.id
            else:
                new_transaction_key = None

            await self._enable_transaction(
                transaction_key=credit_transaction.transactionKey,
                new_transaction_key=new_transaction_key,
                user_id=credit_transaction.userId,
                metadata=SafeJson(checkout_session),
            )

    async def get_credits(self, user_id: str) -> int:
        user = await User.prisma().find_unique(where={"id": user_id})
        return user.balance if user else 0

    async def get_transaction_history(
        self,
        user_id: str,
        transaction_count_limit: int | None = 100,
        transaction_time_ceiling: datetime | None = None,
        transaction_type: str | None = None,
    ) -> TransactionHistory:
        transactions_filter: CreditTransactionWhereInput = {
            "userId": user_id,
            "isActive": True,
        }
        if transaction_time_ceiling:
            transaction_time_ceiling = transaction_time_ceiling.replace(
                tzinfo=timezone.utc
            )
            transactions_filter["createdAt"] = {"lt": transaction_time_ceiling}
        if transaction_type:
            transactions_filter["type"] = CreditTransactionType[transaction_type]
        transactions = await CreditTransaction.prisma().find_many(
            where=transactions_filter,
            order={"createdAt": "desc"},
            take=transaction_count_limit,
        )

        # doesn't fill current_balance, reason, user_email, admin_email, or extra_data
        grouped_transactions: dict[str, UserTransaction] = defaultdict(
            lambda: UserTransaction(user_id=user_id)
        )
        tx_time = None
        for t in transactions:
            metadata = (
                UsageTransactionMetadata.model_validate(t.metadata)
                if t.metadata
                else UsageTransactionMetadata()
            )
            tx_time = t.createdAt.replace(tzinfo=timezone.utc)

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
                gt.transaction_key = t.transactionKey

            gt.amount += t.amount
            gt.transaction_type = t.type

            if tx_time > gt.transaction_time:
                gt.transaction_time = tx_time
                gt.running_balance = t.runningBalance or 0

        return TransactionHistory(
            transactions=list(grouped_transactions.values()),
            next_transaction_time=(
                tx_time if len(transactions) == transaction_count_limit else None
            ),
        )

    async def get_refund_requests(
        self, user_id: str, limit: int = MAX_CREDIT_REFUND_REQUESTS_FETCH
    ) -> list[RefundRequest]:
        return [
            RefundRequest(
                id=r.id,
                user_id=r.userId,
                transaction_key=r.transactionKey,
                amount=r.amount,
                reason=r.reason,
                result=r.result,
                status=r.status,
                created_at=r.createdAt,
                updated_at=r.updatedAt,
            )
            for r in await CreditRefundRequest.prisma().find_many(
                where={"userId": user_id},
                order={"createdAt": "desc"},
                take=limit,
            )
        ]


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
                transaction_type=CreditTransactionType.GRANT,
                transaction_key=f"MONTHLY-CREDIT-TOP-UP-{cur_time}",
                metadata=SafeJson({"reason": "Monthly credit refill"}),
            )
            return balance
        except UniqueViolationError:
            # Already refilled this month
            return (await self._get_credits(user_id))[0]


class DisabledUserCredit(UserCreditBase):
    async def get_credits(self, *args, **kwargs) -> int:
        return 100

    async def get_transaction_history(self, *args, **kwargs) -> TransactionHistory:
        return TransactionHistory(transactions=[], next_transaction_time=None)

    async def get_refund_requests(self, *args, **kwargs) -> list[RefundRequest]:
        return []

    async def spend_credits(self, *args, **kwargs) -> int:
        return 0

    async def top_up_credits(self, *args, **kwargs):
        pass

    async def onboarding_reward(self, *args, **kwargs):
        pass

    async def top_up_intent(self, *args, **kwargs) -> str:
        return ""

    async def top_up_refund(self, *args, **kwargs) -> int:
        return 0

    async def deduct_credits(self, *args, **kwargs):
        pass

    async def handle_dispute(self, *args, **kwargs):
        pass

    async def fulfill_checkout(self, *args, **kwargs):
        pass


def get_user_credit_model() -> UserCreditBase:
    if not settings.config.enable_credit:
        return DisabledUserCredit()

    if settings.config.enable_beta_monthly_credit:
        return BetaUserCredit(settings.config.num_user_credits_refill)

    return UserCredit()


def get_block_costs() -> dict[str, list["BlockCost"]]:
    return {block().id: costs for block, costs in BLOCK_COSTS.items()}


def get_block_cost(block: "Block") -> list["BlockCost"]:
    return BLOCK_COSTS.get(block.__class__, [])


async def get_stripe_customer_id(user_id: str) -> str:
    user = await get_user_by_id(user_id)

    if user.stripe_customer_id:
        return user.stripe_customer_id

    customer = stripe.Customer.create(
        name=user.name or "",
        email=user.email,
        metadata={"user_id": user_id},
    )
    await User.prisma().update(
        where={"id": user_id}, data={"stripeCustomerId": customer.id}
    )
    return customer.id


async def set_auto_top_up(user_id: str, config: AutoTopUpConfig):
    await User.prisma().update(
        where={"id": user_id},
        data={"topUpConfig": SafeJson(config.model_dump())},
    )


async def get_auto_top_up(user_id: str) -> AutoTopUpConfig:
    user = await get_user_by_id(user_id)

    if not user.top_up_config:
        return AutoTopUpConfig(threshold=0, amount=0)

    return AutoTopUpConfig.model_validate(user.top_up_config)


async def admin_get_user_history(
    page: int = 1,
    page_size: int = 20,
    search: str | None = None,
    transaction_filter: CreditTransactionType | None = None,
) -> UserHistoryResponse:

    if page < 1 or page_size < 1:
        raise ValueError("Invalid pagination input")

    where_clause: CreditTransactionWhereInput = {}
    if transaction_filter:
        where_clause["type"] = transaction_filter
    if search:
        where_clause["OR"] = [
            {"userId": {"contains": search, "mode": "insensitive"}},
            {"User": {"is": {"email": {"contains": search, "mode": "insensitive"}}}},
            {"User": {"is": {"name": {"contains": search, "mode": "insensitive"}}}},
        ]
    transactions = await CreditTransaction.prisma().find_many(
        where=where_clause,
        skip=(page - 1) * page_size,
        take=page_size,
        include={"User": True},
        order={"createdAt": "desc"},
    )
    total = await CreditTransaction.prisma().count(where=where_clause)
    total_pages = (total + page_size - 1) // page_size

    history = []
    for tx in transactions:
        admin_id = ""
        admin_email = ""
        reason = ""

        metadata: dict = cast(dict, tx.metadata) or {}

        if metadata:
            admin_id = metadata.get("admin_id")
            admin_email = (
                (await get_user_email_by_id(admin_id) or f"Unknown Admin: {admin_id}")
                if admin_id
                else ""
            )
            reason = metadata.get("reason", "No reason provided")

        balance, last_update = await get_user_credit_model()._get_credits(tx.userId)

        history.append(
            UserTransaction(
                transaction_key=tx.transactionKey,
                transaction_time=tx.createdAt,
                transaction_type=tx.type,
                amount=tx.amount,
                current_balance=balance,
                running_balance=tx.runningBalance or 0,
                user_id=tx.userId,
                user_email=(
                    tx.User.email
                    if tx.User
                    else (await get_user_by_id(tx.userId)).email
                ),
                reason=reason,
                admin_email=admin_email,
                extra_data=str(metadata),
            )
        )
    return UserHistoryResponse(
        history=history,
        pagination=Pagination(
            total_items=total,
            total_pages=total_pages,
            current_page=page,
            page_size=page_size,
        ),
    )
