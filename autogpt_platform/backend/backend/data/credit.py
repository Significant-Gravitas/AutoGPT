import logging
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
from prisma.models import CreditRefundRequest, CreditTransaction, User, UserBalance
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
    async def top_up_credits(
        self,
        user_id: str,
        amount: int,
        top_up_type: TopUpType = TopUpType.UNCATEGORIZED,
    ) -> int:
        """
        Top up the credits for the user.

        Args:
            user_id (str): The user ID.
            amount (int): The amount to top up.
            top_up_type (TopUpType): The type of top-up (default: UNCATEGORIZED).

        Returns:
            int: The new balance after top-up.
        """
        pass

    @abstractmethod
    async def onboarding_reward(
        self, user_id: str, credits: int, step: OnboardingStep
    ) -> int | None:
        """
        Reward the user with credits for completing an onboarding step.
        Won't reward if the user has already received credits for the step.

        Args:
            user_id (str): The user ID.
            credits (int): The amount of credits to reward.
            step (OnboardingStep): The onboarding step.

        Returns:
            int | None: The new balance if successful, None if already rewarded.
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
        Uses UserBalance table for atomic operations and better performance.
        """
        user_balance = await UserBalance.prisma().find_unique(where={"userId": user_id})
        if not user_balance:
            return 0, datetime.min.replace(tzinfo=timezone.utc)

        return user_balance.balance, user_balance.updatedAt

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
                INSERT INTO "UserBalance" ("id", "userId", "balance", "updatedAt")
                SELECT gen_random_uuid()::text, tx_activate."userId", 0, CURRENT_TIMESTAMP
                FROM tx_activate
                ON CONFLICT ("userId") DO UPDATE SET
                    balance = CASE 
                        WHEN tx_activate.amount > 0 AND "UserBalance".balance > $5 - tx_activate.amount THEN $5
                        ELSE "UserBalance".balance + tx_activate.amount
                    END,
                    "updatedAt" = CURRENT_TIMESTAMP
                FROM tx_activate
                WHERE "UserBalance"."userId" = tx_activate."userId"
                RETURNING balance, "updatedAt", tx_activate.new_key
            ),
            transaction_update AS (
                UPDATE "CreditTransaction"
                SET 
                    "runningBalance" = balance_update.balance,
                    "createdAt" = balance_update."updatedAt"
                FROM balance_update
                WHERE 
                    "CreditTransaction"."transactionKey" = balance_update.new_key
                    AND "CreditTransaction"."userId" = $4
                RETURNING "CreditTransaction"."transactionKey"
            )
            SELECT "transactionKey" FROM transaction_update;
            """,
            new_transaction_key,
            metadata,
            transaction_key,
            user_id,
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
        """Unified atomic transaction method that handles all credit operations."""
        # Validate input
        if amount > POSTGRES_INT_MAX or amount < -POSTGRES_INT_MAX:
            raise ValueError(f"Invalid amount: {amount}")

        # Check ceiling balance if specified
        if ceiling_balance and amount > 0:
            user_balance = await UserBalance.prisma().find_unique(
                where={"userId": user_id}
            )
            if user_balance and user_balance.balance >= ceiling_balance:
                raise ValueError(
                    f"You already have enough balance of {format_credits_as_dollars(user_balance.balance)}, top-up is not required when you already have at least {format_credits_as_dollars(ceiling_balance)}"
                )

        # Single unified atomic operation for all transaction types
        result = await db.prisma.query_raw(
            """
            WITH balance_upsert AS (
                INSERT INTO "UserBalance" ("id", "userId", "balance", "updatedAt")
                VALUES (gen_random_uuid()::text, $1, 
                    CASE
                        -- For new users, apply the amount directly (with overflow protection)
                        WHEN $2 > $6::int THEN $6::int
                        WHEN $2 > 0 AND $7::int IS NOT NULL AND $2 > $7::int THEN $7::int
                        ELSE GREATEST(0, $2)  -- Don't allow negative initial balance
                    END, 
                    CURRENT_TIMESTAMP)
                ON CONFLICT ("userId") DO UPDATE SET
                    balance = CASE 
                        -- For spending (amount < 0): Check sufficient balance
                        WHEN $2 < 0 AND $8::boolean = true AND "UserBalance".balance + $2 < 0 THEN "UserBalance".balance  -- No change if insufficient
                        -- For ceiling balance (amount > 0): Apply ceiling (clamp to maximum)
                        WHEN $2 > 0 AND $7::int IS NOT NULL AND "UserBalance".balance + $2 > $7::int THEN $7::int
                        -- For regular operations: Apply with overflow protection  
                        WHEN "UserBalance".balance + $2 > $6::int THEN $6::int
                        ELSE "UserBalance".balance + $2
                    END,
                    "updatedAt" = CURRENT_TIMESTAMP
                WHERE ($2 >= 0 OR $8::boolean = false OR "UserBalance".balance + $2 >= 0)
                RETURNING "UserBalance"."userId", "UserBalance".balance, "UserBalance"."updatedAt"
            ),
            transaction_insert AS (
                INSERT INTO "CreditTransaction" (
                    "userId", "amount", "type", "runningBalance", 
                    "metadata", "isActive", "createdAt", "transactionKey"
                )
                SELECT 
                    balance_upsert."userId",
                    $2::int,
                    $3::"CreditTransactionType",
                    balance_upsert.balance,
                    $4::jsonb,
                    $5::boolean,
                    balance_upsert."updatedAt",
                    COALESCE($9, gen_random_uuid()::text)
                FROM balance_upsert
                RETURNING "runningBalance", "transactionKey"
            )
            SELECT "runningBalance" as balance, "transactionKey" FROM transaction_insert;
            """,
            user_id,  # $1
            amount,  # $2
            transaction_type.value,  # $3
            metadata,  # $4
            is_active,  # $5
            POSTGRES_INT_MAX,  # $6 - overflow protection
            ceiling_balance,  # $7 - ceiling balance (nullable)
            fail_insufficient_credits,  # $8 - check balance for spending
            transaction_key,  # $9 - transaction key (nullable)
        )

        if result:
            return result[0]["balance"], result[0]["transactionKey"]

        # If no result, either user doesn't exist or insufficient balance
        user_balance = await UserBalance.prisma().find_unique(where={"userId": user_id})
        if not user_balance:
            # Create UserBalance record for new user if it doesn't exist
            user_balance = await UserBalance.prisma().create(
                data={
                    "userId": user_id,
                    "balance": 0,
                }
            )
            # If this was a spending operation, it should fail
            if amount < 0 and fail_insufficient_credits:
                raise InsufficientBalanceError(
                    message=f"Insufficient balance of {format_credits_as_dollars(0)}, where this will cost {format_credits_as_dollars(abs(amount))}",
                    user_id=user_id,
                    balance=0,
                    amount=amount,
                )

        # Must be insufficient balance for spending operation
        if amount < 0 and fail_insufficient_credits:
            raise InsufficientBalanceError(
                message=f"Insufficient balance of {format_credits_as_dollars(user_balance.balance)}, where this will cost {format_credits_as_dollars(abs(amount))}",
                user_id=user_id,
                balance=user_balance.balance,
                amount=amount,
            )

        # Unexpected case
        raise ValueError(f"Transaction failed for user {user_id}, amount {amount}")


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
            return await self.get_credits(user_id)

        # Validate input to prevent integer overflow
        if cost < 0 or cost > POSTGRES_INT_MAX:
            raise ValueError(f"Invalid cost amount: {cost}")

        # Use _add_transaction directly as suggested by Swifty
        new_balance, _ = await self._add_transaction(
            user_id=user_id,
            amount=-cost,  # Negative for spending
            transaction_type=CreditTransactionType.USAGE,
            is_active=True,
            transaction_key=None,  # Let it generate UUID
            fail_insufficient_credits=True,  # Check for insufficient balance
            metadata=SafeJson(metadata.model_dump()),
        )

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
    ) -> int:
        # Validate input to prevent integer overflow
        if amount <= 0 or amount > POSTGRES_INT_MAX:
            raise ValueError(f"Invalid top-up amount: {amount}")

        # Use _add_transaction directly as suggested by Swifty
        new_balance, _ = await self._add_transaction(
            user_id=user_id,
            amount=amount,
            transaction_type=CreditTransactionType.TOP_UP,
            is_active=True,
            transaction_key=None,  # Let it generate UUID
            ceiling_balance=POSTGRES_INT_MAX,  # Use max ceiling for manual top-ups
            fail_insufficient_credits=False,  # No need to check balance when adding credits
            metadata=SafeJson(
                {
                    "reason": f"Manual top up credits for {user_id}",
                    "type": top_up_type.value,
                }
            ),
        )

        return new_balance

    async def onboarding_reward(
        self, user_id: str, credits: int, step: OnboardingStep
    ) -> int | None:
        # Validate input to prevent integer overflow
        if credits <= 0 or credits > POSTGRES_INT_MAX:
            raise ValueError(f"Invalid reward amount: {credits}")

        transaction_key = f"REWARD-{user_id}-{step.value}"

        # Check if already rewarded (for idempotency)
        existing = await CreditTransaction.prisma().find_first(
            where={
                "transactionKey": transaction_key,
                "userId": user_id,
            }
        )
        if existing:
            # Already rewarded, return None to indicate duplicate
            return None

        # Use _add_transaction directly as suggested by Swifty
        try:
            new_balance, _ = await self._add_transaction(
                user_id=user_id,
                amount=credits,
                transaction_type=CreditTransactionType.GRANT,
                is_active=True,
                transaction_key=transaction_key,
                ceiling_balance=POSTGRES_INT_MAX,
                fail_insufficient_credits=False,  # No need to check balance when adding credits
                metadata=SafeJson(
                    {"reason": f"Reward for completing {step.value} onboarding step."}
                ),
            )
            return new_balance
        except Exception as e:
            # If transaction fails, check if it was due to race condition (another request created it)
            existing = await CreditTransaction.prisma().find_first(
                where={
                    "transactionKey": transaction_key,
                    "userId": user_id,
                }
            )
            if existing:
                # Race condition detected - another request already created this reward
                return None
            else:
                # Real error occurred, re-raise the original exception
                raise e

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

        # Use unified atomic transaction function (allow going negative for refunds)
        # Create serializable metadata from request
        try:
            # Try to extract basic attributes for Stripe objects
            request_metadata = {
                "id": getattr(request, "id", None),
                "amount": getattr(request, "amount", None),
                "status": getattr(request, "status", None),
                "payment_intent": getattr(request, "payment_intent", None),
                "reason": getattr(request, "reason", None),
                "created": getattr(request, "created", None),
            }
            # Remove None values
            request_metadata = {
                k: v for k, v in request_metadata.items() if v is not None
            }
        except Exception:
            request_metadata = {"error": "Could not serialize request metadata"}

        try:
            balance, _ = await self._add_transaction(
                user_id=transaction.userId,
                amount=-request.amount,  # Negative for deduction
                transaction_type=CreditTransactionType.REFUND,
                is_active=True,
                transaction_key=request.id,
                fail_insufficient_credits=False,  # Allow negative balance for refunds
                metadata=SafeJson(request_metadata),
            )
        except UniqueViolationError:
            # Idempotent retry: fetch existing transaction and continue
            existing = await CreditTransaction.prisma().find_first(
                where={"transactionKey": request.id, "userId": transaction.userId}
            )
            balance = (
                existing.runningBalance
                if existing and existing.runningBalance is not None
                else await self.get_credits(transaction.userId)
            )

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
            # Let database set createdAt with CURRENT_TIMESTAMP via @default(now())
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
            user_balance = await UserBalance.prisma().find_unique(
                where={"userId": user_id}
            )
            current_balance = user_balance.balance if user_balance else 0
            if current_balance >= ceiling_balance:
                raise ValueError(
                    f"You already have enough balance of {format_credits_as_dollars(current_balance)}, top-up is not required when you already have at least {format_credits_as_dollars(ceiling_balance)}"
                )

        tx = await CreditTransaction.prisma().create(data=transaction_data)
        transaction_key = tx.transactionKey

        customer_id = await get_stripe_customer_id(user_id)

        payment_methods = stripe.PaymentMethod.list(customer=customer_id, type="card")
        if not getattr(payment_methods, "data", None):
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
                "metadata": SafeJson(
                    {
                        "id": checkout_session.id,
                        "amount": amount,
                        "status": checkout_session.status,
                    }
                ),
                # Let database set createdAt with CURRENT_TIMESTAMP via @default(now())
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
                metadata=SafeJson(
                    {
                        "session_id": checkout_session.id,
                        "amount": checkout_session.amount_total,
                        "status": checkout_session.payment_status,
                        "payment_intent": getattr(
                            checkout_session, "payment_intent", None
                        ),
                    }
                ),
            )

    async def get_credits(self, user_id: str) -> int:
        user_balance = await UserBalance.prisma().find_unique(where={"userId": user_id})
        return user_balance.balance if user_balance else 0

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
    async def get_credits(self, *_args, **_kwargs) -> int:
        del _args, _kwargs  # Suppress unused parameter warnings
        return 100

    async def get_transaction_history(self, *_args, **_kwargs) -> TransactionHistory:
        del _args, _kwargs  # Suppress unused parameter warnings
        return TransactionHistory(transactions=[], next_transaction_time=None)

    async def get_refund_requests(self, *_args, **_kwargs) -> list[RefundRequest]:
        del _args, _kwargs  # Suppress unused parameter warnings
        return []

    async def spend_credits(self, *_args, **_kwargs) -> int:
        del _args, _kwargs  # Suppress unused parameter warnings
        return 0

    async def top_up_credits(
        self,
        user_id: str,
        amount: int,
        top_up_type: TopUpType = TopUpType.UNCATEGORIZED,
    ) -> int:
        del user_id, amount, top_up_type  # Suppress unused parameter warnings
        return 0

    async def onboarding_reward(self, *_args, **_kwargs) -> int | None:
        del _args, _kwargs  # Suppress unused parameter warnings
        return None

    async def top_up_intent(self, *_args, **_kwargs) -> str:
        del _args, _kwargs  # Suppress unused parameter warnings
        return ""

    async def top_up_refund(self, *_args, **_kwargs) -> int:
        del _args, _kwargs  # Suppress unused parameter warnings
        return 0

    async def deduct_credits(self, *_args, **_kwargs):
        del _args, _kwargs  # Suppress unused parameter warnings
        pass

    async def handle_dispute(self, *_args, **_kwargs):
        del _args, _kwargs  # Suppress unused parameter warnings
        pass

    async def fulfill_checkout(self, *_args, **_kwargs):
        del _args, _kwargs  # Suppress unused parameter warnings
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

    # Pre-fetch all unique admin IDs and user balances to avoid N+1 queries
    unique_admin_ids = {
        cast(dict, tx.metadata or {}).get("admin_id")
        for tx in transactions
        if cast(dict, tx.metadata or {}).get("admin_id")
    }
    unique_user_ids = {tx.userId for tx in transactions}

    # Batch fetch admin emails
    admin_email_map = {}
    for admin_id in unique_admin_ids:
        if admin_id:
            email = await get_user_email_by_id(admin_id)
            admin_email_map[admin_id] = email or f"Unknown Admin: {admin_id}"

    # Batch fetch user balances in one query
    user_balance_map = {}
    if unique_user_ids:
        user_balances = await UserBalance.prisma().find_many(
            where={"userId": {"in": list(unique_user_ids)}}
        )
        user_balance_map = {ub.userId: ub.balance for ub in user_balances}

    history = []
    for tx in transactions:
        admin_id = ""
        admin_email = ""
        reason = ""

        metadata: dict = cast(dict, tx.metadata) or {}

        if metadata:
            admin_id = metadata.get("admin_id")
            admin_email = admin_email_map.get(admin_id, "")
            reason = metadata.get("reason", "No reason provided")

        balance = user_balance_map.get(tx.userId, 0)

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
