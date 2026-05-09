import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Literal, cast

import stripe
from fastapi.concurrency import run_in_threadpool
from prisma.enums import (
    CreditRefundRequestStatus,
    CreditTransactionType,
    NotificationType,
    OnboardingStep,
    SubscriptionTier,
)
from prisma.errors import PrismaError, UniqueViolationError
from prisma.models import CreditRefundRequest, CreditTransaction, User, UserBalance
from prisma.types import CreditRefundRequestCreateInput, CreditTransactionWhereInput
from pydantic import BaseModel

from backend.api.features.admin.model import UserHistoryResponse
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.db import query_raw_with_schema
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
from backend.util.cache import cached
from backend.util.exceptions import InsufficientBalanceError
from backend.util.feature_flag import Flag, get_feature_flag_value
from backend.util.json import SafeJson, dumps
from backend.util.models import Pagination
from backend.util.retry import func_retry
from backend.util.settings import Settings

if TYPE_CHECKING:
    from backend.blocks._base import Block, BlockCost

settings = Settings()
stripe.api_key = settings.secrets.stripe_api_key
logger = logging.getLogger(__name__)
base_url = settings.config.frontend_base_url or settings.config.platform_base_url

# Constants for test compatibility
POSTGRES_INT_MAX = 2147483647
POSTGRES_INT_MIN = -2147483648

BillingCycle = Literal["monthly", "yearly"]


class UsageTransactionMetadata(BaseModel):
    graph_exec_id: str | None = None
    graph_id: str | None = None
    node_id: str | None = None
    node_exec_id: str | None = None
    block_id: str | None = None
    block: str | None = None
    input: dict[str, Any] | None = None
    reason: str | None = None


class InvoiceListItem(BaseModel):
    """A single invoice surfaced from Stripe for the billing UI.

    Mirrors the subset of `stripe.Invoice` we expose to the client. ``hosted_invoice_url``
    opens the Stripe-hosted view; ``invoice_pdf_url`` lets users download the PDF directly.

    ``total_cents`` is the invoice total (what the user owes / will be charged); use it
    for the displayed amount. ``amount_paid_cents`` is what Stripe has actually settled
    so far — `0` for ``open``/``draft`` invoices — and is kept for callers that need to
    show outstanding balances separately.
    """

    id: str
    number: str | None = None
    created_at: datetime
    total_cents: int
    amount_paid_cents: int
    currency: str = "usd"
    status: str
    description: str | None = None
    hosted_invoice_url: str | None = None
    invoice_pdf_url: str | None = None


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
        fail_insufficient_credits: bool = True,
    ) -> int:
        """
        Spend the credits for the user based on the cost.

        Args:
            user_id (str): The user ID.
            cost (int): The cost to spend.
            metadata (UsageTransactionMetadata): The metadata of the transaction.
            fail_insufficient_credits (bool): When True (default) raise
                InsufficientBalanceError if the wallet can't cover the spend.
                When False the spend is recorded unconditionally and the
                balance may go negative — used by post-flight reconciliation
                so a delta charge that exceeds the wallet is captured as
                debt instead of silently leaking.

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
    async def grant_credits(
        self,
        user_id: str,
        amount: int,
        reason: str,
        transaction_key: str | None = None,
    ) -> int:
        """
        Grant non-purchased credits to the user (no Stripe charge).

        Use this for any credit movement that is NOT a user-initiated Stripe
        checkout: in-app refunds for failed services, beta-tester top-ups,
        manual corrections, subscription credit grants, etc. Writes a
        ``GRANT`` row so the dashboard does not misreport free credits as
        ``TOP_UP`` (which is reserved for real Stripe checkouts).

        Args:
            user_id (str): The user ID.
            amount (int): The amount of credits to grant (positive).
            reason (str): Human-readable reason recorded in transaction metadata.
            transaction_key (str | None): Optional deterministic key for
                idempotent retries.  If supplied and a row already exists with
                this key for the user, the existing balance is returned
                without inserting a new row.

        Returns:
            int: The new balance after the grant.
        """
        pass

    @abstractmethod
    async def onboarding_reward(
        self, user_id: str, credits: int, step: OnboardingStep
    ) -> bool:
        """
        Reward the user with credits for completing an onboarding step.
        Won't reward if the user has already received credits for the step.

        Args:
            user_id (str): The user ID.
            credits (int): The amount to reward.
            step (OnboardingStep): The onboarding step.

        Returns:
            bool: True if rewarded, False if already rewarded.
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
            return_url=base_url + "/settings/billing",
        )
        return session.url

    async def list_invoices(
        self, user_id: str, limit: int = 24
    ) -> list["InvoiceListItem"]:
        """List recent Stripe invoices for the given user.

        Defaults to the most-recent ``limit`` invoices. Concrete subclasses
        override this with the actual Stripe call; ``DisabledUserCredit``
        returns an empty list so the UI degrades gracefully when credits
        are disabled.
        """
        return []

    @staticmethod
    def time_now() -> datetime:
        return datetime.now(timezone.utc)

    # ====== Transaction Helper Methods ====== #
    # Any modifications to the transaction table should only be done through these methods #

    async def _get_credits(self, user_id: str) -> tuple[int, datetime]:
        """
        Returns the current balance of the user & the latest balance snapshot time.
        """
        # Check UserBalance first for efficiency and consistency
        user_balance = await UserBalance.prisma().find_unique(where={"userId": user_id})
        if user_balance:
            return user_balance.balance, user_balance.updatedAt

        # Fallback to transaction history computation if UserBalance doesn't exist
        top_time = self.time_now()
        snapshot = await CreditTransaction.prisma().find_first(
            where={
                "userId": user_id,
                "createdAt": {"lte": top_time},
                "isActive": True,
                "NOT": [{"runningBalance": None}],
            },
            order={"createdAt": "desc"},
        )
        datetime_min = datetime.min.replace(tzinfo=timezone.utc)
        snapshot_balance = snapshot.runningBalance or 0 if snapshot else 0
        snapshot_time = snapshot.createdAt if snapshot else datetime_min

        return snapshot_balance, snapshot_time

    @func_retry
    async def _enable_transaction(
        self,
        transaction_key: str,
        user_id: str,
        metadata: SafeJson,
        new_transaction_key: str | None = None,
    ):
        # First check if transaction exists and is inactive (safety check)
        transaction = await CreditTransaction.prisma().find_first(
            where={
                "transactionKey": transaction_key,
                "userId": user_id,
                "isActive": False,
            }
        )
        if not transaction:
            # Transaction doesn't exist or is already active, return early
            return None

        # Atomic operation to enable transaction and update user balance using UserBalance
        result = await query_raw_with_schema(
            """
            WITH user_balance_lock AS (
                SELECT 
                    $2::text as userId, 
                    COALESCE(
                        (SELECT balance FROM {schema_prefix}"UserBalance" WHERE "userId" = $2 FOR UPDATE),
                        -- Fallback: compute balance from transaction history if UserBalance doesn't exist
                        (SELECT COALESCE(ct."runningBalance", 0) 
                         FROM {schema_prefix}"CreditTransaction" ct 
                         WHERE ct."userId" = $2 
                           AND ct."isActive" = true 
                           AND ct."runningBalance" IS NOT NULL 
                         ORDER BY ct."createdAt" DESC 
                         LIMIT 1),
                        0
                    ) as balance
            ),
            transaction_check AS (
                SELECT * FROM {schema_prefix}"CreditTransaction" 
                WHERE "transactionKey" = $1 AND "userId" = $2 AND "isActive" = false
            ),
            balance_update AS (
                INSERT INTO {schema_prefix}"UserBalance" ("userId", "balance", "updatedAt")
                SELECT 
                    $2::text,
                    user_balance_lock.balance + transaction_check.amount,
                    CURRENT_TIMESTAMP
                FROM user_balance_lock, transaction_check
                ON CONFLICT ("userId") DO UPDATE SET
                    "balance" = EXCLUDED."balance",
                    "updatedAt" = EXCLUDED."updatedAt"
                RETURNING "balance", "updatedAt"
            ),
            transaction_update AS (
                UPDATE {schema_prefix}"CreditTransaction"
                SET "transactionKey" = COALESCE($4, $1),
                    "isActive" = true,
                    "runningBalance" = balance_update.balance,
                    "createdAt" = balance_update."updatedAt",
                    "metadata" = $3::jsonb
                FROM balance_update, transaction_check
                WHERE {schema_prefix}"CreditTransaction"."transactionKey" = transaction_check."transactionKey"
                  AND {schema_prefix}"CreditTransaction"."userId" = transaction_check."userId"
                RETURNING {schema_prefix}"CreditTransaction"."runningBalance"
            )
            SELECT "runningBalance" as balance FROM transaction_update;
            """,
            transaction_key,  # $1
            user_id,  # $2
            dumps(metadata.data),  # $3 - use pre-serialized JSON string for JSONB
            new_transaction_key,  # $4
        )

        if result:
            # UserBalance is already updated by the CTE

            # Clear insufficient funds notification flags when credits are added
            # so user can receive alerts again if they run out in the future.
            if transaction.amount > 0 and transaction.type in [
                CreditTransactionType.GRANT,
                CreditTransactionType.TOP_UP,
            ]:
                from backend.executor.billing import (
                    clear_insufficient_funds_notifications,
                )

                await clear_insufficient_funds_notifications(user_id)

            return result[0]["balance"]

    async def _add_transaction(
        self,
        user_id: str,
        amount: int,
        transaction_type: CreditTransactionType,
        is_active: bool = True,
        transaction_key: str | None = None,
        ceiling_balance: int | None = None,
        fail_insufficient_credits: bool = True,
        metadata: SafeJson = SafeJson({}),
    ) -> tuple[int, str]:
        """
        Add a new transaction for the user.
        This is the only method that should be used to add a new transaction.

        ATOMIC OPERATION DESIGN DECISION:
        ================================
        This method uses PostgreSQL row-level locking (FOR UPDATE) for atomic credit operations.
        After extensive analysis of concurrency patterns and correctness requirements, we determined
        that the FOR UPDATE approach is necessary despite the latency overhead.

        WHY FOR UPDATE LOCKING IS REQUIRED:
        ----------------------------------
        1. **Data Consistency**: Credit operations must be ACID-compliant. The balance check,
           calculation, and update must be atomic to prevent race conditions where:
           - Multiple spend operations could exceed available balance
           - Lost update problems could occur with concurrent top-ups
           - Refunds could create negative balances incorrectly

        2. **Serializability**: FOR UPDATE ensures operations are serialized at the database level,
           guaranteeing that each transaction sees a consistent view of the balance before applying changes.

        3. **Correctness Over Performance**: Financial operations require absolute correctness.
           The ~10-50ms latency increase from row locking is acceptable for the guarantee that
           no user will ever have an incorrect balance due to race conditions.

        4. **PostgreSQL Optimization**: Modern PostgreSQL versions optimize row locks efficiently.
           The performance cost is minimal compared to the complexity and risk of lock-free approaches.

        ALTERNATIVES CONSIDERED AND REJECTED:
        ------------------------------------
        - **Optimistic Concurrency**: Using version numbers or timestamps would require complex
          retry logic and could still fail under high contention scenarios.
        - **Application-Level Locking**: Redis locks or similar would add network overhead and
          single points of failure while being less reliable than database locks.
        - **Event Sourcing**: Would require complete architectural changes and eventual consistency
          models that don't fit our real-time balance requirements.

        PERFORMANCE CHARACTERISTICS:
        ---------------------------
        - Single user operations: 10-50ms latency (acceptable for financial operations)
        - Concurrent operations on same user: Serialized (prevents data corruption)
        - Concurrent operations on different users: Fully parallel (no blocking)

        This design prioritizes correctness and data integrity over raw performance,
        which is the appropriate choice for a credit/payment system.

        Args:
            user_id (str): The user ID.
            amount (int): The amount of credits to add.
            transaction_type (CreditTransactionType): The type of transaction.
            is_active (bool): Whether the transaction is active or needs to be manually activated through _enable_transaction.
            transaction_key (str | None): The transaction key. Avoids adding transaction if the key already exists.
            ceiling_balance (int | None): The ceiling balance. Avoids adding more credits if the balance is already above the ceiling.
            fail_insufficient_credits (bool): Whether to fail if the user has insufficient credits.
            metadata (Json): The metadata of the transaction.

        Returns:
            tuple[int, str]: The new balance & the transaction key.
        """
        # Quick validation for ceiling balance to avoid unnecessary database operations
        if ceiling_balance and amount > 0:
            current_balance, _ = await self._get_credits(user_id)
            if current_balance >= ceiling_balance:
                raise ValueError(
                    f"You already have enough balance of ${current_balance / 100}, top-up is not required when you already have at least ${ceiling_balance / 100}"
                )

        # Single unified atomic operation for all transaction types using UserBalance
        try:
            result = await query_raw_with_schema(
                """
                WITH user_balance_lock AS (
                    SELECT 
                        $1::text as userId, 
                        -- CRITICAL: FOR UPDATE lock prevents concurrent modifications to the same user's balance
                        -- This ensures atomic read-modify-write operations and prevents race conditions
                        COALESCE(
                            (SELECT balance FROM {schema_prefix}"UserBalance" WHERE "userId" = $1 FOR UPDATE),
                            -- Fallback: compute balance from transaction history if UserBalance doesn't exist
                            (SELECT COALESCE(ct."runningBalance", 0) 
                             FROM {schema_prefix}"CreditTransaction" ct 
                             WHERE ct."userId" = $1 
                               AND ct."isActive" = true 
                               AND ct."runningBalance" IS NOT NULL 
                             ORDER BY ct."createdAt" DESC 
                             LIMIT 1),
                            0
                        ) as balance
                ),
                balance_update AS (
                    INSERT INTO {schema_prefix}"UserBalance" ("userId", "balance", "updatedAt")
                    SELECT 
                        $1::text,
                        CASE 
                            -- For inactive transactions: Don't update balance
                            WHEN $5::boolean = false THEN user_balance_lock.balance
                            -- For ceiling balance (amount > 0): Apply ceiling
                            WHEN $2 > 0 AND $7::int IS NOT NULL AND user_balance_lock.balance > $7::int - $2 THEN $7::int
                            -- For regular operations: Apply with overflow/underflow protection  
                            WHEN user_balance_lock.balance + $2 > $6::int THEN $6::int
                            WHEN user_balance_lock.balance + $2 < $10::int THEN $10::int
                            ELSE user_balance_lock.balance + $2
                        END,
                        CURRENT_TIMESTAMP
                    FROM user_balance_lock
                    WHERE (
                        $5::boolean = false OR  -- Allow inactive transactions
                        $2 >= 0 OR              -- Allow positive amounts (top-ups, grants)
                        $8::boolean = false OR  -- Allow when insufficient balance check is disabled
                        user_balance_lock.balance + $2 >= 0  -- Allow spending only when sufficient balance
                    )
                    ON CONFLICT ("userId") DO UPDATE SET
                        "balance" = EXCLUDED."balance",
                        "updatedAt" = EXCLUDED."updatedAt"
                    RETURNING "balance", "updatedAt"
                ),
                transaction_insert AS (
                    INSERT INTO {schema_prefix}"CreditTransaction" (
                        "userId", "amount", "type", "runningBalance", 
                        "metadata", "isActive", "createdAt", "transactionKey"
                    )
                    SELECT 
                        $1::text,
                        $2::int,
                        $3::text::{schema_prefix}"CreditTransactionType",
                        CASE 
                            -- For inactive transactions: Set runningBalance to original balance (don't apply the change yet)
                            WHEN $5::boolean = false THEN user_balance_lock.balance
                            ELSE COALESCE(balance_update.balance, user_balance_lock.balance)
                        END,
                        $4::jsonb,
                        $5::boolean,
                        COALESCE(balance_update."updatedAt", CURRENT_TIMESTAMP),
                        COALESCE($9, gen_random_uuid()::text)
                    FROM user_balance_lock
                    LEFT JOIN balance_update ON true
                    WHERE (
                        $5::boolean = false OR  -- Allow inactive transactions
                        $2 >= 0 OR              -- Allow positive amounts (top-ups, grants)
                        $8::boolean = false OR  -- Allow when insufficient balance check is disabled
                        user_balance_lock.balance + $2 >= 0  -- Allow spending only when sufficient balance
                    )
                    RETURNING "runningBalance", "transactionKey"
                )
                SELECT "runningBalance" as balance, "transactionKey" FROM transaction_insert;
                """,
                user_id,  # $1
                amount,  # $2
                transaction_type.value,  # $3
                dumps(metadata.data),  # $4 - use pre-serialized JSON string for JSONB
                is_active,  # $5
                POSTGRES_INT_MAX,  # $6 - overflow protection
                ceiling_balance,  # $7 - ceiling balance (nullable)
                fail_insufficient_credits,  # $8 - check balance for spending
                transaction_key,  # $9 - transaction key (nullable)
                POSTGRES_INT_MIN,  # $10 - underflow protection
            )
        except Exception as e:
            # Convert raw SQL unique constraint violations to UniqueViolationError
            # for consistent exception handling throughout the codebase
            error_str = str(e).lower()
            if (
                "already exists" in error_str
                or "duplicate key" in error_str
                or "unique constraint" in error_str
            ):
                # Extract table and constraint info for better error messages
                # Re-raise as a UniqueViolationError but with proper format
                # Create a minimal data structure that the error constructor expects
                raise UniqueViolationError({"error": str(e), "user_facing_error": {}})
            # For any other error, re-raise as-is
            raise

        if result:
            new_balance, tx_key = result[0]["balance"], result[0]["transactionKey"]
            # UserBalance is already updated by the CTE

            # Clear insufficient funds notification flags when credits are added
            # so user can receive alerts again if they run out in the future.
            if (
                amount > 0
                and is_active
                and transaction_type
                in [CreditTransactionType.GRANT, CreditTransactionType.TOP_UP]
            ):
                # Lazy import to avoid circular dependency with executor.manager
                from backend.executor.billing import (
                    clear_insufficient_funds_notifications,
                )

                await clear_insufficient_funds_notifications(user_id)

            return new_balance, tx_key

        # If no result, either user doesn't exist or insufficient balance
        user = await User.prisma().find_unique(where={"id": user_id})
        if not user:
            raise ValueError(f"User {user_id} not found")

        # Must be insufficient balance for spending operation
        if amount < 0 and fail_insufficient_credits:
            current_balance, _ = await self._get_credits(user_id)
            raise InsufficientBalanceError(
                message=f"Insufficient balance of ${current_balance / 100}, where this will cost ${abs(amount) / 100}",
                user_id=user_id,
                balance=current_balance,
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
        fail_insufficient_credits: bool = True,
    ) -> int:
        if cost == 0:
            return 0

        balance, _ = await self._add_transaction(
            user_id=user_id,
            amount=-cost,
            transaction_type=CreditTransactionType.USAGE,
            metadata=SafeJson(metadata.model_dump()),
            fail_insufficient_credits=fail_insufficient_credits,
        )

        # Auto top-up if balance is below threshold.
        auto_top_up = await get_auto_top_up(user_id)
        if auto_top_up.threshold and balance < auto_top_up.threshold:
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
                    f"Auto top-up failed for user {user_id}, balance: {balance}, amount: {auto_top_up.amount}, error: {e}"
                )

        return balance

    async def top_up_credits(
        self,
        user_id: str,
        amount: int,
        top_up_type: TopUpType = TopUpType.UNCATEGORIZED,
    ):
        await self._top_up_credits(
            user_id=user_id, amount=amount, top_up_type=top_up_type
        )

    async def grant_credits(
        self,
        user_id: str,
        amount: int,
        reason: str,
        transaction_key: str | None = None,
    ) -> int:
        if amount < 0:
            raise ValueError(f"Grant amount must not be negative: {amount}")
        try:
            balance, _ = await self._add_transaction(
                user_id=user_id,
                amount=amount,
                transaction_type=CreditTransactionType.GRANT,
                transaction_key=transaction_key,
                metadata=SafeJson({"reason": reason}),
            )
        except UniqueViolationError:
            # Idempotent: another request with the same transaction_key already
            # granted this — return the current balance without double-crediting.
            balance, _ = await self._get_credits(user_id)
        return balance

    async def onboarding_reward(self, user_id: str, credits: int, step: OnboardingStep):
        try:
            await self._add_transaction(
                user_id=user_id,
                amount=credits,
                transaction_type=CreditTransactionType.GRANT,
                transaction_key=f"REWARD-{user_id}-{step.value}",
                metadata=SafeJson(
                    {"reason": f"Reward for completing {step.value} onboarding step."}
                ),
            )
            return True
        except UniqueViolationError:
            # User already received this reward
            return False

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
                f"Invalid amount to deduct ${request.amount / 100} from ${transaction.amount / 100} top-up"
            )

        balance, _ = await self._add_transaction(
            user_id=transaction.userId,
            amount=-request.amount,
            transaction_type=CreditTransactionType.REFUND,
            transaction_key=request.id,
            metadata=SafeJson(request),
            fail_insufficient_credits=False,
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
            logger.warning(f"Accepting dispute from {user_id} for ${amount / 100}")
            dispute.close()
            return

        logger.warning(
            f"Adding extra info for dispute from {user_id} for ${amount / 100}"
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

        if amount < 0:
            raise ValueError(f"Top up amount must not be negative: {amount}")

        if key is not None and (
            await CreditTransaction.prisma().find_first(
                where={"transactionKey": key, "userId": user_id}
            )
        ):
            raise ValueError(f"Transaction key {key} already exists for user {user_id}")

        if amount == 0:
            transaction_type = CreditTransactionType.CARD_CHECK
        else:
            transaction_type = CreditTransactionType.TOP_UP

        _, transaction_key = await self._add_transaction(
            user_id=user_id,
            amount=amount,
            transaction_type=transaction_type,
            is_active=False,
            transaction_key=key,
            ceiling_balance=ceiling_balance,
            metadata=(SafeJson(metadata)),
        )

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

        # Resolve the Stripe Product ID from LD; when unset (default), keep the
        # legacy inline product_data path (Stripe creates an ephemeral product
        # per Checkout). When set, reference the canonical Product so all
        # top-ups group under one entity in Stripe Dashboard reporting; the
        # amount stays dynamic via unit_amount.
        topup_product_id = await get_feature_flag_value(
            Flag.STRIPE_PRODUCT_ID_TOPUP.value, user_id, default=None
        )
        line_items: list[stripe.checkout.Session.CreateParamsLineItem] = (
            [
                {
                    "price_data": {
                        "currency": "usd",
                        "product": topup_product_id,
                        "unit_amount": amount,
                    },
                    "quantity": 1,
                }
            ]
            if isinstance(topup_product_id, str) and topup_product_id
            else [
                {
                    "price_data": {
                        "currency": "usd",
                        "product_data": {"name": "AutoGPT Platform Credits"},
                        "unit_amount": amount,
                    },
                    "quantity": 1,
                }
            ]
        )

        # Create checkout session
        # https://docs.stripe.com/checkout/quickstart?client=react
        # unit_amount param is always in the smallest currency unit (so cents for usd)
        # which is equal to amount of credits
        checkout_session = stripe.checkout.Session.create(
            customer=await get_stripe_customer_id(user_id),
            line_items=line_items,
            mode="payment",
            ui_mode="hosted",
            payment_intent_data={"setup_future_usage": "off_session"},
            saved_payment_method_options={"payment_method_save": "enabled"},
            success_url=base_url + "/settings/billing?topup=success",
            cancel_url=base_url + "/settings/billing?topup=cancel",
            allow_promotion_codes=True,
            automatic_tax={"enabled": True},
            billing_address_collection="auto",
            customer_update={"address": "auto"},
        )

        await self._add_transaction(
            user_id=user_id,
            amount=amount,
            transaction_type=CreditTransactionType.TOP_UP,
            transaction_key=checkout_session.id,
            is_active=False,
            metadata=SafeJson(checkout_session),
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
        balance, _ = await self._get_credits(user_id)
        return balance

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

    async def list_invoices(
        self, user_id: str, limit: int = 24
    ) -> list[InvoiceListItem]:
        # Skip the Stripe call entirely for users that have never been
        # provisioned a customer — listing invoices must NOT have the side
        # effect of creating a Stripe Customer record (would orphan billable
        # customers for every beta user that opens the billing page).
        user = await get_user_by_id(user_id)
        if not user.stripe_customer_id:
            return []

        # Bound limit to Stripe's per-page maximum (100) and at least 1
        limit = max(1, min(limit, 100))

        try:
            invoices = await run_in_threadpool(
                stripe.Invoice.list,
                customer=user.stripe_customer_id,
                limit=limit,
            )
        except stripe.StripeError:
            logger.exception("Stripe invoice list failed for user %s", user_id)
            return []

        return [
            InvoiceListItem(
                id=invoice.id or "",
                number=invoice.number,
                created_at=datetime.fromtimestamp(
                    invoice.created or 0, tz=timezone.utc
                ),
                total_cents=invoice.total or 0,
                amount_paid_cents=invoice.amount_paid or 0,
                currency=(invoice.currency or "usd").lower(),
                status=invoice.status or "open",
                description=invoice.description,
                hosted_invoice_url=invoice.hosted_invoice_url,
                invoice_pdf_url=invoice.invoice_pdf,
            )
            for invoice in invoices.data
        ]


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

    async def grant_credits(self, *args, **kwargs) -> int:
        return 100

    async def onboarding_reward(self, *args, **kwargs) -> bool:
        return True

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


async def get_user_credit_model(user_id: str) -> UserCreditBase:
    """Return the credit model for a user.

    The ``user_id`` parameter is currently unused but retained for ABI
    stability — many callers already pass it, and the function may need to
    branch on user identity again in the future.
    """
    _ = user_id
    if not settings.config.enable_credit:
        return DisabledUserCredit()
    return UserCredit()


def get_block_costs() -> dict[str, list["BlockCost"]]:
    return {block().id: costs for block, costs in BLOCK_COSTS.items()}


def get_block_cost(block: "Block") -> list["BlockCost"]:
    return BLOCK_COSTS.get(block.__class__, [])


async def get_stripe_customer_id(user_id: str) -> str:
    user = await get_user_by_id(user_id)

    if user.stripe_customer_id:
        return user.stripe_customer_id

    # Race protection: two concurrent calls (e.g. user double-clicks "Upgrade",
    # or any retried request) would each pass the check above and create their
    # own Stripe Customer, leaving an orphaned billable customer in Stripe.
    # Pass an idempotency_key so Stripe collapses concurrent + retried calls
    # into the same Customer object server-side. The 24h Stripe idempotency
    # window comfortably covers any realistic in-flight retry scenario.
    customer = await run_in_threadpool(
        stripe.Customer.create,
        name=user.name or "",
        email=user.email,
        metadata={"user_id": user_id},
        idempotency_key=f"customer-create-{user_id}",
    )
    await User.prisma().update(
        where={"id": user_id}, data={"stripeCustomerId": customer.id}
    )
    get_user_by_id.cache_delete(user_id)
    return customer.id


async def set_auto_top_up(user_id: str, config: AutoTopUpConfig):
    await User.prisma().update(
        where={"id": user_id},
        data={"topUpConfig": SafeJson(config.model_dump())},
    )
    get_user_by_id.cache_delete(user_id)


async def set_subscription_tier(user_id: str, tier: SubscriptionTier) -> None:
    """Set the user's subscription tier (used by webhook and admin flows)."""
    await User.prisma().update(
        where={"id": user_id},
        data={"subscriptionTier": tier},
    )
    get_user_by_id.cache_delete(user_id)
    # Also invalidate the rate-limit tier cache so CoPilot picks up the new
    # tier immediately rather than waiting up to 5 minutes for the TTL to expire.
    from backend.copilot.rate_limit import get_user_tier  # local import avoids circular

    get_user_tier.cache_delete(user_id)  # type: ignore[attr-defined]
    # Invalidate the pending-change cache too — an admin tier override or the
    # webhook-driven phase transition means any cached pending-change state
    # (schedule, cancel_at_period_end) is likely stale. Without this the
    # billing page can show a pending change for up to 30s after the tier
    # has already flipped.
    get_pending_subscription_change.cache_delete(user_id)


async def _cancel_customer_subscriptions(
    customer_id: str,
    exclude_sub_id: str | None = None,
    at_period_end: bool = False,
) -> int:
    """Cancel all billable Stripe subscriptions for a customer, optionally excluding one.

    Cancels both ``active`` and ``trialing`` subscriptions, since trialing subs will
    start billing once the trial ends and must be cleaned up on downgrade/upgrade to
    avoid double-charging or charging users who intended to cancel.

    When ``at_period_end=True``, schedules cancellation at the end of the current
    billing period instead of cancelling immediately — the user keeps their tier
    until the period ends, then ``customer.subscription.deleted`` fires and the
    webhook downgrades them to BASIC.

    Wraps every synchronous Stripe SDK call with run_in_threadpool so the async event
    loop is never blocked. Raises stripe.StripeError on list/cancel failure so callers
    that need strict consistency can react; cleanup callers can catch and log instead.

    Returns the number of subscriptions cancelled/scheduled for cancellation.
    """
    # Query active and trialing separately; Stripe's list API accepts a single status
    # filter at a time (no OR), and we explicitly want to skip canceled/incomplete/
    # past_due subs rather than filter them out client-side via status="all".
    seen_ids: set[str] = set()
    for status in ("active", "trialing"):
        subscriptions = await run_in_threadpool(
            stripe.Subscription.list, customer=customer_id, status=status, limit=10
        )
        # Iterate only the first page (up to 10); avoid auto_paging_iter which would
        # trigger additional sync HTTP calls inside the event loop.
        if subscriptions.has_more:
            logger.error(
                "_cancel_customer_subscriptions: customer %s has more than 10 %s"
                " subscriptions — only the first page was processed; remaining"
                " subscriptions were NOT cancelled",
                customer_id,
                status,
            )
        for sub in subscriptions.data:
            sub_id = sub["id"]
            if exclude_sub_id and sub_id == exclude_sub_id:
                continue
            if sub_id in seen_ids:
                continue
            seen_ids.add(sub_id)
            if at_period_end:
                # Stripe rejects modify(cancel_at_period_end=True) with 400 when a
                # Subscription Schedule is attached (e.g. the user previously
                # queued a paid→paid downgrade and is now clicking "Cancel").
                # Release the schedule first so the cancel flag can be set; the
                # schedule's pending phase change is superseded by the cancel.
                existing_schedule = sub.schedule
                if existing_schedule:
                    schedule_id = (
                        existing_schedule
                        if isinstance(existing_schedule, str)
                        else existing_schedule.id
                    )
                    await _release_schedule_ignoring_terminal(
                        schedule_id, "_cancel_customer_subscriptions"
                    )
                await run_in_threadpool(
                    stripe.Subscription.modify, sub_id, cancel_at_period_end=True
                )
            else:
                await run_in_threadpool(stripe.Subscription.cancel, sub_id)
    return len(seen_ids)


async def cancel_stripe_subscription(user_id: str) -> bool:
    """Schedule cancellation of all active/trialing Stripe subscriptions at period end.

    The subscription stays active until the end of the billing period so the user
    keeps their tier for the time they already paid for. The ``customer.subscription.deleted``
    webhook fires at period end and downgrades the DB tier to BASIC.

    Returns True if at least one subscription was found and scheduled for cancellation,
    False if the customer had no active/trialing subscriptions (e.g., admin-granted tier
    with no associated Stripe subscription). When False, the caller should update the
    DB tier directly since no webhook will fire to do it.

    Raises stripe.StripeError if any modification fails, so the caller can avoid
    updating the DB tier when Stripe is inconsistent.
    """
    # Guard: only proceed if the user already has a Stripe customer ID.  Calling
    # get_stripe_customer_id for a user who has never had a paid subscription would
    # create an orphaned, potentially-billable Stripe Customer object — we avoid that
    # by returning False early so the caller can downgrade the DB tier directly.
    user = await get_user_by_id(user_id)
    if not user.stripe_customer_id:
        return False

    customer_id = user.stripe_customer_id
    try:
        cancelled_count = await _cancel_customer_subscriptions(
            customer_id, at_period_end=True
        )
        if cancelled_count > 0:
            get_pending_subscription_change.cache_delete(user_id)
        return cancelled_count > 0
    except stripe.StripeError:
        logger.warning(
            "cancel_stripe_subscription: Stripe error while cancelling subs for user %s",
            user_id,
        )
        raise


async def get_proration_credit_cents(user_id: str, monthly_cost_cents: int) -> int:
    """Return the prorated credit (in cents) the user would receive if they upgraded now.

    Fetches the user's active Stripe subscription to determine how many seconds
    remain in the current billing period, then calculates the unused portion of
    the monthly cost. Returns 0 for BASIC/ENTERPRISE users or when no active sub
    is found.
    """
    if monthly_cost_cents <= 0:
        return 0
    # Guard: only query Stripe if the user already has a customer ID.  Admin-granted
    # paid tiers have no Stripe record; calling get_stripe_customer_id would create an
    # orphaned customer on every billing-page load for those users.
    user = await get_user_by_id(user_id)
    if not user.stripe_customer_id:
        return 0
    try:
        customer_id = user.stripe_customer_id
        subscriptions = await run_in_threadpool(
            stripe.Subscription.list, customer=customer_id, status="active", limit=1
        )
        if not subscriptions.data:
            return 0
        sub = subscriptions.data[0]
        period_start: int = sub["current_period_start"]
        period_end: int = sub["current_period_end"]
        now = int(time.time())
        total_seconds = period_end - period_start
        remaining_seconds = max(period_end - now, 0)
        if total_seconds <= 0:
            return 0
        return int(monthly_cost_cents * remaining_seconds / total_seconds)
    except Exception:
        logger.warning(
            "get_proration_credit_cents: failed to compute proration for user %s",
            user_id,
        )
        return 0


# Ordered from least- to most-privileged. Used to distinguish upgrades
# (move right) from downgrades (move left); ENTERPRISE is admin-managed and
# never reached via self-service flows.
_TIER_ORDER: tuple[SubscriptionTier, ...] = (
    SubscriptionTier.NO_TIER,
    SubscriptionTier.BASIC,
    SubscriptionTier.PRO,
    SubscriptionTier.MAX,
    SubscriptionTier.BUSINESS,
    SubscriptionTier.ENTERPRISE,
)


def _tier_rank(tier: SubscriptionTier) -> int:
    return _TIER_ORDER.index(tier)


def is_tier_upgrade(current: SubscriptionTier, target: SubscriptionTier) -> bool:
    return _tier_rank(target) > _tier_rank(current)


def is_tier_downgrade(current: SubscriptionTier, target: SubscriptionTier) -> bool:
    return _tier_rank(target) < _tier_rank(current)


class PendingChangeUnknown(Exception):
    """Raised when pending-change state cannot be determined (e.g. LaunchDarkly
    price-id lookup failed). Propagates past the @cached wrapper so the next
    request retries instead of serving a stale `None` for the TTL window."""


async def _get_active_subscription(customer_id: str) -> stripe.Subscription | None:
    """Return the customer's active or trialing subscription, or None."""
    for status in ("active", "trialing"):
        subs = await stripe.Subscription.list_async(
            customer=customer_id, status=status, limit=1
        )
        if subs.data:
            return subs.data[0]
    return None


async def get_user_billing_cycle(user_id: str) -> BillingCycle | None:
    """Return the billing cycle ("monthly"/"yearly") of the user's active sub.

    Resolves cycle by matching the active subscription's price ID against the
    LaunchDarkly-configured monthly/yearly price IDs for the user's current
    tier, falling back to scanning every priceable tier (handles the brief
    window during a tier change where DB tier and Stripe price disagree).
    Returns None when there's no Stripe customer, no active sub, or the price
    ID doesn't match any configured cycle (e.g. legacy unconfigured price).
    """
    user = await get_user_by_id(user_id)
    if not user.stripe_customer_id:
        return None
    try:
        sub = await _get_active_subscription(user.stripe_customer_id)
    except stripe.StripeError:
        logger.warning(
            "get_user_billing_cycle: Stripe lookup failed for user %s", user_id
        )
        return None
    if sub is None:
        return None
    items = sub["items"].data
    if not items:
        return None
    price = items[0].price
    current_price_id = price if isinstance(price, str) else price.id
    if not current_price_id:
        return None

    priceable = (
        SubscriptionTier.BASIC,
        SubscriptionTier.PRO,
        SubscriptionTier.MAX,
        SubscriptionTier.BUSINESS,
    )
    monthly_prices, yearly_prices = await asyncio.gather(
        asyncio.gather(*[get_subscription_price_id(t, "monthly") for t in priceable]),
        asyncio.gather(*[get_subscription_price_id(t, "yearly") for t in priceable]),
    )
    price_to_cycle: dict[str, BillingCycle] = {}
    for pid in monthly_prices:
        if pid:
            price_to_cycle[pid] = "monthly"
    for pid in yearly_prices:
        if pid:
            price_to_cycle[pid] = "yearly"
    return price_to_cycle.get(current_price_id)


async def get_active_subscription_period_end(user_id: str) -> int | None:
    """Return the Unix timestamp of the active sub's current_period_end, or None.

    Used to surface "next invoice on {date}" in upgrade dialog UX. Returns None
    for users without a Stripe customer or active sub. Stripe failures swallow
    to None — UX falls back to generic copy if the lookup misfires.
    """
    user = await get_user_by_id(user_id)
    if not user.stripe_customer_id:
        return None
    try:
        sub = await _get_active_subscription(user.stripe_customer_id)
    except stripe.StripeError:
        logger.warning(
            "get_active_subscription_period_end: Stripe lookup failed for user %s",
            user_id,
        )
        return None
    if sub is None:
        return None
    period_end = sub.current_period_end
    return int(period_end) if period_end else None


# Substrings Stripe uses in InvalidRequestError messages when the schedule is
# already in a terminal state (released / completed / canceled) and therefore
# cannot be released again. We only swallow the error when one of these appears;
# anything else (typo'd schedule id, wrong subscription, 404, etc.) must
# propagate so bugs aren't masked as silent no-ops.
_TERMINAL_SCHEDULE_ERROR_SUBSTRINGS = (
    "already been released",
    "already released",
    "already been completed",
    "already completed",
    "already been canceled",
    "already been cancelled",
    "already canceled",
    "already cancelled",
    "is not active",
    "is not in a state",
)


async def _release_schedule_ignoring_terminal(
    schedule_id: str, log_context: str
) -> bool:
    """Release a Stripe schedule; swallow InvalidRequestError on terminal state.

    Returns True if the release call succeeded, False if the schedule was
    already in a terminal (released / completed / canceled) state. Any other
    Stripe error — including non-terminal InvalidRequestErrors such as typo'd
    ids or 404s — propagates so the caller can surface the failure instead of
    silently masking a bug.
    """
    try:
        await stripe.SubscriptionSchedule.release_async(schedule_id)
        return True
    except stripe.InvalidRequestError as e:
        message = getattr(e, "user_message", None) or str(e)
        if not any(
            marker in message.lower() for marker in _TERMINAL_SCHEDULE_ERROR_SUBSTRINGS
        ):
            logger.warning(
                "%s: schedule %s release failed with non-terminal"
                " InvalidRequestError (%s); re-raising",
                log_context,
                schedule_id,
                message,
            )
            raise
        logger.warning(
            "%s: schedule %s not releasable (%s); treating as already released",
            log_context,
            schedule_id,
            message,
        )
        return False


async def _schedule_downgrade_at_period_end(
    sub: stripe.Subscription,
    new_price_id: str,
    user_id: str,
    tier: SubscriptionTier,
) -> None:
    """Create a Subscription Schedule that defers a tier change to period end.

    Stripe's Subscription Schedule drives an existing subscription through a
    series of phases. By keeping the current price for the remainder of the
    billing period and switching to ``new_price_id`` afterwards, the user does
    NOT receive an immediate proration charge and keeps their current tier
    until period end.

    Stripe allows at most one active schedule per subscription and rejects
    ``SubscriptionSchedule.create`` if either (a) a schedule is already
    attached to the subscription or (b) ``cancel_at_period_end=True`` is set.
    Both conditions mean the user is overwriting a pending change they made
    earlier (e.g. BUSINESS→BASIC cancel, now switching to BUSINESS→PRO
    downgrade). We clear the conflicting state first so the new schedule can
    be created. These defensive reads serialize through Stripe's own atomic
    operations — by the time modify/release returns, the subscription is in a
    known-clean state for the subsequent create.
    """
    sub_id = sub.id
    # ``sub["items"]`` (dict-item) rather than ``sub.items`` because the latter
    # is shadowed by Python's dict.items() method on StripeObject.
    items = sub["items"].data
    if not items:
        raise ValueError(f"Subscription {sub_id} has no items; cannot schedule")
    price = items[0].price
    current_price_id = price if isinstance(price, str) else price.id
    period_start: int = sub["current_period_start"]
    period_end: int = sub["current_period_end"]

    if sub.cancel_at_period_end:
        await stripe.Subscription.modify_async(sub_id, cancel_at_period_end=False)
        logger.info(
            "_schedule_downgrade_at_period_end: cleared cancel_at_period_end"
            " on sub %s for user %s before scheduling downgrade",
            sub_id,
            user_id,
        )
    if sub.schedule:
        existing_schedule_id = (
            sub.schedule if isinstance(sub.schedule, str) else sub.schedule.id
        )
        await _release_schedule_ignoring_terminal(
            existing_schedule_id, "_schedule_downgrade_at_period_end"
        )

    # Create + modify as a two-step transaction. If modify fails (network,
    # Stripe 500) the created schedule is orphaned AND attached to the
    # subscription, which blocks any future Stripe-side change until manually
    # released. Roll back by releasing the orphan, then re-raise so the caller
    # sees the original failure.
    schedule = await stripe.SubscriptionSchedule.create_async(from_subscription=sub_id)
    try:
        await stripe.SubscriptionSchedule.modify_async(
            schedule.id,
            phases=[
                {
                    "items": [{"price": current_price_id, "quantity": 1}],
                    "start_date": period_start,
                    "end_date": period_end,
                    "proration_behavior": "none",
                },
                {
                    "items": [{"price": new_price_id, "quantity": 1}],
                    "proration_behavior": "none",
                },
            ],
            metadata={"user_id": user_id, "pending_tier": tier.value},
        )
    except stripe.StripeError:
        logger.exception(
            "_schedule_downgrade_at_period_end: modify failed for schedule %s"
            " on sub %s user %s; attempting rollback release",
            schedule.id,
            sub_id,
            user_id,
        )
        try:
            await _release_schedule_ignoring_terminal(
                schedule.id, "_schedule_downgrade_at_period_end_rollback"
            )
        except stripe.StripeError:
            logger.exception(
                "_schedule_downgrade_at_period_end: rollback release also failed"
                " for orphaned schedule %s on sub %s user %s; manual cleanup"
                " required",
                schedule.id,
                sub_id,
                user_id,
            )
        raise
    logger.info(
        "modify_stripe_subscription_for_tier: scheduled sub %s downgrade for user %s → %s at %d",
        sub_id,
        user_id,
        tier,
        period_end,
    )


async def modify_stripe_subscription_for_tier(
    user_id: str,
    tier: SubscriptionTier,
    billing_cycle: BillingCycle = "monthly",
) -> bool:
    """Change a Stripe subscription to a new paid tier.

    Upgrades (e.g. PRO→BUSINESS) apply immediately via ``stripe.Subscription.modify``
    with ``proration_behavior="create_prorations"``: Stripe credits unused time on
    the old plan and charges the pro-rated amount for the new plan in the same
    billing cycle.

    Downgrades (e.g. BUSINESS→PRO) are deferred to the end of the current billing
    period via a Stripe Subscription Schedule: the user keeps their current tier
    for the time they already paid for, and the new tier takes effect when the
    next invoice is generated. The DB tier flip happens via the webhook fired
    when the schedule advances to its next phase.

    Returns:
        True  — a subscription was found and modified/scheduled successfully.
        False — no active/trialing subscription exists (e.g. admin-granted tier or
                first-time paid signup); caller should fall back to Checkout.

    Raises stripe.StripeError on API failures so callers can propagate a 502.
    Raises ValueError when no Stripe price ID is configured for the tier.
    """
    price_id = await get_subscription_price_id(tier, billing_cycle)
    if not price_id:
        raise ValueError(f"No Stripe price ID configured for tier {tier}")

    user = await get_user_by_id(user_id)
    if not user.stripe_customer_id:
        return False
    current_tier = user.subscription_tier or SubscriptionTier.NO_TIER

    sub = await _get_active_subscription(user.stripe_customer_id)
    if sub is None:
        return False
    items = sub["items"].data
    if not items:
        return False
    sub_id = sub.id

    # Invalidate the cache unconditionally on exit (success OR failure): any
    # Stripe mutation below — clearing cancel_at_period_end, releasing an old
    # schedule, creating a new one — may have landed partially before an error
    # was raised, and the cached pending-change state would otherwise go stale
    # for up to 30s until the TTL expires.
    try:
        if is_tier_downgrade(current_tier, tier):
            await _schedule_downgrade_at_period_end(sub, price_id, user_id, tier)
            return True

        # Same-tier yearly→monthly is a cycle *downgrade*: the user is moving
        # from a longer commitment to a shorter one. Route it through the
        # period-end schedule so the dialog promise ("no charge today, switch
        # at end of yearly period") actually holds. Same-tier monthly→yearly
        # stays on the immediate proration path below — the user is committing
        # to more time, immediate billing is the correct semantic.
        if current_tier == tier:
            current_price = items[0].price
            current_price_id = (
                current_price if isinstance(current_price, str) else current_price.id
            )
            current_cycle = await _resolve_cycle_for_price_id(current_price_id)
            if current_cycle == "yearly" and billing_cycle == "monthly":
                await _schedule_downgrade_at_period_end(sub, price_id, user_id, tier)
                return True

        # Upgrade path. If a schedule is attached from a previous pending
        # downgrade, release it first — an upgrade expresses the user's
        # intent to be on this tier immediately, which overrides any pending
        # deferred change. Ignore terminal-state errors from release.
        if sub.schedule:
            existing_schedule_id = (
                sub.schedule if isinstance(sub.schedule, str) else sub.schedule.id
            )
            await _release_schedule_ignoring_terminal(
                existing_schedule_id, "modify_stripe_subscription_for_tier"
            )

        # If a paid→BASIC cancel is pending (cancel_at_period_end=True), clear it
        # as part of the upgrade — the user is explicitly choosing to stay on a
        # paid tier. Without this, the sub would be upgraded AND still cancelled
        # at period end, leaving a confusing dual state.
        # always_invoice + error_if_incomplete bill the prorated upgrade now and
        # roll the modify back if the auto-charge fails (instead of deferring).
        # Refresh metadata so the live sub reflects the new tier+cycle — the
        # backend derives tier from price_id, but Stripe-side dashboards and
        # downstream tooling read sub.metadata directly and otherwise see the
        # stale tier/cycle from the original checkout.
        modify_kwargs: dict = {
            "items": [{"id": items[0].id, "price": price_id}],
            "proration_behavior": "always_invoice",
            "payment_behavior": "error_if_incomplete",
            "metadata": {
                "user_id": user_id,
                "tier": tier.value,
                "billing_cycle": billing_cycle,
            },
        }
        if sub.cancel_at_period_end:
            modify_kwargs["cancel_at_period_end"] = False

        await stripe.Subscription.modify_async(sub_id, **modify_kwargs)
        # Flip the DB tier immediately. The customer.subscription.updated webhook
        # will also fire and set it again — idempotent. Without this synchronous
        # update, the UI refetches before the webhook lands and shows the old
        # tier, making the upgrade look like a no-op to the user.
        #
        # Swallow DB-write exceptions here: Stripe is authoritative and the
        # modify above already succeeded (the user has been charged). If the
        # DB write fails and we re-raised, the API would return 5xx and the UI
        # would surface a failed upgrade to a user who was already charged.
        # The customer.subscription.updated webhook will reconcile the DB shortly.
        #
        # Only catch actual DB/connection failures — letting KeyError,
        # AttributeError etc. propagate so programming errors surface in Sentry
        # instead of being silently masked as benign DB-write-swallow events.
        try:
            await set_subscription_tier(user_id, tier)
        except (PrismaError, ConnectionError, asyncio.TimeoutError):
            logger.exception(
                "modify_stripe_subscription_for_tier: Stripe modify on sub %s"
                " succeeded for user %s → %s but DB tier flip failed; webhook"
                " will reconcile",
                sub_id,
                user_id,
                tier,
            )
        logger.info(
            "modify_stripe_subscription_for_tier: upgraded sub %s for user %s → %s",
            sub_id,
            user_id,
            tier,
        )
        return True
    finally:
        get_pending_subscription_change.cache_delete(user_id)


async def release_pending_subscription_schedule(user_id: str) -> bool:
    """Cancel any pending subscription change (scheduled downgrade or cancellation).

    Two pending-change mechanisms can be attached to a Stripe subscription:

    - **Subscription Schedule** (paid→paid downgrade): ``stripe.SubscriptionSchedule.release``
      detaches the schedule and lets the subscription continue on its current
      phase's price.
    - **cancel_at_period_end=True** (paid→BASIC cancel): clearing that flag via
      ``stripe.Subscription.modify`` keeps the subscription active indefinitely.

    Returns True if a pending change was found and reverted, False otherwise.
    """
    user = await get_user_by_id(user_id)
    if not user.stripe_customer_id:
        return False

    sub = await _get_active_subscription(user.stripe_customer_id)
    if sub is None:
        return False

    sub_id = sub.id
    did_anything = False
    schedule_released = False
    schedule_id: str | None = None
    try:
        if sub.schedule:
            schedule_id = (
                sub.schedule if isinstance(sub.schedule, str) else sub.schedule.id
            )
            schedule_released = await _release_schedule_ignoring_terminal(
                schedule_id, "release_pending_subscription_schedule"
            )
            if schedule_released:
                logger.info(
                    "release_pending_subscription_schedule: released schedule %s for user %s",
                    schedule_id,
                    user_id,
                )
                did_anything = True
        if sub.cancel_at_period_end:
            try:
                await stripe.Subscription.modify_async(
                    sub_id, cancel_at_period_end=False
                )
            except stripe.StripeError:
                if schedule_released:
                    logger.exception(
                        "release_pending_subscription_schedule: partial release"
                        " — schedule %s released but cancel_at_period_end clear"
                        " failed on sub %s for user %s; manual reconciliation"
                        " may be needed",
                        schedule_id,
                        sub_id,
                        user_id,
                    )
                raise
            did_anything = True
            logger.info(
                "release_pending_subscription_schedule: cleared cancel_at_period_end"
                " on sub %s for user %s",
                sub_id,
                user_id,
            )
    finally:
        if did_anything:
            get_pending_subscription_change.cache_delete(user_id)
    return did_anything


@cached(ttl_seconds=30, maxsize=512, cache_none=True, shared_cache=True)
async def get_pending_subscription_change(
    user_id: str,
) -> tuple[SubscriptionTier, datetime, BillingCycle | None] | None:
    """Return ``(pending_tier, effective_at, pending_cycle)`` when a change is queued, else ``None``.

    Reflects both Subscription Schedule phase transitions (paid→paid downgrade)
    and ``cancel_at_period_end=True`` (paid→BASIC cancel).

    Cached for 30 seconds per user_id. *Why the cache exists:* this function
    runs on every dashboard/home fetch and would otherwise fire
    2× Subscription.list + 1× Schedule.retrieve per page load. A busy user
    polling the billing page would quickly brush up against Stripe's per-API
    rate limits; the 30s TTL absorbs dashboard polling while being short
    enough that the UI reconciles quickly after a downgrade / cancel action.

    *Invalidation contract.* Every call-site that mutates Stripe state which
    could change the pending-change answer MUST call
    ``get_pending_subscription_change.cache_delete(user_id)`` so the UI never
    shows a stale pending badge after a user-visible action. Current
    invalidators (keep this list in sync when adding new mutators):

    - ``set_subscription_tier`` — admin or webhook-driven tier flip.
    - ``modify_stripe_subscription_for_tier`` — ``finally`` block (covers
      upgrade path clear + downgrade-schedule create + any partial failure).
    - ``release_pending_subscription_schedule`` — ``finally`` block when a
      schedule release OR ``cancel_at_period_end`` clear succeeded.
    - ``cancel_stripe_subscription`` — after scheduling period-end cancel.
    - ``sync_subscription_from_stripe`` — webhook entry point.
    - ``set_user_tier`` (``backend.copilot.rate_limit``) — admin tier override
      invalidates any cached pending state keyed off the old tier.
    """
    user = await get_user_by_id(user_id)
    if not user.stripe_customer_id:
        # Short-circuit for users with no Stripe customer (admin-granted tiers,
        # BASIC-only users): skip the Stripe API calls entirely.
        return None

    priceable = (
        SubscriptionTier.BASIC,
        SubscriptionTier.PRO,
        SubscriptionTier.MAX,
        SubscriptionTier.BUSINESS,
    )
    # Gather monthly + yearly price IDs so a schedule whose next phase points
    # at a yearly price still resolves to the correct tier.
    monthly_prices, yearly_prices = await asyncio.gather(
        asyncio.gather(*[get_subscription_price_id(t, "monthly") for t in priceable]),
        asyncio.gather(*[get_subscription_price_id(t, "yearly") for t in priceable]),
    )
    price_to_tier: dict[str, SubscriptionTier] = {}
    price_to_cycle: dict[str, BillingCycle] = {}
    for t, pid in zip(priceable, monthly_prices):
        if pid:
            price_to_tier[pid] = t
            price_to_cycle[pid] = "monthly"
    for t, pid in zip(priceable, yearly_prices):
        if pid:
            price_to_tier[pid] = t
            price_to_cycle[pid] = "yearly"
    if not price_to_tier:
        logger.warning(
            "get_pending_subscription_change: no Stripe price IDs resolvable for"
            " BASIC/PRO/MAX/BUSINESS (LaunchDarkly fetch failed?); raising to bypass"
            " the None cache so the next request retries fresh"
        )
        raise PendingChangeUnknown(
            "Stripe price lookup failed; pending-change state cannot be determined"
        )

    sub = await _get_active_subscription(user.stripe_customer_id)
    if sub is None:
        return None
    period_end = sub.current_period_end
    if not isinstance(period_end, int):
        return None
    effective_at = datetime.fromtimestamp(period_end, tz=timezone.utc)
    if sub.cancel_at_period_end:
        return SubscriptionTier.NO_TIER, effective_at, None
    if not sub.schedule:
        return None
    schedule_id = sub.schedule if isinstance(sub.schedule, str) else sub.schedule.id
    schedule = await stripe.SubscriptionSchedule.retrieve_async(schedule_id)
    return _next_phase_tier_and_start(schedule, price_to_tier, price_to_cycle)


def _next_phase_tier_and_start(
    schedule: stripe.SubscriptionSchedule,
    price_to_tier: dict[str, SubscriptionTier],
    price_to_cycle: dict[str, BillingCycle],
) -> tuple[SubscriptionTier, datetime, BillingCycle | None] | None:
    """Return ``(tier, start_datetime, billing_cycle)`` of the phase following the active one.

    ``billing_cycle`` is the cycle of the next-phase price (``"monthly"``/``"yearly"``)
    when resolvable, ``None`` for unconfigured/legacy prices. Same-tier yearly→monthly
    schedules need this so the UI can distinguish a cycle-only change (where
    ``pending_tier == current_tier``) from a real tier downgrade.

    Using the phase's own ``start_date`` (not the subscription's current_period_end)
    is correct even for schedules created outside this flow — a dashboard-authored
    schedule can have phase transitions at arbitrary timestamps.
    """
    now = int(time.time())
    for phase in schedule.phases or []:
        if not isinstance(phase.start_date, int) or phase.start_date <= now:
            continue
        # ``phase["items"]`` because ``phase.items`` is shadowed by dict.items().
        items = phase["items"] or []
        if not items:
            continue
        price = items[0].price
        price_id = price if isinstance(price, str) else price.id
        if price_id in price_to_tier:
            return (
                price_to_tier[price_id],
                datetime.fromtimestamp(phase.start_date, tz=timezone.utc),
                price_to_cycle.get(price_id),
            )
        logger.warning(
            "next_phase_tier_and_start: unknown price %s on schedule %s",
            price_id,
            schedule.id,
        )
    return None


async def get_auto_top_up(user_id: str) -> AutoTopUpConfig:
    user = await get_user_by_id(user_id)

    if not user.top_up_config:
        return AutoTopUpConfig(threshold=0, amount=0)

    return AutoTopUpConfig.model_validate(user.top_up_config)


async def _resolve_cycle_for_price_id(price_id: str | None) -> BillingCycle | None:
    """Map a Stripe price ID back to its billing cycle via the LD price flag.

    Used by ``modify_stripe_subscription_for_tier`` to detect a same-tier
    cycle change (yearly→monthly downgrade vs monthly→yearly upgrade) before
    deciding between the immediate-proration path and the period-end schedule.
    Returns None when the price ID isn't configured for any tier+cycle (legacy
    or unconfigured price), which keeps the caller on the upgrade path.
    """
    if not price_id:
        return None
    priceable = (
        SubscriptionTier.BASIC,
        SubscriptionTier.PRO,
        SubscriptionTier.MAX,
        SubscriptionTier.BUSINESS,
    )
    monthly_prices, yearly_prices = await asyncio.gather(
        asyncio.gather(*[get_subscription_price_id(t, "monthly") for t in priceable]),
        asyncio.gather(*[get_subscription_price_id(t, "yearly") for t in priceable]),
    )
    if price_id in monthly_prices:
        return "monthly"
    if price_id in yearly_prices:
        return "yearly"
    return None


def _ld_price_key(tier: SubscriptionTier, billing_cycle: BillingCycle) -> str:
    """Compose the LaunchDarkly key for a tier+cycle.

    Monthly keeps the legacy ``<TIER>`` key (so a flag value flipped from
    monthly-only to also-yearly never breaks an older deploy that still reads
    only the monthly key). Yearly lives under ``<TIER>_YEARLY``.
    """
    if billing_cycle == "yearly":
        return f"{tier.value}_YEARLY"
    return tier.value


@cached(ttl_seconds=60, maxsize=16, cache_none=False)
async def get_subscription_price_id(
    tier: SubscriptionTier, billing_cycle: BillingCycle = "monthly"
) -> str | None:
    """Return Stripe Price ID for a tier+cycle from LaunchDarkly, cached 60s.

    Reads the ``copilot-tier-stripe-prices`` JSON flag and looks up:

    - Monthly: ``raw["<TIER>"]`` (e.g. ``raw["PRO"]``) — the existing key
      pre-yearly. Older deploys see this exact key, so adding yearly keys
      alongside it never breaks an in-flight rollout.
    - Yearly: ``raw["<TIER>_YEARLY"]`` (e.g. ``raw["PRO_YEARLY"]``). Yearly
      requests for a tier without a configured yearly key fail closed
      (return ``None``) instead of silently falling back to the monthly
      price.

    ``cache_none=False`` prevents a transient LD failure from caching ``None``
    and blocking subscription upgrades for the full 60-second TTL window.
    """
    raw = await get_feature_flag_value(
        Flag.COPILOT_TIER_STRIPE_PRICES.value, user_id="system", default=None
    )
    if raw is None:
        return None
    if not isinstance(raw, dict):
        logger.warning(
            "Invalid LD value for copilot-tier-stripe-prices (expected JSON object): %r",
            raw,
        )
        return None
    price_id = raw.get(_ld_price_key(tier, billing_cycle))
    return price_id if isinstance(price_id, str) and price_id else None


async def _expire_open_subscription_sessions(customer_id: str) -> None:
    """Expire open subscription checkout sessions for the customer.

    An abandoned subscription session leaves an incomplete subscription + open invoice
    in Stripe. Expiring it triggers Stripe to cancel that subscription and void the
    invoice, so the user is not shown phantom charges on their billing page.
    """
    try:
        starting_after: str | None = None
        while True:
            list_kwargs: dict = {
                "customer": customer_id,
                "status": "open",
                "limit": 100,
            }
            if starting_after:
                list_kwargs["starting_after"] = starting_after
            sessions = await stripe.checkout.Session.list_async(**list_kwargs)
            for s in sessions.data:
                if s.mode == "subscription":
                    try:
                        await stripe.checkout.Session.expire_async(s.id)
                    except stripe.StripeError:
                        logger.warning(
                            "create_subscription_checkout: could not expire session %s",
                            s.id,
                        )
            if not sessions.has_more or not sessions.data:
                break
            starting_after = sessions.data[-1].id
    except Exception:
        logger.warning(
            "create_subscription_checkout: could not list open sessions for %s",
            customer_id,
        )


async def reconcile_stripe_tier_for_user(user_id: str) -> bool:
    """Check Stripe for an active subscription and sync tier if found.

    Called as a lazy fallback when a user is on NO_TIER to recover from
    missed webhooks. Returns True if an active subscription was found and
    synced. Does NOT create a Stripe customer — only checks existing ones.
    """
    user = await get_user_by_id(user_id)
    if not user.stripe_customer_id:
        return False
    try:
        sub = await _get_active_subscription(user.stripe_customer_id)
    except Exception:
        logger.warning(
            "reconcile_stripe_tier_for_user: Stripe lookup failed for user %s",
            user_id[:8],
        )
        return False
    if sub is None:
        return False
    await sync_subscription_from_stripe(dict(sub))
    return True


async def sync_tier_from_checkout_session(data_object: dict) -> None:
    """Sync subscription tier from a checkout.session.completed event payload.

    Retrieves the Stripe subscription and calls sync_subscription_from_stripe so
    the tier is set immediately without waiting for customer.subscription.created.
    No-op when mode != "subscription" or subscription ID is absent.
    Raises stripe.StripeError if the subscription cannot be retrieved.
    """
    if data_object.get("mode") != "subscription":
        return
    sub_id = data_object.get("subscription")
    if not sub_id:
        return
    sub = await stripe.Subscription.retrieve_async(sub_id)
    await sync_subscription_from_stripe(dict(sub))


async def create_subscription_checkout(
    user_id: str,
    tier: SubscriptionTier,
    success_url: str,
    cancel_url: str,
    billing_cycle: BillingCycle = "monthly",
) -> str:
    """Create a Stripe Checkout Session for a subscription. Returns the redirect URL."""
    price_id = await get_subscription_price_id(tier, billing_cycle)
    if not price_id:
        raise ValueError(f"Subscription not available for tier {tier.value}")
    customer_id = await get_stripe_customer_id(user_id)
    await _expire_open_subscription_sessions(customer_id)
    session = await run_in_threadpool(
        stripe.checkout.Session.create,
        customer=customer_id,
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=success_url,
        cancel_url=cancel_url,
        subscription_data={
            "metadata": {
                "user_id": user_id,
                "tier": tier.value,
                "billing_cycle": billing_cycle,
            }
        },
        allow_promotion_codes=True,
        automatic_tax={"enabled": True},
        billing_address_collection="auto",
        customer_update={"address": "auto"},
    )
    if not session.url:
        # An empty checkout URL for a paid upgrade is always an error; surfacing it
        # as ValueError means the API handler returns 422 instead of silently
        # redirecting the client to an empty URL.
        raise ValueError("Stripe did not return a checkout session URL")
    return session.url


async def _cleanup_stale_subscriptions(customer_id: str, new_sub_id: str) -> None:
    """Best-effort cancel of any active subs for the customer other than new_sub_id.

    Called from the webhook handler after a new subscription becomes active. Failures
    are logged but not raised so a transient Stripe error doesn't crash the webhook —
    a periodic reconciliation job is the intended backstop for persistent drift.

    NOTE: until that reconcile job lands, a failure here means the user is silently
    billed for two simultaneous subscriptions. The error log below is intentionally
    `logger.exception` so it surfaces in Sentry with the customer/sub IDs needed to
    manually reconcile, and the metric `stripe_stale_subscription_cleanup_failed`
    is bumped so on-call can alert on persistent drift.
    TODO(#stripe-reconcile-job): replace this best-effort cleanup with a periodic
    reconciliation job that queries Stripe for customers with >1 active sub.
    """
    try:
        await _cancel_customer_subscriptions(customer_id, exclude_sub_id=new_sub_id)
    except stripe.StripeError:
        # Use exception() (not warning) so this surfaces as an error in Sentry —
        # any failure here means a paid-to-paid upgrade may have left the user
        # with two simultaneous active subscriptions.
        logger.exception(
            "stripe_stale_subscription_cleanup_failed: customer=%s new_sub=%s —"
            " user may be billed for two simultaneous subscriptions; manual"
            " reconciliation required",
            customer_id,
            new_sub_id,
        )


async def sync_subscription_from_stripe(stripe_subscription: dict) -> None:
    """Update User.subscriptionTier from a Stripe subscription object.

    Expected shape of stripe_subscription (subset of Stripe's Subscription object):
        customer: str                  — Stripe customer ID
        status:   str                  — "active" | "trialing" | "canceled" | ...
        id:       str                  — Stripe subscription ID
        items.data[].price.id: str     — Stripe price ID identifying the tier
    """
    customer_id = stripe_subscription.get("customer")
    if not customer_id:
        logger.warning(
            "sync_subscription_from_stripe: missing 'customer' field in event, "
            "skipping (keys: %s)",
            list(stripe_subscription.keys()),
        )
        return
    user = await User.prisma().find_first(where={"stripeCustomerId": customer_id})
    if not user:
        logger.warning(
            "sync_subscription_from_stripe: no user for customer %s", customer_id
        )
        return
    # Cross-check: if the subscription carries a metadata.user_id (set during
    # Checkout Session creation), verify it matches the user we found via
    # stripeCustomerId.  A mismatch indicates a customer↔user mapping
    # inconsistency — updating the wrong user's tier would be a data-corruption
    # bug, so we log loudly and bail out.  Absence of metadata.user_id (e.g.
    # subscriptions created outside the Checkout flow) is not an error — we
    # simply skip the check and proceed with the customer-ID-based lookup.
    metadata = stripe_subscription.get("metadata") or {}
    metadata_user_id = metadata.get("user_id") if isinstance(metadata, dict) else None
    if metadata_user_id and metadata_user_id != user.id:
        logger.error(
            "sync_subscription_from_stripe: metadata.user_id=%s does not match"
            " user.id=%s found via stripeCustomerId=%s — refusing to update tier"
            " to avoid corrupting the wrong user's subscription state",
            metadata_user_id,
            user.id,
            customer_id,
        )
        return
    # ENTERPRISE tiers are admin-managed. Never let a Stripe webhook flip an
    # ENTERPRISE user to a different tier — if a user on ENTERPRISE somehow has
    # a self-service Stripe sub, it's a data-consistency issue for an operator,
    # not something the webhook should automatically "fix".
    current_tier = user.subscriptionTier or SubscriptionTier.NO_TIER
    if current_tier == SubscriptionTier.ENTERPRISE:
        logger.warning(
            "sync_subscription_from_stripe: refusing to overwrite ENTERPRISE tier"
            " for user %s (customer %s); event status=%s",
            user.id,
            customer_id,
            stripe_subscription.get("status", ""),
        )
        return
    status = stripe_subscription.get("status", "")
    new_sub_id = stripe_subscription.get("id", "")
    if status in ("active", "trialing"):
        price_id = ""
        items = stripe_subscription.get("items", {}).get("data", [])
        if items:
            price_id = items[0].get("price", {}).get("id", "")
        priceable = (
            SubscriptionTier.BASIC,
            SubscriptionTier.PRO,
            SubscriptionTier.MAX,
            SubscriptionTier.BUSINESS,
        )
        # Gather monthly + yearly price IDs for every priceable tier so a user
        # on a yearly plan still maps back to the correct tier.
        prices = await asyncio.gather(
            *[get_subscription_price_id(t, "monthly") for t in priceable],
            *[get_subscription_price_id(t, "yearly") for t in priceable],
        )
        price_to_tier: dict[str, SubscriptionTier] = {}
        for t, pid in zip(priceable + priceable, prices):
            if pid:
                price_to_tier[pid] = t
        matched = price_to_tier.get(price_id) if price_id else None
        if matched is not None:
            tier = matched
        else:
            # Unknown or unconfigured price ID — preserve the user's current tier
            # rather than defaulting to BASIC. This prevents accidental downgrades
            # during a price migration or when LD flags are not yet configured.
            logger.warning(
                "sync_subscription_from_stripe: unknown price %s for customer %s,"
                " preserving current tier",
                price_id,
                customer_id,
            )
            return
    else:
        # A subscription was cancelled or ended. DO NOT unconditionally downgrade
        # to BASIC — Stripe does not guarantee webhook delivery order, so a
        # `customer.subscription.deleted` for the OLD sub can arrive after we've
        # already processed `customer.subscription.created` for a new paid sub.
        # Ask Stripe whether any OTHER active/trialing subs exist for this
        # customer; if they do, keep the user's current tier (the other sub's
        # own event will/has already set the correct tier).
        try:
            other_subs_active, other_subs_trialing = await asyncio.gather(
                run_in_threadpool(
                    stripe.Subscription.list,
                    customer=customer_id,
                    status="active",
                    limit=10,
                ),
                run_in_threadpool(
                    stripe.Subscription.list,
                    customer=customer_id,
                    status="trialing",
                    limit=10,
                ),
            )
        except stripe.StripeError:
            logger.warning(
                "sync_subscription_from_stripe: could not verify other active"
                " subs for customer %s on cancel event %s; preserving current"
                " tier to avoid an unsafe downgrade",
                customer_id,
                new_sub_id,
            )
            return
        # Filter out the cancelled subscription to check if other active subs
        # exist. When new_sub_id is empty (malformed event with no 'id' field),
        # we cannot safely exclude any sub — preserve current tier to avoid
        # an unsafe downgrade on a malformed webhook payload.
        if not new_sub_id:
            logger.warning(
                "sync_subscription_from_stripe: cancel event missing 'id' field"
                " for customer %s; preserving current tier",
                customer_id,
            )
            return
        other_active_ids = {sub["id"] for sub in other_subs_active.data} - {new_sub_id}
        other_trialing_ids = {sub["id"] for sub in other_subs_trialing.data} - {
            new_sub_id
        }
        still_has_active_sub = bool(other_active_ids or other_trialing_ids)
        if still_has_active_sub:
            logger.info(
                "sync_subscription_from_stripe: sub %s cancelled but customer %s"
                " still has another active sub; keeping tier %s",
                new_sub_id,
                customer_id,
                current_tier.value,
            )
            return
        tier = SubscriptionTier.NO_TIER
    # Idempotency: Stripe retries webhooks on delivery failure, and several event
    # types map to the same final tier. Skip the DB write + cache invalidation
    # when the tier is already correct to avoid redundant writes on replay.
    if current_tier == tier:
        return
    # When a new subscription becomes active (e.g. paid-to-paid tier upgrade
    # via a fresh Checkout Session), cancel any OTHER active subscriptions for
    # the same customer so the user isn't billed twice. We do this in the
    # webhook rather than the API handler so that abandoning the checkout
    # doesn't leave the user without a subscription.
    # IMPORTANT: this runs AFTER the idempotency check above so that webhook
    # replays for an already-applied event do NOT trigger another cleanup round
    # (which could otherwise cancel a legitimately new subscription the user
    # signed up for between the original event and its replay).
    if status in ("active", "trialing") and new_sub_id:
        # NOTE: paid-to-paid upgrade race (e.g. PRO → BUSINESS):
        # _cleanup_stale_subscriptions cancels the old PRO sub before
        # set_subscription_tier writes BUSINESS to the DB.  If Stripe delivers
        # the PRO `customer.subscription.deleted` event concurrently and it
        # processes after the PRO cancel but before set_subscription_tier
        # commits, the user could momentarily appear as BASIC in the DB.
        # This window is very short in practice (two sequential awaits),
        # but is a known limitation of the current webhook-driven approach.
        # A future improvement would be to write the new tier first, then
        # cancel the old sub.
        await _cleanup_stale_subscriptions(customer_id, new_sub_id)
    await set_subscription_tier(user.id, tier)
    # Tier changed — bust any cached pending-change view so the next
    # dashboard fetch reflects the new state immediately.
    get_pending_subscription_change.cache_delete(user.id)


async def sync_subscription_schedule_from_stripe(stripe_schedule: dict) -> None:
    """Sync the DB tier from a ``subscription_schedule.*`` webhook event.

    Stripe fires ``subscription_schedule.released`` / ``.completed`` /
    ``.updated`` when a schedule advances phases or is detached. The regular
    ``customer.subscription.updated`` webhook with the new price covers the
    phase transition in most cases, but listening to schedule events is a
    safety net that also catches releases done via the Stripe dashboard.

    The schedule payload doesn't carry the active price directly — it carries
    a ``subscription`` id that we look up to get the current item.

    Webhook-ordering safety: we deliberately funnel both event sources through
    ``sync_subscription_from_stripe`` so they share one code path and one DB
    write. That function is idempotent — it no-ops when ``current_tier ==
    tier`` — so concurrent or out-of-order deliveries of
    ``subscription_schedule.*`` and ``customer.subscription.updated`` converge
    to the same DB state regardless of which arrives first.
    """
    # When a schedule is released, Stripe clears `subscription` and moves the id
    # to `released_subscription`. Fall back to that so `.released` events — the
    # main reason we listen to schedule webhooks as a safety net — are processed.
    sub_id = stripe_schedule.get("subscription") or stripe_schedule.get(
        "released_subscription"
    )
    if not isinstance(sub_id, str) or not sub_id:
        logger.warning(
            "sync_subscription_schedule_from_stripe: no 'subscription' id; skipping"
        )
        return
    try:
        sub = await stripe.Subscription.retrieve_async(sub_id)
    except stripe.StripeError:
        logger.warning(
            "sync_subscription_schedule_from_stripe: failed to retrieve sub %s",
            sub_id,
        )
        return
    await sync_subscription_from_stripe(dict(sub))


def _invoice_subscription_id(invoice: dict) -> str:
    """Resolve the subscription ID from a Stripe Invoice payload.

    Stripe API ≥2025-04-01 deprecated the top-level ``invoice.subscription``
    field; subscription invoices now carry it at
    ``invoice.parent.subscription_details.subscription``. Read the new path
    first and fall back to the legacy field so older API versions still work.
    Returns "" when neither is set (one-off invoices, etc.).
    """
    parent = invoice.get("parent") or {}
    if isinstance(parent, dict):
        details = parent.get("subscription_details") or {}
        if isinstance(details, dict):
            new_sub = details.get("subscription")
            if isinstance(new_sub, str) and new_sub:
                return new_sub
    legacy = invoice.get("subscription")
    return legacy if isinstance(legacy, str) and legacy else ""


async def handle_subscription_payment_failure(invoice: dict) -> None:
    """Handle a failed Stripe subscription payment.

    Tries to cover the invoice amount from the user's credit balance.

    - Balance sufficient  → deduct from balance, then pay the Stripe invoice so
      Stripe stops retrying it. The sub stays intact and the user keeps their tier.
    - Balance insufficient → cancel Stripe sub immediately, downgrade to BASIC.
      Cancelling here avoids further Stripe retries on an invoice we cannot cover.
    """
    customer_id = invoice.get("customer")
    if not customer_id:
        logger.warning(
            "handle_subscription_payment_failure: missing customer in invoice; skipping"
        )
        return

    user = await User.prisma().find_first(where={"stripeCustomerId": customer_id})
    if not user:
        logger.warning(
            "handle_subscription_payment_failure: no user found for customer %s",
            customer_id,
        )
        return

    current_tier = user.subscriptionTier or SubscriptionTier.NO_TIER
    if current_tier == SubscriptionTier.ENTERPRISE:
        logger.warning(
            "handle_subscription_payment_failure: skipping ENTERPRISE user %s"
            " (customer %s) — tier is admin-managed",
            user.id,
            customer_id,
        )
        return

    amount_due: int = invoice.get("amount_due", 0)
    sub_id = _invoice_subscription_id(invoice)
    invoice_id: str = invoice.get("id", "")

    if amount_due <= 0:
        logger.info(
            "handle_subscription_payment_failure: amount_due=%d for user %s;"
            " nothing to deduct",
            amount_due,
            user.id,
        )
        return

    credit_model = UserCredit()
    try:
        await credit_model._add_transaction(
            user_id=user.id,
            amount=-amount_due,
            transaction_type=CreditTransactionType.SUBSCRIPTION,
            fail_insufficient_credits=True,
            # Use invoice_id as the idempotency key so that Stripe webhook retries
            # (e.g. on a transient stripe.Invoice.pay failure) do not double-charge.
            transaction_key=invoice_id or None,
            metadata=SafeJson(
                {
                    "stripe_customer_id": customer_id,
                    "stripe_subscription_id": sub_id,
                    "reason": "subscription_payment_failure_covered_by_balance",
                }
            ),
        )
        # Balance covered the invoice. Pay the Stripe invoice with
        # ``paid_out_of_band=True`` so Stripe marks the invoice paid without
        # retrying the card charge — the card already failed and the user is
        # paying via their AutoGPT balance, so a card retry here would
        # double-bill the user (card charge + balance debit). Stripe still
        # fires ``invoice.payment_succeeded`` on the transition; the success
        # handler reads ``paid_out_of_band`` and skips the credit grant so
        # the balance debit isn't reversed.
        if invoice_id:
            try:
                await run_in_threadpool(
                    stripe.Invoice.pay, invoice_id, paid_out_of_band=True
                )
            except stripe.StripeError:
                logger.warning(
                    "handle_subscription_payment_failure: balance deducted for user"
                    " %s but failed to mark invoice %s as paid; Stripe may retry",
                    user.id,
                    invoice_id,
                )
        logger.info(
            "handle_subscription_payment_failure: deducted %d cents from balance"
            " for user %s; Stripe invoice %s paid, sub %s intact, tier preserved",
            amount_due,
            user.id,
            invoice_id,
            sub_id,
        )
    except InsufficientBalanceError:
        # Balance insufficient — cancel Stripe subscription first, then downgrade DB.
        # Order matters: if we downgrade the DB first and the Stripe cancel fails, the
        # user is permanently stuck on BASIC while Stripe continues billing them.
        # Cancelling Stripe first is safe: if the DB write then fails, the webhook
        # customer.subscription.deleted will fire and correct the tier eventually.
        logger.info(
            "handle_subscription_payment_failure: insufficient balance for user %s;"
            " cancelling Stripe sub %s then downgrading to BASIC",
            user.id,
            sub_id,
        )
        try:
            await _cancel_customer_subscriptions(customer_id)
        except stripe.StripeError:
            logger.warning(
                "handle_subscription_payment_failure: failed to cancel Stripe sub %s"
                " for user %s (customer %s); skipping tier downgrade to avoid"
                " inconsistency — Stripe may continue retrying the invoice",
                sub_id,
                user.id,
                customer_id,
            )
            return
        await set_subscription_tier(user.id, SubscriptionTier.NO_TIER)


async def handle_subscription_payment_success(invoice: dict) -> None:
    """Grant AutoGPT credits equal to the paid Stripe invoice amount.

    Fires on every paid subscription invoice (initial signup, monthly renewal,
    and prorated upgrade charges). Credits = ``invoice.amount_paid`` cents,
    keyed by ``invoice_id`` for idempotency so Stripe retries don't double-grant.

    Skipped:
    - Non-subscription invoices (no ``subscription`` field).
    - Zero-amount invoices (e.g. card-validation checks, $0 trials).
    - ENTERPRISE users (admin-managed; they don't pay via self-service).
    """
    customer_id = invoice.get("customer")
    if not customer_id:
        logger.warning(
            "handle_subscription_payment_success: missing customer in invoice; skipping"
        )
        return
    sub_id = _invoice_subscription_id(invoice)
    if not sub_id:
        # Non-subscription invoices (one-off invoices, etc.) — no credit grant.
        return
    user = await User.prisma().find_first(where={"stripeCustomerId": customer_id})
    if not user:
        logger.warning(
            "handle_subscription_payment_success: no user for customer %s",
            customer_id,
        )
        return
    if (
        user.subscriptionTier or SubscriptionTier.NO_TIER
    ) == SubscriptionTier.ENTERPRISE:
        logger.warning(
            "handle_subscription_payment_success: skipping ENTERPRISE user %s"
            " (customer %s) — tier is admin-managed",
            user.id,
            customer_id,
        )
        return

    amount_paid: int = invoice.get("amount_paid", 0)
    invoice_id: str = invoice.get("id", "")
    if amount_paid <= 0 or not invoice_id:
        return

    # Skip when ``handle_subscription_payment_failure`` already covered this
    # invoice from the user's balance and marked it paid out of band — the
    # balance was debited there, granting matching credits here would reverse
    # the debit and give the user a free billing period.
    if invoice.get("paid_out_of_band"):
        logger.info(
            "handle_subscription_payment_success: skipping invoice %s for user %s"
            " (paid_out_of_band — covered by balance in failure handler)",
            invoice_id,
            user.id,
        )
        return

    try:
        await UserCredit()._add_transaction(
            user_id=user.id,
            amount=amount_paid,
            transaction_type=CreditTransactionType.GRANT,
            transaction_key=f"INVOICE-{invoice_id}",
            metadata=SafeJson(
                {
                    "stripe_customer_id": customer_id,
                    "stripe_subscription_id": sub_id,
                    "stripe_invoice_id": invoice_id,
                    "billing_reason": invoice.get("billing_reason", ""),
                    "reason": "subscription_invoice_paid",
                }
            ),
        )
        logger.info(
            "handle_subscription_payment_success: granted %d credits to user %s"
            " for invoice %s (sub %s)",
            amount_paid,
            user.id,
            invoice_id,
            sub_id,
        )
    except UniqueViolationError:
        # Idempotency key collision — Stripe retried this invoice's webhook and
        # we already granted the credits. Safe to ignore.
        return


async def admin_get_user_history(
    page: int = 1,
    page_size: int = 20,
    search: str | None = None,
    transaction_filter: CreditTransactionType | None = None,
    include_inactive: bool = False,
) -> UserHistoryResponse:

    if page < 1 or page_size < 1:
        raise ValueError("Invalid pagination input")

    where_clause: CreditTransactionWhereInput = {}
    # Off by default so phantom rows from abandoned Stripe checkouts aren't surfaced.
    if not include_inactive:
        where_clause["isActive"] = True
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
            # Older _top_up_credits rows wrap reason as {"reason": {"reason": "..."}};
            # unwrap so the dashboard column shows the plain string.
            raw_reason = metadata.get("reason", "No reason provided")
            if isinstance(raw_reason, dict):
                raw_reason = raw_reason.get("reason", "No reason provided")
            reason = str(raw_reason)

        user_credit_model = await get_user_credit_model(tx.userId)
        balance, _ = await user_credit_model._get_credits(tx.userId)

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


# Limits for credit-transaction CSV export. Window cap matches a typical
# finance-month query; row cap protects the API from accidental wide pulls
# (one big tenant can easily exceed 100k rows in 90 days).
CREDIT_EXPORT_MAX_DAYS = 90
CREDIT_EXPORT_MAX_ROWS = 100_000


async def admin_export_user_history(
    start: datetime,
    end: datetime,
    transaction_type: CreditTransactionType | None = None,
    user_id: str | None = None,
    include_inactive: bool = False,
) -> list[UserTransaction]:
    """Return all CreditTransactions in the [start, end] window for export.

    Caps the window at CREDIT_EXPORT_MAX_DAYS and the row count at
    CREDIT_EXPORT_MAX_ROWS — callers should validate the window before calling
    so the user sees a 4xx instead of a silently truncated CSV.

    By default filters out `isActive=False` rows (e.g. abandoned Stripe
    checkouts whose `runningBalance` snapshot never advanced the user's real
    balance).  Pass `include_inactive=True` to surface them when debugging
    why a checkout never completed.
    """
    # Normalize naive datetimes to UTC so direct API callers that send
    # `2026-01-01T00:00:00` (no tz) don't trip a TypeError when subtracted
    # against an aware `2026-01-31T00:00:00Z` partner.
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    if end < start:
        raise ValueError("end must be >= start")
    # Compare timedeltas directly so 90d + any sub-day remainder still trips
    # the cap (.days truncates fractional days and was letting ~91d through).
    if (end - start) > timedelta(days=CREDIT_EXPORT_MAX_DAYS):
        raise ValueError(
            f"Export window must be <= {CREDIT_EXPORT_MAX_DAYS} days "
            f"(got {(end - start).total_seconds() / 86400:.2f} days)"
        )

    where: CreditTransactionWhereInput = {
        "createdAt": {"gte": start, "lte": end},
    }
    if transaction_type:
        where["type"] = transaction_type
    if user_id:
        where["userId"] = user_id
    if not include_inactive:
        where["isActive"] = True

    # Fetch one over the cap and reject — avoids the TOCTOU race a separate
    # count() + take=cap pair would have if rows land between the two queries.
    transactions = await CreditTransaction.prisma().find_many(
        where=where,
        include={"User": True},
        order={"createdAt": "desc"},
        take=CREDIT_EXPORT_MAX_ROWS + 1,
    )
    if len(transactions) > CREDIT_EXPORT_MAX_ROWS:
        raise ValueError(
            f"Export would return more than {CREDIT_EXPORT_MAX_ROWS} rows; "
            "narrow the window or add filters."
        )

    admin_id_to_email: dict[str, str] = {}

    async def _resolve_admin_email(admin_id: str) -> str:
        if admin_id in admin_id_to_email:
            return admin_id_to_email[admin_id]
        email = await get_user_email_by_id(admin_id) or ""
        admin_id_to_email[admin_id] = email
        return email

    history: list[UserTransaction] = []
    for tx in transactions:
        metadata: dict = cast(dict, tx.metadata) or {}
        admin_id = metadata.get("admin_id") or ""
        admin_email = await _resolve_admin_email(admin_id) if admin_id else ""
        # _top_up_credits writes reason as {"reason": "..."}; unwrap so the CSV
        # column carries a plain string regardless of source.
        raw_reason = metadata.get("reason", "") if metadata else ""
        if isinstance(raw_reason, dict):
            raw_reason = raw_reason.get("reason", "")
        reason = str(raw_reason) if raw_reason is not None else ""
        history.append(
            UserTransaction(
                transaction_key=tx.transactionKey,
                transaction_time=tx.createdAt,
                transaction_type=tx.type,
                amount=tx.amount,
                running_balance=tx.runningBalance or 0,
                user_id=tx.userId,
                user_email=tx.User.email if tx.User else None,
                reason=reason,
                admin_email=admin_email,
            )
        )
    return history
