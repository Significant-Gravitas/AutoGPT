import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

import stripe
from prisma.enums import (
    CreditRefundRequestStatus,
    CreditTransactionType,
    NotificationType,
    OnboardingStep,
)
from prisma.errors import UniqueViolationError
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
from backend.util.exceptions import InsufficientBalanceError
from backend.util.feature_flag import Flag, is_feature_enabled
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
                from backend.executor.manager import (
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
                    f"You already have enough balance of ${current_balance/100}, top-up is not required when you already have at least ${ceiling_balance/100}"
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
                from backend.executor.manager import (
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
                message=f"Insufficient balance of ${current_balance/100}, where this will cost ${abs(amount)/100}",
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
    ) -> int:
        if cost == 0:
            return 0

        balance, _ = await self._add_transaction(
            user_id=user_id,
            amount=-cost,
            transaction_type=CreditTransactionType.USAGE,
            metadata=SafeJson(metadata.model_dump()),
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
                f"Invalid amount to deduct ${request.amount/100} from ${transaction.amount/100} top-up"
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
            logger.warning(f"Accepting dispute from {user_id} for ${amount/100}")
            dispute.close()
            return

        logger.warning(
            f"Adding extra info for dispute from {user_id} for ${amount/100}"
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
    """
    Get the credit model for a user, considering LaunchDarkly flags.

    Args:
        user_id (str): The user ID to check flags for.

    Returns:
        UserCreditBase: The appropriate credit model for the user
    """
    if not settings.config.enable_credit:
        return DisabledUserCredit()

    # Check LaunchDarkly flag for payment pilot users
    # Default to False (beta monthly credit behavior) to maintain current behavior
    is_payment_enabled = await is_feature_enabled(
        Flag.ENABLE_PLATFORM_PAYMENT, user_id, default=False
    )

    if is_payment_enabled:
        # Payment enabled users get UserCredit (no monthly refills, enable payments)
        return UserCredit()
    else:
        # Default behavior: users get beta monthly credits
        return BetaUserCredit(settings.config.num_user_credits_refill)


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
