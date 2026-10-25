"use client";

import type { HomeExpert } from "@/app/api/__generated__/models/homeExpert";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { HeldReviewDialog } from "@/app/(platform)/home/components/NeedsYou/components/HeldReviewDialog";
import { cn } from "@/lib/utils";
import { ApprovalCard } from "./components/ApprovalCard/ApprovalCard";
import { CompactApprovalLine } from "./components/CompactApprovalLine";
import { QueueFooter } from "./components/QueueFooter";
import { QueueHeader } from "./components/QueueHeader";
import { ReceiptLine } from "./components/ReceiptLine";
import {
  type ApprovalItem,
  approvalCardId,
  canApproveAll,
  isBare,
  modeLine,
} from "./helpers";
import { useApprovalQueue } from "./useApprovalQueue";
import { useChatReview } from "./useChatReview";

interface Props {
  // Oldest first, as the gate raised them.
  items: ApprovalItem[];
  // The same cards as rows, which the review dialog reads; none, no dialog.
  reviews?: PendingHumanReviewModel[];
  // Cards answered before this mount whose calls have not run yet.
  answered?: PendingHumanReviewModel[];
  // The chat's Expert, or null in a chat with Otto.
  expertName?: string | null;
  expert?: HomeExpert | null;
  onAnswered: (turnFollows: boolean) => void;
  onContinued?: () => void;
}

// From this many cards up, each collapses to one line.
export const COMPACT_FROM = 4;

export function ApprovalQueue({
  items,
  reviews = [],
  answered,
  expertName = null,
  expert = null,
  onAnswered,
  onContinued,
}: Props) {
  const queue = useApprovalQueue({ items, answered, onAnswered, onContinued });
  const review = useChatReview({ queue, reviews, expert });
  const { pending, receipts, leaving } = queue;
  const dialogOpen = review.carousel.open;
  if (pending.length === 0 && receipts.length === 0 && !dialogOpen) return null;

  const compact = pending.length >= COMPACT_FROM;
  const mode = pending.find((item) => item.mode)?.mode ?? null;
  const showModeLine = pending.some((item) => item.reasonKind === "mode");
  const approvable = canApproveAll(pending, compact) ? pending : null;
  const anyBusy = pending.some(
    (item) => queue.statusOf(item.reviewId) !== "idle",
  );

  const dialog = (
    <HeldReviewDialog
      carousel={review.carousel}
      held={review.held}
      chatLink={false}
    />
  );

  if (queue.collapsed)
    return (
      <section
        aria-label="Waiting for you"
        className="overflow-hidden rounded-xl border border-border-base bg-surface"
      >
        <QueueHeader
          count={pending.length}
          mode={mode}
          approveAll={null}
          onReviewAll={null}
          onShow={queue.expand}
        />
        {dialog}
      </section>
    );

  return (
    <section
      aria-label="Waiting for you"
      className={cn(
        "overflow-hidden rounded-xl border border-border-base bg-surface transition-opacity duration-200 motion-reduce:transition-none",
        leaving && "opacity-0",
      )}
    >
      <QueueHeader
        count={pending.length}
        mode={mode}
        approveAll={
          approvable
            ? {
                count: approvable.length,
                busy: anyBusy,
                onClick: () => queue.answer(approvable, true),
              }
            : null
        }
        onReviewAll={
          reviews.length > 0 && pending.length >= 2 ? review.open : null
        }
        onShow={null}
      />
      {showModeLine && pending.length > 0 && (
        <p className="border-b border-separator px-4 py-2 text-sm text-foreground-tertiary">
          {modeLine(mode)}
        </p>
      )}
      <ol className="divide-y divide-separator">
        {pending.map((item) => (
          <li
            key={item.reviewId}
            id={approvalCardId(item.reviewId)}
            tabIndex={-1}
            className="scroll-mb-24 outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-border-strong"
          >
            {compact &&
            queue.openId !== item.reviewId &&
            !queue.hasFailed(item.reviewId) ? (
              <CompactApprovalLine
                item={item}
                bare={isBare(item)}
                status={queue.statusOf(item.reviewId)}
                onOpen={() => queue.setOpenId(item.reviewId)}
                onApprove={() => queue.answer([item], true)}
                onReject={() => queue.answer([item], false)}
              />
            ) : (
              <ApprovalCard
                item={item}
                status={queue.statusOf(item.reviewId)}
                failed={queue.hasFailed(item.reviewId)}
                expertName={expertName}
                onApprove={(rule, scope) =>
                  queue.answer([item], true, rule, scope)
                }
                onReject={() => queue.answer([item], false)}
              />
            )}
          </li>
        ))}
      </ol>
      <ul aria-live="polite" className="divide-y divide-separator">
        {receipts.map((receipt) => (
          <ReceiptLine key={receipt.item.reviewId} receipt={receipt} />
        ))}
      </ul>
      {(pending.length >= 2 || queue.canContinue) && (
        <QueueFooter
          rejectAll={
            pending.length >= 2
              ? {
                  count: pending.length,
                  confirming: queue.confirmRejectAll,
                  onAsk: () => queue.setConfirmRejectAll(true),
                  onCancel: () => queue.setConfirmRejectAll(false),
                  onConfirm: () => queue.answer(pending, false),
                }
              : null
          }
          onContinue={queue.canContinue ? queue.continueNow : null}
          busy={anyBusy || queue.isContinuing}
        />
      )}
      {dialog}
    </section>
  );
}
