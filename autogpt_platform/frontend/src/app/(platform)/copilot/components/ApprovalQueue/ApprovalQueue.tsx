"use client";

import { cn } from "@/lib/utils";
import { ApprovalCard } from "./components/ApprovalCard/ApprovalCard";
import { CompactApprovalLine } from "./components/CompactApprovalLine";
import { QueueHeader } from "./components/QueueHeader";
import { ReceiptLine } from "./components/ReceiptLine";
import { RejectAllFooter } from "./components/RejectAllFooter";
import {
  type ApprovalItem,
  approvalCardId,
  canApproveAll,
  isBare,
  modeLine,
} from "./helpers";
import { useApprovalQueue } from "./useApprovalQueue";
import { useStuckToBottom } from "./useStuckToBottom";

interface Props {
  // Oldest first, as the gate raised them.
  items: ApprovalItem[];
  onAnswered: () => void;
}

// From this many cards up, each collapses to one line.
export const COMPACT_FROM = 4;

export function ApprovalQueue({ items, onAnswered }: Props) {
  const queue = useApprovalQueue({ items, onAnswered });
  const { sentinelRef, stuck, expand } = useStuckToBottom();
  const { pending, receipts } = queue;
  if (pending.length === 0 && receipts.length === 0) return null;

  const compact = pending.length >= COMPACT_FROM;
  const mode = pending.find((item) => item.mode)?.mode ?? null;
  const showModeLine = pending.some((item) => item.reasonKind === "mode");
  const approvable = canApproveAll(pending, compact) ? pending : null;
  const anyBusy = pending.some(
    (item) => queue.statusOf(item.reviewId) !== "idle",
  );

  return (
    <>
      <div ref={sentinelRef} aria-hidden="true" className="h-px" />
      <section
        aria-label="Waiting for you"
        className={cn(
          "overflow-hidden rounded-xl border border-zinc-200 bg-white",
          stuck && "sticky bottom-2 z-10 shadow-md",
        )}
      >
        <QueueHeader
          count={pending.length}
          mode={mode}
          collapsed={stuck}
          onExpand={expand}
          approveAll={
            approvable && !stuck
              ? {
                  count: approvable.length,
                  busy: anyBusy,
                  onClick: () => queue.answer(approvable, true),
                }
              : null
          }
        />
        {!stuck && (
          <>
            {showModeLine && pending.length > 0 && (
              <p className="border-b border-zinc-100 px-4 py-2 text-sm text-zinc-500">
                {modeLine(mode)}
              </p>
            )}
            <ol className="divide-y divide-zinc-100">
              {pending.map((item) => (
                <li
                  key={item.reviewId}
                  id={approvalCardId(item.reviewId)}
                  tabIndex={-1}
                  className="scroll-mb-24 outline-none focus-visible:bg-zinc-50"
                >
                  {compact && queue.openId !== item.reviewId ? (
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
                      onApprove={(rule) => queue.answer([item], true, rule)}
                      onReject={() => queue.answer([item], false)}
                    />
                  )}
                </li>
              ))}
            </ol>
            <ul aria-live="polite" className="divide-y divide-zinc-100">
              {receipts.map((receipt) => (
                <ReceiptLine key={receipt.item.reviewId} receipt={receipt} />
              ))}
            </ul>
            {pending.length >= 2 && (
              <RejectAllFooter
                count={pending.length}
                confirming={queue.confirmRejectAll}
                busy={anyBusy}
                onAsk={() => queue.setConfirmRejectAll(true)}
                onCancel={() => queue.setConfirmRejectAll(false)}
                onConfirm={() => queue.answer(pending, false)}
              />
            )}
          </>
        )}
      </section>
    </>
  );
}
