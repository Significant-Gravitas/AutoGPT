import { useRef, useState } from "react";
import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import type { HomeExpert } from "@/app/api/__generated__/models/homeExpert";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import type { AttentionListRow } from "@/app/(platform)/home/components/NeedsYou/useHeldReview";
import { useReviewCarousel } from "@/app/(platform)/home/components/NeedsYou/useReviewCarousel";
import { type ChatRule, type RuleScope, toApprovalItem } from "./helpers";
import type { Receipt, useApprovalQueue } from "./useApprovalQueue";

interface Args {
  queue: ReturnType<typeof useApprovalQueue>;
  // The chat's waiting cards, oldest first.
  reviews: PendingHumanReviewModel[];
  expert: HomeExpert | null;
}

// Home's review dialog over this chat's queue, so the box and the dialog share one set of answers.
export function useChatReview({ queue, reviews, expert }: Args) {
  // Every card this chat has shown, so an answered one keeps its row.
  const [seen, setSeen] = useState<
    ReadonlyMap<string, PendingHumanReviewModel>
  >(() => new Map());
  const known = rememberReviews(seen, reviews);
  if (known !== seen) setSeen(known);
  const answeredHere = useRef(0);

  const waiting = new Set(reviews.map((review) => review.node_exec_id));
  const receipts = new Map(queue.answered.map((r) => [r.item.reviewId, r]));
  const rows: AttentionListRow[] = [...known.values()]
    .filter((r) => waiting.has(r.node_exec_id) || receipts.has(r.node_exec_id))
    .map((review) => ({
      item: toAttentionItem(review, expert),
      receipt: heldReceipt(receipts.get(review.node_exec_id)),
    }));

  const held = {
    statusOf: queue.statusOf,
    hasFailed: queue.hasFailed,
    async decide(
      batch: HomeAttentionItem[],
      approved: boolean,
      { rule, scope }: { rule?: ChatRule; scope?: RuleScope } = {},
    ) {
      const items = batch.map((item) => toApprovalItem(item.review!));
      if (!(await queue.answer(items, approved, rule, scope))) return [];
      answeredHere.current += batch.length;
      return batch.map((item) => item.id);
    },
  };
  const carousel = useReviewCarousel({ rows, held });

  function open() {
    const first = rows.find((row) => !row.receipt);
    if (!first) return;
    answeredHere.current = 0;
    queue.setReviewing(true);
    carousel.openAt(first.item.id);
  }

  // Closing part-way is Close and continue; the last answer already started the turn.
  function close() {
    carousel.close();
    queue.setReviewing(false);
    if (answeredHere.current > 0 && queue.pending.length > 0)
      void queue.continueNow();
  }

  return { held, carousel: { ...carousel, close }, open };
}

function rememberReviews(
  seen: ReadonlyMap<string, PendingHumanReviewModel>,
  reviews: PendingHumanReviewModel[],
) {
  if (reviews.every((review) => seen.get(review.node_exec_id) === review))
    return seen;
  const next = new Map(seen);
  for (const review of reviews) next.set(review.node_exec_id, review);
  return next;
}

function toAttentionItem(
  review: PendingHumanReviewModel,
  expert: HomeExpert | null,
): HomeAttentionItem {
  const { headline } = toApprovalItem(review);
  return {
    id: review.node_exec_id,
    kind: "approval",
    priority: "normal",
    title: headline.ask,
    headline,
    description: "",
    why_it_matters: "",
    expert,
    created_at: review.created_at,
    review,
    primary_action: { label: "", href: "" },
  };
}

function heldReceipt(receipt: Receipt | undefined) {
  return receipt ? { outcome: receipt.outcome, text: receipt.text } : null;
}
