import { useContext, useEffect, useRef, useState } from "react";
import { useProcessReviews } from "@/hooks/useProcessReviews";
import { HeldOutcomesContext } from "../ChatMessagesContainer/HeldOutcomesContext";
import type { CardStatus } from "./components/ApprovalCard/ApprovalCard";
import { type ApprovalItem, type ChatRule, isHeldRead } from "./helpers";

export interface Receipt {
  item: ApprovalItem;
  text: string;
}

interface Args {
  items: ApprovalItem[];
  onAnswered: () => void;
}

export function useApprovalQueue({ items, onAnswered }: Args) {
  const outcomes = useContext(HeldOutcomesContext);
  const { processReviews } = useProcessReviews();
  const [statuses, setStatuses] = useState<Record<string, CardStatus>>({});
  const [failed, setFailed] = useState<string[]>([]);
  const [receipts, setReceipts] = useState<Receipt[]>([]);
  const [openId, setOpenId] = useState<string | null>(null);
  const [confirmRejectAll, setConfirmRejectAll] = useState(false);
  const seen = useRef(new Map<string, ApprovalItem>());
  const answeredHere = useRef(new Set<string>());

  // A card that leaves the poll without an answer from here was answered on Home or another tab.
  useEffect(() => {
    const current = new Set(items.map((item) => item.reviewId));
    const gone = [...seen.current.values()].filter(
      (item) =>
        !current.has(item.reviewId) && !answeredHere.current.has(item.reviewId),
    );
    if (gone.length > 0) {
      setReceipts((prev) => [
        ...prev,
        ...gone.map((item) => ({ item, text: "Answered elsewhere" })),
      ]);
    }
    seen.current = new Map(items.map((item) => [item.reviewId, item]));
  }, [items]);

  async function answer(
    batch: ApprovalItem[],
    approved: boolean,
    rule?: ChatRule,
  ) {
    const ids = batch.map((item) => item.reviewId);
    setFailed((prev) => prev.filter((id) => !ids.includes(id)));
    setStatuses((prev) => ({
      ...prev,
      ...Object.fromEntries(
        ids.map((id) => [id, approved ? "approving" : "rejecting"]),
      ),
    }));
    ids.forEach((id) => answeredHere.current.add(id));
    let ok = false;
    try {
      const res = await processReviews(
        batch.map((item) => ({
          node_exec_id: item.reviewId,
          approved,
          chat_rule: approved ? (rule ?? null) : null,
        })),
        batch.map((item) => item.scope),
      );
      ok = res.status === 200 && res.data.failed_count === 0;
    } catch {
      ok = false;
    }
    if (ok) {
      setReceipts((prev) => [
        ...prev,
        ...batch.map((item) => ({
          item,
          text: receiptText(item, approved),
        })),
      ]);
      setConfirmRejectAll(false);
      onAnswered();
    } else {
      ids.forEach((id) => answeredHere.current.delete(id));
      setFailed((prev) => [...prev, ...ids]);
    }
    setStatuses((prev) => {
      const next = { ...prev };
      ids.forEach((id) => delete next[id]);
      return next;
    });
  }

  const receiptIds = new Set(receipts.map((r) => r.item.reviewId));
  const currentIds = new Set(items.map((item) => item.reviewId));
  return {
    pending: items.filter((item) => !receiptIds.has(item.reviewId)),
    // A receipt stays until the chain row shows the answer.
    receipts: receipts.filter((r) =>
      r.item.toolCallId
        ? !outcomes.has(r.item.toolCallId)
        : currentIds.has(r.item.reviewId),
    ),
    statusOf: (id: string) => statuses[id] ?? "idle",
    hasFailed: (id: string) => failed.includes(id),
    openId,
    setOpenId,
    confirmRejectAll,
    setConfirmRejectAll,
    answer,
  };
}

function receiptText(item: ApprovalItem, approved: boolean) {
  if (isHeldRead(item)) return approved ? "Released" : "Kept out";
  return approved ? "Approved" : "Rejected";
}
