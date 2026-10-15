import { useContext, useEffect, useRef, useState } from "react";
import { usePostV2ContinueChatWithAnsweredReviews } from "@/app/api/__generated__/endpoints/executions/executions";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useProcessReviews } from "@/hooks/useProcessReviews";
import type { HeldReceipt } from "@/app/(platform)/home/components/NeedsYou/useHeldReview";
import { HeldOutcomesContext } from "../ChatMessagesContainer/HeldOutcomesContext";
import { useHeldAnswersStore } from "./heldAnswersStore";
import type { CardStatus } from "./components/ApprovalCard/ApprovalCard";
import {
  type ApprovalItem,
  type ChatRule,
  type RuleScope,
  isHeldRead,
} from "./helpers";

export interface Receipt extends HeldReceipt {
  item: ApprovalItem;
}

interface Args {
  items: ApprovalItem[];
  // Whether the chat's turn starts now; false while other cards still wait.
  onAnswered: (turnFollows: boolean) => void;
  onContinued?: () => void;
}

export function useApprovalQueue({ items, onAnswered, onContinued }: Args) {
  const outcomes = useContext(HeldOutcomesContext);
  const { processReviews } = useProcessReviews();
  const { mutateAsync: continueChat, isPending: isContinuing } =
    usePostV2ContinueChatWithAnsweredReviews();
  const recordAnswers = useHeldAnswersStore((state) => state.record);
  const [statuses, setStatuses] = useState<Record<string, CardStatus>>({});
  const [failed, setFailed] = useState<string[]>([]);
  const [receipts, setReceipts] = useState<Receipt[]>([]);
  const [openId, setOpenId] = useState<string | null>(null);
  const [confirmRejectAll, setConfirmRejectAll] = useState(false);
  const [continued, setContinued] = useState<Set<string>>(new Set());
  // The cards waiting when the box collapsed; a new one opens it again.
  const [collapsedOver, setCollapsedOver] = useState<Set<string> | null>(null);
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
        ...gone.map((item) => ({
          item,
          outcome: "elsewhere" as const,
          text: "Answered elsewhere",
        })),
      ]);
    }
    seen.current = new Map(items.map((item) => [item.reviewId, item]));
  }, [items]);

  async function answer(
    batch: ApprovalItem[],
    approved: boolean,
    rule?: ChatRule,
    scope?: RuleScope,
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
    let turnFollows = true;
    try {
      const res = await processReviews(
        batch.map((item) => ({
          node_exec_id: item.reviewId,
          approved,
          chat_rule: approved ? (rule ?? null) : null,
          ...(approved && rule && scope ? { chat_rule_scope: scope } : {}),
        })),
        batch.map((item) => item.scope),
      );
      ok = res.status === 200 && res.data.failed_count === 0;
      turnFollows = !(res.status === 200 && res.data.awaiting_other_cards);
    } catch {
      ok = false;
    }
    if (ok) {
      recordAnswers(ids, approved);
      setReceipts((prev) => [
        ...prev,
        ...batch.map((item) => ({
          item,
          outcome: approved ? ("approved" as const) : ("rejected" as const),
          text: receiptText(item, approved),
        })),
      ]);
      setConfirmRejectAll(false);
      onAnswered(turnFollows);
    } else {
      ids.forEach((id) => answeredHere.current.delete(id));
      setFailed((prev) => [...prev, ...ids]);
    }
    setStatuses((prev) => {
      const next = { ...prev };
      ids.forEach((id) => delete next[id]);
      return next;
    });
    return ok;
  }

  // Starts the turn with what is answered; the rest stay here, still answerable.
  async function continueNow() {
    const chatSessionId = pending[0]?.scope.session_id;
    if (!chatSessionId) return;
    try {
      const res = await continueChat({ chatSessionId });
      if (res.status !== 204) throw new Error(`status ${res.status}`);
    } catch {
      toast({
        title: "Could not continue",
        description: "Try again, or send a message to continue.",
        variant: "destructive",
      });
      return;
    }
    setContinued(new Set(shown.map((r) => r.item.reviewId)));
    setCollapsedOver(new Set(pending.map((item) => item.reviewId)));
    onContinued?.();
  }

  const receiptIds = new Set(receipts.map((r) => r.item.reviewId));
  const currentIds = new Set(items.map((item) => item.reviewId));
  const pending = items.filter((item) => !receiptIds.has(item.reviewId));
  // A receipt stays until the chain row shows the answer.
  const shown = receipts.filter((r) =>
    r.item.toolCallId
      ? !outcomes.has(r.item.toolCallId)
      : currentIds.has(r.item.reviewId),
  );
  return {
    pending,
    receipts: shown,
    // Every answer given here, for the review dialog's sidebar.
    answered: receipts,
    canContinue:
      pending.length > 0 && shown.some((r) => !continued.has(r.item.reviewId)),
    isContinuing,
    continueNow,
    collapsed:
      collapsedOver !== null &&
      pending.length > 0 &&
      pending.every((item) => collapsedOver.has(item.reviewId)),
    expand: () => setCollapsedOver(null),
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
