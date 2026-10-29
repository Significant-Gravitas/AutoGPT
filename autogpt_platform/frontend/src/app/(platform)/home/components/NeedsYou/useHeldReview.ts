import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { getGetHomeDashboardQueryKey } from "@/app/api/__generated__/endpoints/home/home";
import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import type { CardStatus } from "@/app/(platform)/copilot/components/ApprovalQueue/components/ApprovalCard/ApprovalCard";
import type {
  ChatRule,
  RuleScope,
} from "@/app/(platform)/copilot/components/ApprovalQueue/helpers";
import { toApprovalItem } from "@/app/(platform)/copilot/components/ApprovalQueue/helpers";
import { useProcessReviews } from "@/hooks/useProcessReviews";
import { trackFunnel } from "@/services/experts/experts-analytics";
import { headlineButtonId, isHeldCall, receiptText } from "./helpers";

export interface HeldReceipt {
  outcome: "approved" | "rejected" | "elsewhere";
  text: string;
}

export interface AttentionListRow {
  item: HomeAttentionItem;
  receipt: HeldReceipt | null;
}

interface Args {
  items: HomeAttentionItem[];
}

const REFRESH_AFTER_MS = 400;

interface DecideOptions {
  rule?: ChatRule;
  scope?: RuleScope;
  focusNext?: boolean;
}

// Held calls decided here stay as receipts in place, so the list never shifts under the pointer.
export function useHeldReview({ items }: Args) {
  const queryClient = useQueryClient();
  const { processReviews } = useProcessReviews();
  const [statuses, setStatuses] = useState<Record<string, CardStatus>>({});
  const [failed, setFailed] = useState<string[]>([]);
  const [receipts, setReceipts] = useState<Record<string, HeldReceipt>>({});
  const [announcement, setAnnouncement] = useState("");
  const seen = useRef(new Map<string, HomeAttentionItem>());
  const answeredHere = useRef(new Set<string>());
  const left = useRef(new Set<string>());
  const refreshTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const order = useRef<string[]>([]);

  const rows = orderRows(order.current, items, seen.current, receipts);

  useEffect(() => {
    order.current = rows.map((row) => row.item.id);
  });

  // Leaving the page with a refresh pending still refreshes.
  useEffect(
    () => () => {
      if (refreshTimer.current) {
        clearTimeout(refreshTimer.current);
        refresh();
      }
    },
    [],
  );

  // A held call that leaves the feed without an answer from here was answered in its chat.
  useEffect(() => {
    const current = new Set(items.map((item) => item.id));
    // The gate deletes a decided row, so the same call asked again returns under the same id.
    const back = [...left.current].filter((id) => current.has(id));
    if (back.length > 0) {
      setReceipts((prev) => {
        const next = { ...prev };
        back.forEach((id) => delete next[id]);
        return next;
      });
      back.forEach((id) => {
        left.current.delete(id);
        answeredHere.current.delete(id);
      });
    }
    for (const id of seen.current.keys())
      if (!current.has(id)) left.current.add(id);
    const gone = [...seen.current.values()].filter(
      (item) => !current.has(item.id) && !answeredHere.current.has(item.id),
    );
    if (gone.length > 0) {
      setReceipts((prev) => ({
        ...prev,
        ...Object.fromEntries(
          gone.map((item) => [
            item.id,
            { outcome: "elsewhere", text: "Answered elsewhere" },
          ]),
        ),
      }));
    }
    for (const item of items) {
      if (isHeldCall(item)) seen.current.set(item.id, item);
    }
    for (const item of gone) answeredHere.current.add(item.id);
  }, [items]);

  // Resolves to the ids that landed; the list moves focus on, the dialog advances itself.
  async function decide(
    batch: HomeAttentionItem[],
    approved: boolean,
    { rule, scope, focusNext = true }: DecideOptions = {},
  ): Promise<string[]> {
    const ids = batch.map((item) => item.id);
    setFailed((prev) => prev.filter((id) => !ids.includes(id)));
    setStatuses((prev) => ({
      ...prev,
      ...Object.fromEntries(
        ids.map((id) => [id, approved ? "approving" : "rejecting"]),
      ),
    }));
    ids.forEach((id) => answeredHere.current.add(id));

    // The endpoint takes one chat per request, so a batch is split and sent in parallel.
    const results = await Promise.all(
      [...byChat(batch).values()].map(async (chatItems) => ({
        chatItems,
        ok: await send(chatItems, approved, rule, scope),
      })),
    );

    const done = results.filter((r) => r.ok).flatMap((r) => r.chatItems);
    const lost = results.filter((r) => !r.ok).flatMap((r) => r.chatItems);
    lost.forEach((item) => answeredHere.current.delete(item.id));
    if (lost.length > 0)
      setFailed((prev) => [...prev, ...lost.map((item) => item.id)]);
    if (done.length > 0) {
      const next = done.map((item) => ({
        id: item.id,
        receipt: {
          outcome: approved ? "approved" : "rejected",
          text: receiptText(toApprovalItem(item.review!), approved),
        } satisfies HeldReceipt,
      }));
      setReceipts((prev) => ({
        ...prev,
        ...Object.fromEntries(next.map((r) => [r.id, r.receipt])),
      }));
      setAnnouncement(
        next.length === 1
          ? `${done[0].title}: ${next[0].receipt.text}`
          : `${next.length} answered: ${next[0].receipt.text.split(" · ")[0]}`,
      );
      if (focusNext) focusNextAfter(ids);
      done.forEach((item) =>
        trackFunnel("home_attention_actioned", {
          kind: item.kind,
          action: approved ? "approve" : "decline",
        }),
      );
    }
    setStatuses((prev) => {
      const next = { ...prev };
      ids.forEach((id) => delete next[id]);
      return next;
    });
    if (done.length > 0) refreshSoon();
    return done.map((item) => item.id);
  }

  // One dashboard refetch per burst of decisions; receipts keep the list steady meanwhile.
  function refreshSoon() {
    if (refreshTimer.current) clearTimeout(refreshTimer.current);
    refreshTimer.current = setTimeout(refresh, REFRESH_AFTER_MS);
  }

  function refresh() {
    refreshTimer.current = null;
    void queryClient.invalidateQueries({
      queryKey: getGetHomeDashboardQueryKey(),
    });
  }

  async function send(
    chatItems: HomeAttentionItem[],
    approved: boolean,
    rule?: ChatRule,
    scope?: RuleScope,
  ) {
    try {
      const res = await processReviews(
        chatItems.map((item) => ({
          node_exec_id: item.review!.node_exec_id,
          approved,
          chat_rule: approved ? (rule ?? null) : null,
          ...(approved && rule && scope ? { chat_rule_scope: scope } : {}),
        })),
        chatItems.map((item) => item.review!),
      );
      return res.status === 200 && res.data.failed_count === 0;
    } catch {
      return false;
    }
  }

  function focusNextAfter(ids: string[]) {
    const current = order.current;
    const last = Math.max(...ids.map((id) => current.indexOf(id)));
    // Only a held call's row has a headline button; setup and question rows do not.
    const next = current
      .slice(last + 1)
      .find((id) => seen.current.has(id) && !ids.includes(id) && !receipts[id]);
    if (!next) return;
    requestAnimationFrame(() =>
      document.getElementById(headlineButtonId(next))?.focus(),
    );
  }

  return {
    rows,
    pendingCount: items.filter((item) => !receipts[item.id]).length,
    statusOf: (itemID: string): CardStatus => statuses[itemID] ?? "idle",
    hasFailed: (itemID: string) => failed.includes(itemID),
    decide,
    announcement,
  };
}

// The order the list was last drawn in, with anything new below it.
function orderRows(
  previous: string[],
  items: HomeAttentionItem[],
  seen: Map<string, HomeAttentionItem>,
  receipts: Record<string, HeldReceipt>,
): AttentionListRow[] {
  const byId = new Map(items.map((item) => [item.id, item]));
  const kept = previous.flatMap((id) => {
    // A held call that left the feed is about to be, or already is, a receipt.
    const item = byId.get(id) ?? seen.get(id);
    return item ? [item] : [];
  });
  const keptIds = new Set(kept.map((item) => item.id));
  return [...kept, ...items.filter((item) => !keptIds.has(item.id))].map(
    (item) => ({ item, receipt: receipts[item.id] ?? null }),
  );
}

function byChat(batch: HomeAttentionItem[]) {
  const chats = new Map<string, HomeAttentionItem[]>();
  for (const item of batch) {
    const key = item.review?.session_id ?? item.review?.graph_exec_id ?? "";
    chats.set(key, [...(chats.get(key) ?? []), item]);
  }
  return chats;
}
