import { useEffect, useRef, useState } from "react";
import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import type {
  ChatRule,
  RuleScope,
} from "@/app/(platform)/copilot/components/ApprovalQueue/helpers";
import { isHeldCall } from "./helpers";
import type { AttentionListRow, useHeldReview } from "./useHeldReview";

interface Args {
  rows: AttentionListRow[];
  held: ReturnType<typeof useHeldReview>;
}

// One dialog over every held call: the current one is kept by review id across the poll.
export function useReviewCarousel({ rows, held }: Args) {
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [finished, setFinished] = useState(false);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [openedWith, setOpenedWith] = useState<Set<string>>(new Set());
  const paneRef = useRef<HTMLDivElement>(null);
  const returnTo = useRef<HTMLElement | null>(null);

  // The list's own order, so the sidebar reads as the list does.
  const order = rows.filter((row) => isHeldCall(row.item));
  const index = order.findIndex((row) => row.item.id === currentId);
  const current = index >= 0 ? order[index] : null;
  const left = order.filter((row) => !row.receipt).length;

  // Focus the item's headline, never the button just pressed, so a second Enter decides nothing.
  useEffect(() => {
    if (!open) return;
    const frame = requestAnimationFrame(() => {
      const target =
        paneRef.current?.querySelector<HTMLElement>("h3, [data-pane-focus]") ??
        null;
      if (!target) return;
      target.tabIndex = -1;
      target.focus();
    });
    return () => cancelAnimationFrame(frame);
  }, [open, currentId, finished]);

  // The drawer below lg does not restore focus itself.
  useEffect(() => {
    if (open || !returnTo.current) return;
    const target = returnTo.current;
    returnTo.current = null;
    requestAnimationFrame(() => {
      if (target.isConnected) target.focus();
    });
  }, [open]);

  function openAt(itemID: string) {
    returnTo.current =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    setCurrentId(itemID);
    setFinished(false);
    setSheetOpen(false);
    setOpenedWith(new Set(order.map((row) => row.item.id)));
    setOpen(true);
  }

  function close() {
    setOpen(false);
  }

  function move(delta: number) {
    const next = order[index + delta];
    if (!next) return;
    setFinished(false);
    setCurrentId(next.item.id);
  }

  function jump(itemID: string) {
    setFinished(false);
    setSheetOpen(false);
    setCurrentId(itemID);
  }

  async function decide(
    item: HomeAttentionItem,
    approved: boolean,
    rule?: ChatRule,
    scope?: RuleScope,
  ) {
    const done = await held.decide([item], approved, {
      rule,
      scope,
      focusNext: false,
    });
    if (!done.includes(item.id)) return;
    const undecided = (row: AttentionListRow) =>
      !row.receipt && row.item.id !== item.id;
    const at = order.findIndex((row) => row.item.id === item.id);
    const next =
      order.slice(at + 1).find(undecided) ??
      order.slice(0, Math.max(at, 0)).find(undecided);
    if (next) setCurrentId(next.item.id);
    else setFinished(true);
  }

  return {
    open,
    order,
    current,
    position: index + 1,
    total: order.length,
    left,
    finished,
    isNew: (itemID: string) => open && !openedWith.has(itemID),
    canPrev: index > 0,
    canNext: index >= 0 && index < order.length - 1,
    paneRef,
    sheetOpen,
    setSheetOpen,
    openAt,
    close,
    move,
    jump,
    decide,
  };
}
