import Link from "next/link";
import type { KeyboardEvent } from "react";
import { useEffect, useRef } from "react";
import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import {
  ApprovalCard,
  type CardStatus,
} from "@/app/(platform)/copilot/components/ApprovalQueue/components/ApprovalCard/ApprovalCard";
import {
  type ApprovalItem,
  type ChatRule,
  type RuleScope,
  modeLine,
} from "@/app/(platform)/copilot/components/ApprovalQueue/helpers";
import { isKey } from "@/lib/keyboard";
import { HeldAvatar } from "./HeldAvatar";

interface Props {
  item: HomeAttentionItem;
  approval: ApprovalItem;
  status: CardStatus;
  failed: boolean;
  avatarSize: number | null;
  onClose: () => void;
  onDecide: (approved: boolean, rule?: ChatRule, scope?: RuleScope) => void;
}

// The chat's own approval card, opened in place of the row.
export function HeldCallDetail({
  item,
  approval,
  status,
  failed,
  avatarSize,
  onClose,
  onDecide,
}: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    ref.current?.focus();
  }, []);

  function handleKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    // The rule menu portals out of this subtree; its Esc is the menu's.
    if (!isKey(e, "Escape") || !ref.current?.contains(e.target as Node)) return;
    e.stopPropagation();
    onClose();
  }

  return (
    <div
      ref={ref}
      tabIndex={-1}
      role="group"
      aria-label={item.title}
      onKeyDown={handleKeyDown}
      className="flex border-l-2 border-zinc-300 bg-zinc-50/40 outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-zinc-300"
    >
      {avatarSize ? (
        <div className="hidden shrink-0 pl-4 pt-3 sm:block">
          <HeldAvatar item={item} size={avatarSize} />
        </div>
      ) : null}
      <div className="min-w-0 flex-1">
        <ApprovalCard
          item={approval}
          status={status}
          failed={failed}
          expertName={item.expert?.name ?? null}
          note={approval.reasonKind === "mode" ? modeLine(approval.mode) : null}
          onApprove={(rule, scope) => onDecide(true, rule, scope)}
          onReject={() => onDecide(false)}
          aside={
            <Link
              href={item.primary_action.href}
              className="rounded-sm text-sm font-medium text-zinc-800 underline underline-offset-4 hover:text-zinc-950 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-300"
            >
              Open chat
            </Link>
          }
        />
      </div>
    </div>
  );
}
