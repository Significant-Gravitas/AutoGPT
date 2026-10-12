"use client";

import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { type ChatDelegation, formatElapsed } from "../../../../../delegations";
import {
  formatCost,
  getDelegationStatusView,
} from "../../../../../delegationViews";
import { useDelegationLive } from "../../../../../useDelegationLive";
import { DOT_CLASS, delegationLine } from "../helpers";

interface Props {
  delegation: ChatDelegation;
  onOpen: () => void;
}

export function DelegationRow({ delegation, onOpen }: Props) {
  const live = useDelegationLive(delegation);
  const view = getDelegationStatusView(live.status);
  const elapsed = formatElapsed(live.elapsedSeconds);
  const cost = formatCost(delegation.costUsd);
  const waiting = view.tone === "waiting";

  return (
    <button
      type="button"
      onClick={onOpen}
      data-testid="delegation-row"
      data-status={live.status}
      className={cn(
        "flex w-full gap-2.5 rounded-xl border px-3 py-2.5 text-left transition-colors hover:bg-zinc-50",
        waiting
          ? "border-amber-200 bg-amber-50/60"
          : "border-zinc-200 bg-white",
      )}
    >
      <ExpertAvatar
        name={live.expert.name}
        avatarUrl={live.expert.avatarUrl}
        color={live.expert.color}
        size={28}
      />
      <span className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span className="flex items-center justify-between gap-2">
          <span className="flex min-w-0 items-baseline gap-1.5">
            <span className="truncate text-sm font-medium text-zinc-900">
              {live.expert.name}
            </span>
            {live.expert.role && (
              <span className="truncate text-xs text-zinc-600">
                {live.expert.role}
              </span>
            )}
          </span>
          <span className="flex shrink-0 items-center gap-1.5 text-xs font-medium text-zinc-700">
            <span
              aria-hidden
              className={cn("size-1.5 rounded-full", DOT_CLASS[view.tone])}
            />
            {view.label}
          </span>
        </span>
        <span className="truncate text-xs text-zinc-600">
          {delegationLine(
            delegation,
            live.status,
            live.question,
            live.latestText,
          )}
        </span>
        <span className="flex items-center gap-2.5 text-xs text-zinc-500">
          {elapsed && <span>{elapsed}</span>}
          {cost && <span>{cost}</span>}
          <span className="flex-1" />
          <Icon icon={ArrowRight01Icon} size={14} />
        </span>
      </span>
    </button>
  );
}
