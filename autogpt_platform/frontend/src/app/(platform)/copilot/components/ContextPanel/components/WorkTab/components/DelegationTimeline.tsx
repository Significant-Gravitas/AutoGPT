"use client";

import {
  CheckmarkCircle02Icon,
  FlashIcon,
  BulbIcon,
  MessageQuestionIcon,
  Message01Icon,
  Alert02Icon,
  File02Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { ShimmerText } from "../../../../ToolChain/ShimmerText";
import {
  formatClock,
  type TimelineEntry,
  type TimelineKind,
} from "../timeline";

interface Props {
  entries: TimelineEntry[];
}

const KIND_ICON: Record<TimelineKind, IconSvgElement> = {
  Approved: CheckmarkCircle02Icon,
  Action: FlashIcon,
  Thought: BulbIcon,
  Question: MessageQuestionIcon,
  "You answered": Message01Icon,
  Response: File02Icon,
  Error: Alert02Icon,
};

const MAX_ENTRIES = 12;

/** The teammate's run step by step: what kind of step, when, and what. */
export function DelegationTimeline({ entries }: Props) {
  const recent = entries.slice(-MAX_ENTRIES);
  if (recent.length === 0)
    return <p className="text-sm text-zinc-500">Nothing to show yet.</p>;
  return (
    <ol className="flex flex-col gap-3" data-testid="delegation-timeline">
      {recent.map((entry) => (
        <li key={entry.key} className="flex items-start gap-2.5">
          <span
            className={cn(
              "flex size-7 shrink-0 items-center justify-center rounded-full border",
              entry.kind === "Error"
                ? "border-red-200 text-red-500"
                : "border-zinc-200 text-zinc-600",
            )}
          >
            <Icon icon={KIND_ICON[entry.kind]} size={14} />
          </span>
          <span className="flex min-w-0 flex-col gap-0.5">
            <span className="flex items-baseline gap-2 text-xs">
              <span className="font-medium text-zinc-900">{entry.kind}</span>
              {formatClock(entry.at) && (
                <span className="text-zinc-500">{formatClock(entry.at)}</span>
              )}
            </span>
            {entry.live ? (
              <ShimmerText text={entry.text} className="text-sm" />
            ) : (
              <span className="line-clamp-3 text-sm text-zinc-700">
                {entry.text}
              </span>
            )}
          </span>
        </li>
      ))}
    </ol>
  );
}
