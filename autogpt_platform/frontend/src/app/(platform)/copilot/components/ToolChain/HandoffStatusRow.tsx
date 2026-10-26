"use client";

import { Alert02Icon, ArrowUpRight01Icon } from "@hugeicons/core-free-icons";
import NextLink from "next/link";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  frozenStatusOf,
  type HandoffOutcome,
  possessive,
} from "./expertHandoff";
import { useSubSessionEffectiveStatus } from "./SubSessionLive";

interface Props {
  outcome: HandoffOutcome;
  expertName: string;
}

/** Where the work went once the hire and hand-off landed. The confirm only
 *  queued the task, so the row follows the expert's thread from there, the
 *  way a delegation card does. */
export function HandoffStatusRow({ outcome, expertName }: Props) {
  const live = useSubSessionEffectiveStatus(
    outcome.subSessionId,
    frozenStatusOf(outcome),
  );
  if (outcome.status === "failed") {
    return (
      <div
        role="status"
        className="flex items-start gap-2 border-t border-separator px-4 py-3 text-sm text-foreground-secondary"
      >
        <Icon
          icon={Alert02Icon}
          size={16}
          className="mt-0.5 shrink-0 text-warning-foreground"
          aria-hidden
        />
        <p className="min-w-0 [overflow-wrap:anywhere]">
          {expertName} is hired, but {outcome.title} didn&apos;t start.
        </p>
      </div>
    );
  }
  const stopped = live === "completed";
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-t border-separator px-4 py-3">
      <p className="min-w-[min(100%,14rem)] flex-1 text-sm text-foreground-secondary [overflow-wrap:anywhere]">
        <HandoffSentence
          outcome={outcome}
          expertName={expertName}
          stopped={stopped}
        />
      </p>
      {live && !stopped && <LiveBadge status={live} />}
      {outcome.link && (
        <NextLink
          href={outcome.link}
          className="inline-flex shrink-0 items-center gap-1.5 rounded-full border border-border-base bg-surface px-3 py-1.5 text-sm font-medium text-foreground-emphasis shadow-sm transition-colors hover:border-border-strong hover:bg-surface-muted focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-focus"
        >
          Open {possessive(expertName)} chat
          <Icon
            icon={ArrowUpRight01Icon}
            size={14}
            aria-hidden
            className="rtl:-scale-x-100"
          />
        </NextLink>
      )}
    </div>
  );
}

// The status pill of the delegation cards still uses raw palette colours.
const BADGES: Record<string, { label: string; className: string }> = {
  running: { label: "Working", className: "bg-info-surface text-info" },
  queued: {
    label: "Waiting",
    className: "bg-warning-surface text-warning-foreground",
  },
};

interface BadgeProps {
  status: string;
}

/** What the expert's thread is doing now; "Status unknown" when the poll
 *  can't say, so a stale "Working" never reads as fact. */
function LiveBadge({ status }: BadgeProps) {
  const badge = BADGES[status];
  return (
    <span
      className={
        badge
          ? `inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-caption-sm font-medium leading-inherit ${badge.className}`
          : "inline-flex shrink-0 items-center rounded-full bg-surface-hover px-2 py-0.5 text-caption-sm font-medium leading-inherit text-foreground-tertiary"
      }
    >
      {badge && (
        <span
          aria-hidden
          className="size-2.5 animate-spin rounded-full border border-current border-t-transparent motion-reduce:animate-none"
        />
      )}
      {badge?.label ?? "Status unknown"}
    </span>
  );
}

interface SentenceProps extends Props {
  stopped: boolean;
}

function HandoffSentence({ outcome, expertName, stopped }: SentenceProps) {
  const name = (
    <span className="font-medium text-foreground-emphasis">{expertName}</span>
  );
  if (outcome.status === "unconfirmed") {
    return (
      <>
        {outcome.title} may have started in {possessive(expertName)} chat. Open
        it to check.
      </>
    );
  }
  // An idle thread holds a result, a question for the user or an error; the
  // poll can't tell which, so the row only says the work has paused there.
  if (stopped) {
    return (
      <>
        {name} stopped working on {outcome.title}. Open their chat for the
        result, or anything they need from you.
      </>
    );
  }
  if (outcome.status === "queued") {
    return (
      <>
        {name} will start on {outcome.title} as soon as one of your other tasks
        finishes. The work happens in their own chat.
      </>
    );
  }
  return (
    <>
      Sent {outcome.title} to {name}. The work happens in their own chat.
    </>
  );
}
