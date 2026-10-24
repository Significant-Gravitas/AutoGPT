"use client";

import {
  Alert02Icon,
  ArrowUpRight01Icon,
  Task01Icon,
} from "@hugeicons/core-free-icons";
import NextLink from "next/link";
import { useId, useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  type HandoffOutcome,
  type HandoffTask,
  possessive,
} from "./expertHandoff";

// A brief longer than this is clamped to two lines with a Show all toggle.
const LONG_BRIEF = 160;

interface TaskProps {
  task: HandoffTask;
}

/** The task the user already described, shown on the hire card so they
 *  can see exactly what the expert will be handed. */
export function HandoffTaskBlock({ task }: TaskProps) {
  const [expanded, setExpanded] = useState(false);
  const labelId = useId();
  const briefId = useId();
  const long = task.brief.length > LONG_BRIEF;
  return (
    <section
      aria-labelledby={labelId}
      className="ml-[42px] mt-3 rounded-2xl bg-zinc-50 px-3 py-2.5"
    >
      <p
        id={labelId}
        className="flex items-center gap-1.5 text-xs font-medium text-zinc-500"
      >
        <Icon icon={Task01Icon} size={14} aria-hidden />
        First task
      </p>
      <p className="mt-1 text-sm font-medium text-zinc-800 [overflow-wrap:anywhere]">
        {task.title}
      </p>
      <p
        id={briefId}
        className={
          long && !expanded
            ? "mt-0.5 line-clamp-2 text-sm text-zinc-500 [overflow-wrap:anywhere]"
            : "mt-0.5 whitespace-pre-line text-sm text-zinc-500 [overflow-wrap:anywhere]"
        }
      >
        {task.brief}
      </p>
      {long && (
        <button
          type="button"
          aria-expanded={expanded}
          aria-controls={briefId}
          onClick={() => setExpanded(!expanded)}
          className="mt-1 rounded text-xs font-medium text-zinc-600 underline-offset-2 hover:underline focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-zinc-800"
        >
          {expanded ? "Show less" : "Show all"}
        </button>
      )}
    </section>
  );
}

interface ActionsProps {
  name: string;
  task: HandoffTask;
  onHandOff: () => void;
  onHireOnly: () => void;
  onDecline: () => void;
}

/** Hire and hand off is the answer the card expects, so it is the one
 *  filled button and names both the expert and the task. When the card is
 *  too narrow for one row it wraps onto its own row above the two quieter
 *  choices, keeping the reading and tab order Decline, Hire only, hire and
 *  hand off. */
export function HandoffActions({
  name,
  task,
  onHandOff,
  onHireOnly,
  onDecline,
}: ActionsProps) {
  return (
    <div className="ms-auto flex max-w-full flex-wrap-reverse items-center justify-end gap-2">
      <div className="flex items-center gap-1.5">
        <Button
          variant="ghost"
          size="small"
          onClick={onDecline}
          className="h-8 !min-w-0 px-3"
        >
          Decline
        </Button>
        <Button
          variant="secondary"
          size="small"
          onClick={onHireOnly}
          className="h-8 !min-w-0 px-3"
        >
          Hire only
        </Button>
      </div>
      <Button
        variant="primary"
        size="small"
        onClick={onHandOff}
        className="h-auto min-h-8 !min-w-0 max-w-full whitespace-normal !rounded-2xl px-3.5 py-1.5 text-center [overflow-wrap:anywhere]"
      >
        Hire {name} and hand off {task.title}
      </Button>
    </div>
  );
}

interface StatusProps {
  outcome: HandoffOutcome;
  expertName: string;
}

/** Where the work went once the hire and hand-off landed. */
export function HandoffStatusRow({ outcome, expertName }: StatusProps) {
  if (outcome.status === "failed") {
    return (
      <div
        role="status"
        className="flex items-start gap-2 border-t border-zinc-100 px-4 py-3 text-sm text-zinc-600"
      >
        <Icon
          icon={Alert02Icon}
          size={16}
          className="mt-0.5 shrink-0 text-amber-600"
          aria-hidden
        />
        <p className="min-w-0 [overflow-wrap:anywhere]">
          {expertName} is hired, but {outcome.title} didn&apos;t start.
        </p>
      </div>
    );
  }
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-t border-zinc-100 px-4 py-3">
      <p className="min-w-[min(100%,14rem)] flex-1 text-sm text-zinc-600 [overflow-wrap:anywhere]">
        <HandoffSentence outcome={outcome} expertName={expertName} />
      </p>
      {outcome.link && (
        <NextLink
          href={outcome.link}
          className="inline-flex shrink-0 items-center gap-1.5 rounded-full border border-zinc-200 bg-white px-3 py-1.5 text-sm font-medium text-zinc-800 shadow-sm transition-colors hover:border-zinc-300 hover:bg-zinc-50 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-zinc-800"
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

function HandoffSentence({ outcome, expertName }: StatusProps) {
  const name = <span className="font-medium text-zinc-800">{expertName}</span>;
  if (outcome.status === "unconfirmed") {
    return (
      <>
        {outcome.title} was sent to {possessive(expertName)} chat.
      </>
    );
  }
  if (outcome.status === "queued") {
    return (
      <>
        {name} will start on {outcome.title} as soon as a task slot frees. The
        work happens in their own chat.
      </>
    );
  }
  return (
    <>
      {name} started on {outcome.title}. The work happens in their own chat.
    </>
  );
}
