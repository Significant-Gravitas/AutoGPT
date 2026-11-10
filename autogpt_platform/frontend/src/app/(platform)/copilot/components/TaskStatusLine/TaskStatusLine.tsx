"use client";

import {
  Alert02Icon,
  ArrowUpRight01Icon,
  Clock01Icon,
  StopIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import NextLink from "next/link";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import {
  clock,
  duration,
  type StartedTask,
  type TaskLineState,
} from "./helpers";
import { useTaskStatusLine } from "./useTaskStatusLine";

const BADGES: Record<
  TaskLineState,
  { className: string; icon: typeof Clock01Icon | null }
> = {
  working: { className: "bg-info-surface text-info", icon: null },
  needs_input: {
    className: "bg-warning-surface text-warning-foreground",
    icon: Clock01Icon,
  },
  done: {
    className: "bg-success-surface text-success-foreground",
    icon: Tick02Icon,
  },
  failed: { className: "bg-error-surface text-error", icon: Alert02Icon },
  stopped: {
    className: "bg-surface-hover text-foreground-secondary",
    icon: StopIcon,
  },
};

const AVATAR_SIZE = 24;

interface Props {
  task: StartedTask;
}

/** One line under the steps for a task Otto started: who has it, what it is
 *  doing now and how long it has run. It stays when the steps fold away, so
 *  the user can always tell that work is going on. */
export function TaskStatusLine({ task }: Props) {
  const { state, seconds, title, owner } = useTaskStatusLine(task);
  const badge = BADGES[state];
  const time =
    seconds === null
      ? null
      : state === "working"
        ? clock(seconds)
        : duration(seconds);

  return (
    <div className="flex items-start gap-2.5">
      <span className="relative inline-flex shrink-0">
        <ExpertAvatar
          name={owner.name}
          avatarUrl={owner.avatarUrl}
          size={AVATAR_SIZE}
        />
        <span
          aria-hidden
          data-testid="task-status-badge"
          data-state={state}
          className={cn(
            "absolute -end-0.5 -bottom-0.5 flex size-3 items-center justify-center rounded-full ring-2 ring-surface",
            badge.className,
          )}
        >
          {badge.icon ? (
            <Icon icon={badge.icon} size={8} />
          ) : (
            <span
              // oxlint-disable-next-line design-system/no-arbitrary-values -- a 1.5px spinner stroke inside a small avatar badge; border-2 would thicken it
              className="size-[60%] animate-spin rounded-full border-[1.5px] border-current border-t-transparent motion-reduce:animate-none"
            />
          )}
        </span>
      </span>
      <div className="flex min-w-0 flex-1 flex-wrap items-baseline gap-x-2.5 gap-y-0.5 pt-0.5">
        <p
          role={state === "working" ? undefined : "status"}
          className="min-w-0 text-body [overflow-wrap:anywhere] text-foreground-secondary"
        >
          <Sentence state={state} name={owner.name} title={title} />
        </p>
        {time && (
          <span className="shrink-0 text-body text-foreground-tertiary tabular-nums">
            {time}
          </span>
        )}
        {task.link && state !== "needs_input" && (
          <NextLink
            href={task.link}
            className="relative inline-flex shrink-0 touch-manipulation items-center gap-1 rounded text-body font-medium text-foreground-emphasis underline-offset-2 before:absolute before:-inset-x-2 before:-inset-y-2.5 hover:underline focus-visible:ring-1 focus-visible:ring-focus focus-visible:outline-none"
          >
            See the work
            <Icon
              icon={ArrowUpRight01Icon}
              size={14}
              aria-hidden
              className="rtl:-scale-x-100"
            />
          </NextLink>
        )}
      </div>
    </div>
  );
}

interface SentenceProps {
  state: TaskLineState;
  name: string;
  title: string;
}

function Sentence({ state, name, title }: SentenceProps) {
  const who = (
    <span className="font-medium text-foreground-emphasis">{name}</span>
  );
  switch (state) {
    case "needs_input":
      return (
        <>
          {who} needs your answer to carry on with {title}.
        </>
      );
    case "done":
      return (
        <>
          {who} finished {title}.
        </>
      );
    case "failed":
      return <>{title} didn&apos;t finish.</>;
    case "stopped":
      return <>{title} was stopped.</>;
    case "working":
      return (
        <>
          {who} is working on {title}.
        </>
      );
  }
}
