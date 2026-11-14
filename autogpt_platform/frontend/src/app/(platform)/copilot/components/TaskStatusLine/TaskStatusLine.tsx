"use client";

import {
  Alert02Icon,
  ArrowUpRight01Icon,
  StopIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { OverflowText } from "@/components/atoms/OverflowText/OverflowText";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { useReducedMotion } from "framer-motion";
import { Eclipse } from "loading-dev";
import NextLink from "next/link";
import { useWorkAreaStore } from "../../workArea";
import { possessive } from "../ToolChain/expertHandoff";
import {
  clock,
  duration,
  type StartedTask,
  type TaskLineState,
} from "./helpers";
import { useTaskStatusLine } from "./useTaskStatusLine";

const RING_SIZE = 18;
const RING_RADIUS = 7;
const RING_LENGTH = 2 * Math.PI * RING_RADIUS;

const SETTLED_MARK = {
  done: { icon: Tick02Icon, className: "bg-status-ok text-surface" },
  failed: { icon: Alert02Icon, className: "bg-error text-surface" },
  stopped: { icon: StopIcon, className: "bg-foreground-faint text-surface" },
} as const;

interface Props {
  task: StartedTask;
}

/** The box a started task leaves in the chat: a ring for how far it is, its
 *  title, who has it and where it is in words, a way into the chat doing
 *  the work, and Open, which brings the task up beside this chat. It stays
 *  when the steps fold away, so work running in the background is always in
 *  view. */
export function TaskStatusLine({ task }: Props) {
  const line = useTaskStatusLine(task);
  const openTask = useWorkAreaStore((s) => s.openTask);
  const { state, seconds, title, owner } = line;
  const time =
    seconds === null
      ? null
      : state === "working"
        ? clock(seconds)
        : duration(seconds);

  return (
    <div className="flex min-h-[3.25rem] items-center gap-3 rounded-card bg-surface py-2 pr-2 pl-3 shadow-(--inbox-shadow-card) transition-shadow duration-150 ease-out hover:shadow-(--inbox-shadow-card-hover)">
      <TaskRing state={state} subtasks={line.subtasks} />
      <div className="flex min-w-0 flex-1 flex-col">
        <OverflowText
          value={title}
          overflow="fade"
          variant="inbox-medium"
          tone="ink"
        />
        <span className="flex min-w-0 items-center gap-1.5">
          <span aria-hidden className="inline-flex shrink-0">
            <ExpertAvatar
              name={owner.name}
              avatarUrl={owner.avatarUrl}
              size={16}
              shape="soft"
            />
          </span>
          <Text as="span" variant="inbox" tone="text" className="shrink-0">
            {owner.name}
          </Text>
          <Text as="span" variant="inbox" tone="faint" aria-hidden>
            ·
          </Text>
          <OverflowText
            value={whereItIs(state, line.statusText)}
            overflow="fade"
            variant="inbox"
            tone="quiet"
            className="min-w-0 flex-1"
            role={state === "working" ? undefined : "status"}
          />
        </span>
      </div>
      {line.subtasks ? (
        <Text
          as="span"
          variant="inbox"
          tone="faint"
          tabular
          className="shrink-0"
        >
          {line.subtasks.done}/{line.subtasks.total}
        </Text>
      ) : null}
      {state === "needs_input" ? (
        <span className="shrink-0 rounded-sm bg-status-waiting/15 px-1.5 text-caption font-medium text-warning-foreground">
          Needs you{line.openAsks > 1 ? ` · ${line.openAsks}` : ""}
        </span>
      ) : null}
      {time ? (
        <Text
          as="span"
          variant="inbox"
          tone="faint"
          tabular
          className="shrink-0"
        >
          {time}
        </Text>
      ) : null}
      {task.link ? (
        <NextLink
          href={task.link}
          aria-label={`Open ${possessive(owner.name)} chat`}
          title={`Open ${possessive(owner.name)} chat`}
          className="flex h-7 shrink-0 items-center gap-1 rounded-control px-2 text-body-sm font-medium text-foreground-quiet transition-colors duration-100 ease-out outline-none hover:bg-surface-tray hover:text-foreground-ink focus-visible:ring-2 focus-visible:ring-focus-inbox"
        >
          Chat
          <Icon icon={ArrowUpRight01Icon} size={13} tone="inherit" />
        </NextLink>
      ) : null}
      <button
        type="button"
        onClick={() => openTask(task.id, title)}
        className="flex h-7 shrink-0 items-center rounded-control bg-surface px-2.5 text-body-sm font-medium text-foreground-ink shadow-(--inbox-shadow-card) transition-transform duration-150 ease-out outline-none hover:shadow-(--inbox-shadow-card-hover) focus-visible:ring-2 focus-visible:ring-focus-inbox active:scale-95 motion-reduce:transition-none"
      >
        Open
      </button>
    </div>
  );
}

/** Where the task is, after its owner's name. */
function whereItIs(state: TaskLineState, statusText: string | null) {
  switch (state) {
    case "needs_input":
      return "Needs your answer";
    case "done":
      return "Done";
    case "failed":
      return "Didn't finish";
    case "stopped":
      return "Stopped";
    case "working":
      return statusText ?? "Working on it";
  }
}

interface RingProps {
  state: TaskLineState;
  subtasks: { done: number; total: number } | null;
}

// How far the task is: a ring that fills with its subtasks, an eclipse spinner
// while a task without subtasks is worked on, and a mark once it settles.
function TaskRing({ state, subtasks }: RingProps) {
  const prefersReducedMotion = useReducedMotion();
  if (state === "done" || state === "failed" || state === "stopped") {
    const mark = SETTLED_MARK[state];
    return (
      <span
        aria-hidden
        data-testid="task-status-badge"
        data-state={state}
        className={cn(
          "flex size-[1.125rem] shrink-0 items-center justify-center rounded-full",
          mark.className,
        )}
      >
        <Icon icon={mark.icon} size={11} tone="inherit" />
      </span>
    );
  }
  // Worked on with no subtasks to count: the eclipse spinner turns.
  if (state === "working" && !subtasks) {
    return (
      <span
        aria-hidden
        data-testid="task-status-badge"
        data-state={state}
        className="flex shrink-0 text-foreground-text"
      >
        <Eclipse
          size={RING_SIZE}
          playState={prefersReducedMotion ? "paused" : "running"}
        />
      </span>
    );
  }
  const share = subtasks ? subtasks.done / Math.max(1, subtasks.total) : 0;
  return (
    <svg
      aria-hidden
      data-testid="task-status-badge"
      data-state={state}
      width={RING_SIZE}
      height={RING_SIZE}
      viewBox={`0 0 ${RING_SIZE} ${RING_SIZE}`}
      className={cn(
        "shrink-0",
        state === "needs_input"
          ? "text-status-waiting"
          : "text-foreground-text",
      )}
    >
      <circle
        cx={RING_SIZE / 2}
        cy={RING_SIZE / 2}
        r={RING_RADIUS}
        fill="none"
        stroke="currentColor"
        strokeOpacity={0.18}
        strokeWidth={2}
      />
      <circle
        cx={RING_SIZE / 2}
        cy={RING_SIZE / 2}
        r={RING_RADIUS}
        fill="none"
        stroke="currentColor"
        strokeWidth={2}
        strokeLinecap="round"
        strokeDasharray={RING_LENGTH}
        strokeDashoffset={RING_LENGTH * (1 - share)}
        // The arc starts at the top.
        transform={`rotate(-90 ${RING_SIZE / 2} ${RING_SIZE / 2})`}
        className="transition-[stroke-dashoffset] duration-300 ease-out motion-reduce:transition-none"
      />
    </svg>
  );
}
