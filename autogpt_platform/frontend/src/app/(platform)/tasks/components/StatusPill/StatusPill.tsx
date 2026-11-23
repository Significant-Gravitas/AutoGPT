import type { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import type { Tone } from "@/components/styles/tones";
import { statusOf } from "../../helpers";
import { QUEUED_WORD } from "../../queue";
import { SCHEDULE_WORDS, type ScheduleState } from "../../schedule";

const PILL_FILL: Record<TaskStatus, string> = {
  TODO: "bg-surface-wash",
  IN_PROGRESS: "bg-info-surface",
  NEEDS_INPUT: "bg-status-waiting/10",
  FAILED: "bg-error-surface",
  DONE: "bg-status-ok-surface",
  CANCELLED: "bg-surface-wash",
};

const SCHEDULE_VIEWS: Record<
  ScheduleState,
  { word: string; tone: Tone; fill: string }
> = {
  scheduled: {
    word: SCHEDULE_WORDS.scheduled,
    tone: "info",
    fill: "bg-info-surface",
  },
  missed: {
    word: SCHEDULE_WORDS.missed,
    tone: "waiting",
    fill: "bg-status-waiting/10",
  },
  repeating: {
    word: SCHEDULE_WORDS.repeating,
    tone: "info",
    fill: "bg-info-surface",
  },
  watching: {
    word: SCHEDULE_WORDS.watching,
    tone: "info",
    fill: "bg-info-surface",
  },
  paused: {
    word: SCHEDULE_WORDS.paused,
    tone: "quiet",
    fill: "bg-surface-wash",
  },
};

interface Props {
  status: TaskStatus;
  /** A to-do waiting for its start time, or one that missed it. */
  schedule?: ScheduleState | null;
  /** Started, but waiting for a free slot (`isQueued`). */
  queued?: boolean;
  className?: string;
}

const QUEUED_VIEW = {
  word: QUEUED_WORD,
  tone: "quiet",
  fill: "bg-surface-wash",
} as const;

/** A task's one status word on a soft tint of its colour. */
export function StatusPill({ status, schedule, queued, className }: Props) {
  const timed = queued
    ? QUEUED_VIEW
    : schedule
      ? SCHEDULE_VIEWS[schedule]
      : null;
  const view = timed ?? statusOf(status);
  return (
    <span
      data-testid="task-status"
      data-status={status}
      data-schedule={schedule ?? undefined}
      data-queued={queued || undefined}
      className={cn(
        "inline-flex h-5 shrink-0 items-center rounded-control px-1.5",
        timed ? timed.fill : PILL_FILL[status],
        className,
      )}
    >
      <Text
        as="span"
        variant="inbox-medium"
        tone={view.tone}
        className="whitespace-nowrap"
      >
        {view.word}
      </Text>
    </span>
  );
}
