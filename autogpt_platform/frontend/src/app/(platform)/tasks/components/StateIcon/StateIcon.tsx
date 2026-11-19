import type { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  Cancel01Icon,
  Clock01Icon,
  Loading03Icon,
  PauseIcon,
  RepeatIcon,
  Tick02Icon,
  FlashIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import type { Tone } from "@/components/styles/tones";
import type { ScheduleState } from "../../schedule";

const SCHEDULE_ICONS: Record<ScheduleState, IconSvgElement> = {
  scheduled: Clock01Icon,
  missed: Clock01Icon,
  repeating: RepeatIcon,
  watching: FlashIcon,
  paused: PauseIcon,
};

const SCHEDULE_TONES: Record<ScheduleState, Tone> = {
  scheduled: "info",
  missed: "waiting",
  repeating: "info",
  watching: "info",
  paused: "quiet",
};

interface Props {
  status: TaskStatus;
  /** A to-do waiting for its start time, or one that missed it. */
  schedule?: ScheduleState | null;
  className?: string;
}

/** The mark before a task's title: a turning ring while it works, a dot
 *  while it waits, a clock while it waits for its start time, a check, a
 *  warning or a cross once it settles. It sits beside the status word, so
 *  it is decoration only. */
export function StateIcon({ status, schedule, className }: Props) {
  return (
    <span
      aria-hidden
      data-testid="task-state-icon"
      data-status={status}
      data-schedule={schedule ?? undefined}
      className={cn(
        "inline-flex size-4 shrink-0 items-center justify-center",
        className,
      )}
    >
      {schedule ? (
        <Icon
          icon={SCHEDULE_ICONS[schedule]}
          size={14}
          tone={SCHEDULE_TONES[schedule]}
        />
      ) : (
        <Glyph status={status} />
      )}
    </span>
  );
}

function Glyph({ status }: { status: TaskStatus }) {
  switch (status) {
    case "IN_PROGRESS":
      return <Icon icon={Loading03Icon} size={14} tone="info" spin />;
    case "DONE":
      return <Icon icon={Tick02Icon} size={14} tone="ok" />;
    case "FAILED":
      return <Icon icon={Alert02Icon} size={14} tone="error" />;
    case "CANCELLED":
      return <Icon icon={Cancel01Icon} size={14} tone="faint" />;
    case "NEEDS_INPUT":
      return <span className="size-2 rounded-full bg-status-waiting" />;
    default:
      return (
        <span className="size-2 rounded-full border border-foreground-faint" />
      );
  }
}
