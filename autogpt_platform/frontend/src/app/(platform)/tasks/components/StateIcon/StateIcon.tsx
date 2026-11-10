import type { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  Cancel01Icon,
  Loading03Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";

interface Props {
  status: TaskStatus;
  className?: string;
}

/** The mark before a task's title: a turning ring while it works, a dot
 *  while it waits, a check, a warning or a cross once it settles. It sits
 *  beside the status word, so it is decoration only. */
export function StateIcon({ status, className }: Props) {
  return (
    <span
      aria-hidden
      data-testid="task-state-icon"
      data-status={status}
      className={cn(
        "inline-flex size-4 shrink-0 items-center justify-center",
        className,
      )}
    >
      <Glyph status={status} />
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
