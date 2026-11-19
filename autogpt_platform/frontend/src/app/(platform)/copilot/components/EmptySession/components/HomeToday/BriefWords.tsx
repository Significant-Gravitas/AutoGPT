import { formatUntil } from "@/app/(platform)/home/components/NowNext/helpers";
import { ownerOf } from "@/app/(platform)/tasks/helpers";
import type { ReactNode } from "react";
import { dayPart } from "./helpers";
import type { useHomeToday } from "./useHomeToday";
import type { useWeekStats } from "./useWeekStats";
import { span } from "./weekStats";

interface Props {
  today: ReturnType<typeof useHomeToday>;
  stats: ReturnType<typeof useWeekStats>;
  /** Follows a paragraph that already greeted the reader. */
  isFollowUp?: boolean;
}

/** The day as Otto would say it: what waits on you, what is ready, what is
 *  still moving, what runs next and how the week went. Each task it names
 *  opens that task. */
export function BriefWords({ today, stats, isFollowUp }: Props) {
  const [need] = today.needs;
  const [ready] = today.ready;
  const [next] = today.upcoming;
  const finished = stats.doneWeek;

  return (
    <p className="text-body leading-6 text-pretty text-foreground-text">
      {isFollowUp ? null : (
        <>
          <span className="font-medium text-foreground-ink">
            {dayPart(today.now).replace("This", "Good")}.
          </span>{" "}
        </>
      )}
      {need ? (
        <>
          {today.needs.length === 1
            ? "One task is"
            : `${today.needs.length} tasks are`}{" "}
          waiting on you, starting with{" "}
          <Named onClick={() => today.openTask(need)}>{need.title}</Named>.{" "}
        </>
      ) : (
        "Nothing is waiting on you. "
      )}
      {ready ? (
        <>
          {ownerOf(ready, today.experts).name} finished{" "}
          <Named onClick={() => today.openTask(ready)}>{ready.title}</Named>
          {today.ready.length > 1
            ? `, and ${today.ready.length - 1} more are ready to read. `
            : ". "}
        </>
      ) : null}
      {today.working.length > 0
        ? `${today.working.length === 1 ? "One is" : `${today.working.length} are`} still in progress. `
        : null}
      {next
        ? `Next up is ${next.title}, ${formatUntil(next.next_run_time, new Date(today.now))}. `
        : null}
      {finished > 0
        ? `This week your team finished ${finished} ${finished === 1 ? "task" : "tasks"}, ${span(stats.workSeconds)} of work.`
        : null}
    </p>
  );
}

interface NamedProps {
  onClick: () => void;
  children: ReactNode;
}

function Named({ onClick, children }: NamedProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="rounded-sm text-left font-medium text-foreground-ink underline decoration-foreground-ink/25 underline-offset-4 transition-colors duration-100 outline-none hover:decoration-foreground-ink focus-visible:ring-2 focus-visible:ring-focus-inbox"
    >
      {children}
    </button>
  );
}
