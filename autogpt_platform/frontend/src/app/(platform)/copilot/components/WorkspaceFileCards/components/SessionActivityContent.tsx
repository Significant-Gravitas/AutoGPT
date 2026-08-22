"use client";

import {
  CancelCircleIcon,
  CheckmarkCircle02Icon,
  ClockIcon,
  Loading03Icon,
  PlayIcon,
  RepeatIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import Link from "next/link";
import { useGetV2GetLibraryAgent } from "@/app/api/__generated__/endpoints/library/library";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { formatWhen } from "../../ToolChain/resultHelpers";
import type { SessionRun, SessionSchedule } from "../helpers";

const RUN_STATUS: Record<
  string,
  { icon: IconSvgElement; className: string; spin: boolean }
> = {
  COMPLETED: {
    icon: CheckmarkCircle02Icon,
    className: "text-green-600",
    spin: false,
  },
  FAILED: { icon: CancelCircleIcon, className: "text-red-500", spin: false },
  RUNNING: { icon: Loading03Icon, className: "text-purple-600", spin: true },
  QUEUED: { icon: ClockIcon, className: "text-amber-600", spin: false },
};

const FALLBACK_STATUS = {
  icon: PlayIcon,
  className: "text-zinc-400",
  spin: false,
};

const LIST =
  "-mx-2.5 grid max-h-72 gap-0.5 overflow-y-auto px-2.5 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200";

/** Runs this chat triggered. Chrome-free (the section title lives outside the
 *  card) so the floating stack and the popover can each frame it. */
export function RunsList({ runs }: { runs: SessionRun[] }) {
  return (
    <div className={LIST}>
      {runs.map((run) => (
        <RunRow key={run.executionId} run={run} />
      ))}
    </div>
  );
}

/** Schedules this chat created. */
export function SchedulesList({ schedules }: { schedules: SessionSchedule[] }) {
  return (
    <div className={LIST}>
      {schedules.map((schedule) => (
        <ScheduleRow key={schedule.scheduleId} schedule={schedule} />
      ))}
    </div>
  );
}

function RunRow({ run }: { run: SessionRun }) {
  // An in-flight run started by library-agent id has no name in its tool
  // input, so the row fetches it rather than showing a placeholder.
  const { data: fetchedName } = useGetV2GetLibraryAgent(
    run.libraryAgentId ?? "",
    {
      query: {
        enabled: !run.name && !!run.libraryAgentId,
        select: (res) => (res.data as LibraryAgent).name,
      },
    },
  );
  const name = run.name ?? fetchedName ?? "Agent run";
  const meta = run.startedAt ? formatWhen(run.startedAt) : null;
  // The leading icon carries the status on its own — a tick, a cross, or a
  // spinner reads faster than the word did, so the label is gone.
  const status = RUN_STATUS[run.status?.toUpperCase() ?? ""] ?? FALLBACK_STATUS;
  const content = (
    <>
      <Icon
        icon={status.icon}
        size={16}
        aria-label={run.status?.toLowerCase()}
        className={cn(
          "shrink-0",
          status.className,
          status.spin && "animate-spin motion-reduce:animate-none",
        )}
      />
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm text-zinc-800">{name}</p>
        {meta && <p className="truncate text-xs text-zinc-400">{meta}</p>}
      </div>
    </>
  );
  // Same bleed as WorkspaceFileCard: the negative margin cancels the list's
  // padding so row content lands on the card's content edge, with the hover
  // background spilling into the gutter.
  const rowClass =
    "-mx-2.5 flex items-center gap-3 rounded-xl px-2.5 py-1.5 transition-colors hover:bg-zinc-50";

  if (run.href) {
    return (
      <Link href={run.href} title={name} className={rowClass}>
        {content}
      </Link>
    );
  }
  return (
    <div title={name} className={rowClass}>
      {content}
    </div>
  );
}

function ScheduleRow({ schedule }: { schedule: SessionSchedule }) {
  const when = schedule.nextRunTime
    ? `Next ${formatWhen(schedule.nextRunTime)}`
    : null;
  return (
    <div className="-mx-2.5 flex items-start gap-3 rounded-xl px-2.5 py-1.5 transition-colors hover:bg-zinc-50">
      <Icon
        icon={schedule.isRecurring ? RepeatIcon : ClockIcon}
        size={16}
        className="mt-0.5 shrink-0 text-zinc-700"
      />
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm text-zinc-800">{schedule.name}</p>
        {schedule.detail && (
          <p className="line-clamp-2 text-xs text-zinc-500">
            {schedule.detail}
          </p>
        )}
        {(when || schedule.cron) && (
          <p className="flex min-w-0 items-center gap-1.5 text-xs text-zinc-400">
            {when && <span className="truncate">{when}</span>}
            {when && schedule.cron && <span aria-hidden>·</span>}
            {schedule.cron && (
              <span className="shrink-0 font-mono">{schedule.cron}</span>
            )}
            {schedule.timezone && (
              <span className="shrink-0">{schedule.timezone}</span>
            )}
          </p>
        )}
      </div>
    </div>
  );
}
