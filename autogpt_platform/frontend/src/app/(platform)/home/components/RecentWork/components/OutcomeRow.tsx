import { AlertDiamondIcon, CheckListIcon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import type { HomeBriefingOutcome } from "@/app/api/__generated__/models/homeBriefingOutcome";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { formatCurrency, formatDuration } from "../../../helpers";
import { formatWorkTime } from "../helpers";

interface Props {
  outcome: HomeBriefingOutcome;
  timezone: string;
  /** An expert runs workflows, so the run names which one; a workflow
   *  group already carries the name in its header. */
  showAgentName: boolean;
  /** Title and time only: for the runs after the first in a group. */
  compact?: boolean;
}

const ROW_CLASS = "flex gap-2 py-2";
const LINK_CLASS =
  "-mx-1 rounded px-1 outline-none transition-colors hover:bg-zinc-50 focus-visible:bg-zinc-50";

export function OutcomeRow({
  outcome,
  timezone,
  showAgentName,
  compact = false,
}: Props) {
  const failed = outcome.status === "failed";
  const mark = (
    <span
      className={cn(
        "mt-0.5 flex size-[18px] shrink-0 items-center justify-center rounded-md",
        failed ? "bg-rose-50 text-rose-600" : "bg-zinc-100 text-zinc-500",
      )}
    >
      <Icon
        icon={failed ? AlertDiamondIcon : CheckListIcon}
        size={11}
        aria-hidden="true"
      />
    </span>
  );

  const content = compact ? (
    <>
      {mark}
      <Text
        variant="body"
        className="min-w-0 flex-1 truncate text-[13px] leading-5 text-zinc-700"
      >
        {outcome.title}
      </Text>
      <span className="shrink-0 text-[11px] tabular-nums text-zinc-400">
        {formatWorkTime(outcome.occurred_at, timezone)}
      </span>
    </>
  ) : (
    <>
      {mark}
      <div className="min-w-0 flex-1">
        <Text
          variant="body-medium"
          className="text-pretty text-[13px] leading-5 text-zinc-900"
        >
          {outcome.title}
        </Text>
        <Text
          variant="small"
          className="mt-0.5 line-clamp-2 text-pretty text-[13px] leading-5 text-zinc-500"
        >
          {outcome.summary}
        </Text>
        <RunMeta
          outcome={outcome}
          timezone={timezone}
          showAgentName={showAgentName}
        />
      </div>
    </>
  );

  if (!outcome.link) {
    return (
      <div className={cn(ROW_CLASS, compact && "items-center")}>{content}</div>
    );
  }
  return (
    <Link
      href={outcome.link}
      className={cn(ROW_CLASS, LINK_CLASS, compact && "items-center")}
    >
      {content}
    </Link>
  );
}

function RunMeta({ outcome, timezone, showAgentName }: Omit<Props, "compact">) {
  const failed = outcome.status === "failed";
  const parts = [
    failed ? (
      <span key="status" className="font-medium text-rose-600">
        Failed
      </span>
    ) : null,
    showAgentName ? (
      <span key="agent" className="font-medium text-zinc-500">
        {outcome.agent_name}
      </span>
    ) : null,
    <span key="time" className="tabular-nums">
      {formatWorkTime(outcome.occurred_at, timezone)}
    </span>,
    outcome.duration_seconds ? (
      <span key="duration" className="tabular-nums">
        {formatDuration(outcome.duration_seconds)}
      </span>
    ) : null,
    outcome.cost_cents ? (
      <span key="cost" className="tabular-nums">
        {formatCurrency(outcome.cost_cents)}
      </span>
    ) : null,
  ].filter(Boolean);

  return (
    <div className="mt-1 flex flex-wrap items-center gap-x-1.5 gap-y-1 text-[11px] text-zinc-400">
      {parts.map((part, index) => (
        <span key={index} className="flex items-center gap-x-1.5">
          {index > 0 ? <span aria-hidden="true">·</span> : null}
          {part}
        </span>
      ))}
    </div>
  );
}
