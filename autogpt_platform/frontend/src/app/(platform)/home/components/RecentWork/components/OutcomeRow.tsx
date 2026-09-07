import { AlertDiamondIcon, CheckListIcon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import type { HomeBriefingOutcome } from "@/app/api/__generated__/models/homeBriefingOutcome";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { formatWorkTime, getRunTriggerLabel } from "../helpers";

interface Props {
  outcome: HomeBriefingOutcome;
  timezone: string;
  /** An expert runs workflows, so the run names which one; a workflow
   *  group already carries the name in its header. */
  showAgentName: boolean;
}

const ROW_CLASS = "flex items-center gap-2 py-2";
const LINK_CLASS =
  "-mx-1 rounded px-1 outline-none transition-colors hover:bg-zinc-50 focus-visible:bg-zinc-50";

/** One line per run: what it did, then how and when it ran. The AI summary
 *  only shows on hover; the row itself opens the run. */
export function OutcomeRow({ outcome, timezone, showAgentName }: Props) {
  const failed = outcome.status === "failed";
  const content = (
    <>
      <span
        className={cn(
          "flex size-[18px] shrink-0 items-center justify-center rounded-md",
          failed ? "bg-rose-50 text-rose-600" : "bg-zinc-100 text-zinc-500",
        )}
      >
        <Icon
          icon={failed ? AlertDiamondIcon : CheckListIcon}
          size={11}
          aria-hidden="true"
        />
      </span>
      <Text
        variant="body"
        className="min-w-0 flex-1 truncate text-[13px] leading-5 text-zinc-700"
      >
        {outcome.title}
      </Text>
      <RunMeta
        outcome={outcome}
        timezone={timezone}
        showAgentName={showAgentName}
      />
    </>
  );

  if (!outcome.link) {
    return (
      <div className={ROW_CLASS} title={outcome.summary}>
        {content}
      </div>
    );
  }
  return (
    <Link
      href={outcome.link}
      title={outcome.summary}
      className={cn(ROW_CLASS, LINK_CLASS)}
    >
      {content}
    </Link>
  );
}

function RunMeta({ outcome, timezone, showAgentName }: Props) {
  const parts = [
    outcome.status === "failed" ? (
      <span key="status" className="font-medium text-rose-600">
        Failed
      </span>
    ) : null,
    showAgentName ? (
      <span key="agent" className="font-medium text-zinc-500">
        {outcome.agent_name}
      </span>
    ) : null,
    <span key="trigger">{getRunTriggerLabel(outcome.trigger)}</span>,
    <span key="time" className="tabular-nums">
      {formatWorkTime(outcome.occurred_at, timezone)}
    </span>,
  ].filter(Boolean);

  return (
    <span className="flex shrink-0 items-center gap-x-1.5 text-[11px] text-zinc-400">
      {parts.map((part, index) => (
        <span key={index} className="flex items-center gap-x-1.5">
          {index > 0 ? <span aria-hidden="true">·</span> : null}
          {part}
        </span>
      ))}
    </span>
  );
}
