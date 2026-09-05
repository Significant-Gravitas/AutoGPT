import {
  AlertDiamondIcon,
  ArrowUpRight01Icon,
  CheckListIcon,
} from "@hugeicons/core-free-icons";
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
}

const ROW_CLASS = "flex gap-2 py-2";

export function OutcomeRow({ outcome, timezone, showAgentName }: Props) {
  const failed = outcome.status === "failed";
  const meta = [
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

  const content = (
    <>
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
        <div className="mt-1 flex flex-wrap items-center gap-x-1.5 gap-y-1 text-[11px] text-zinc-400">
          {meta.map((part, index) => (
            <span key={index} className="flex items-center gap-x-1.5">
              {index > 0 ? <span aria-hidden="true">·</span> : null}
              {part}
            </span>
          ))}
        </div>
      </div>
      {outcome.link ? (
        <Icon
          icon={ArrowUpRight01Icon}
          size={14}
          className="mt-1 shrink-0 text-zinc-300 transition-colors group-hover:text-zinc-600"
          aria-hidden="true"
        />
      ) : null}
    </>
  );

  if (!outcome.link) {
    return <div className={ROW_CLASS}>{content}</div>;
  }
  return (
    <Link
      href={outcome.link}
      className={cn(
        ROW_CLASS,
        "group -mx-1 rounded px-1 outline-none transition-colors hover:bg-zinc-50 focus-visible:bg-zinc-50",
      )}
    >
      {content}
    </Link>
  );
}
