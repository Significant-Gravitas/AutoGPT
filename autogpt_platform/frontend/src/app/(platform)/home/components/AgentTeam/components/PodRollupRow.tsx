import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { formatWeeklySpend } from "../../../helpers";
import type { PodRollup } from "../helpers";
import { AgentRow } from "./AgentRow";

interface Props {
  rollup: PodRollup;
  expanded: boolean;
  onToggle: () => void;
}

export function PodRollupRow({ rollup, expanded, onToggle }: Props) {
  const spend = formatWeeklySpend(rollup.spendCents);
  const summary = [
    `${rollup.activeCount} active`,
    rollup.needsYouCount > 0 ? `${rollup.needsYouCount} need you` : null,
    spend,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <div>
      <button
        type="button"
        aria-expanded={expanded}
        onClick={onToggle}
        className="group -mx-2 flex w-[calc(100%+16px)] min-w-0 items-center gap-3 rounded-lg px-2 py-2.5 text-left transition-colors hover:bg-zinc-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400"
      >
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <Text variant="body-medium" className="truncate text-zinc-900">
              {rollup.name}
            </Text>
            <span className="shrink-0 rounded-md bg-zinc-100 px-1.5 py-0.5 text-xs font-medium tabular-nums text-zinc-600">
              {rollup.agents.length}
            </span>
          </div>
          <Text variant="small" className="min-w-0 truncate text-zinc-500">
            {summary}
          </Text>
        </div>
        <Icon
          icon={ArrowDown01Icon}
          size={15}
          className={cn(
            "shrink-0 text-zinc-300 transition-transform group-hover:text-zinc-600",
            expanded && "rotate-180",
          )}
          aria-hidden="true"
        />
      </button>
      {expanded ? (
        <div className="divide-y divide-zinc-100 pl-3">
          {rollup.agents.map((agent) => (
            <AgentRow key={agent.expert.id} agent={agent} />
          ))}
        </div>
      ) : null}
    </div>
  );
}
