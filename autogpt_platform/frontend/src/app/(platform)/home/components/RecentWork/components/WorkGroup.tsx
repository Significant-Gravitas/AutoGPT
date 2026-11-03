import { ArrowUpRight01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import type { HomeRecentWorkGroup } from "@/app/api/__generated__/models/homeRecentWorkGroup";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { formatGroupCounts, getActorChip } from "../helpers";
import { ActorMark } from "./ActorMark";
import { OutcomeRow } from "./OutcomeRow";
import { WorkItemRow } from "./WorkItemRow";

interface Props {
  group: HomeRecentWorkGroup;
  timezone: string;
}

export function WorkGroup({ group, timezone }: Props) {
  const { actor } = group;
  const chip = getActorChip(actor.kind);
  const runs = group.runs ?? [];
  const items = group.items ?? [];
  const header = (
    <div className="flex min-w-0 items-center gap-2">
      <ActorMark actor={actor} />
      <Text variant="body-medium-tight" tone="primary" className="truncate">
        {actor.name}
      </Text>
      <span
        className={cn(
          "shrink-0 rounded-full border px-1.5",
          // oxlint-disable-next-line design-system/require-static-classes -- ACTOR_CHIPS holds whole, static class strings
          chip.className,
        )}
      >
        <Text
          variant="small-medium-tight"
          as="span"
          tone="inherit"
          transform="capitalize"
          className="block"
        >
          {chip.label}
        </Text>
      </span>
      <Text
        variant="small"
        tone="muted"
        tabular
        className="ml-auto shrink-0"
        as="span"
      >
        {formatGroupCounts(group)}
      </Text>
      {actor.link ? (
        <Icon
          icon={ArrowUpRight01Icon}
          size={14}
          tone="disabled"
          motion="colors"
          className="shrink-0 group-hover:text-foreground-secondary"
          aria-hidden="true"
        />
      ) : null}
    </div>
  );

  return (
    <article aria-label={actor.name}>
      <div className="bg-surface-hover/80 px-4 py-2">
        {actor.link ? (
          <Link
            href={actor.link}
            className="group block rounded outline-none focus-visible:ring-2 focus-visible:ring-border-strong"
          >
            {header}
          </Link>
        ) : (
          header
        )}
      </div>
      {runs.length > 0 ? (
        <div className="divide-y divide-separator px-4">
          {runs.map((run) => (
            <OutcomeRow
              key={run.id}
              outcome={run}
              timezone={timezone}
              showAgentName={actor.kind === "expert"}
            />
          ))}
        </div>
      ) : null}
      {items.length > 0 ? (
        <div className="flex flex-col gap-1 px-4 py-2">
          {items.map((item) => (
            <WorkItemRow key={item.id} item={item} timezone={timezone} />
          ))}
        </div>
      ) : null}
    </article>
  );
}
