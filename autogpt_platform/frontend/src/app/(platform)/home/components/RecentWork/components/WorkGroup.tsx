import { ArrowUpRight01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import type { HomeRecentWorkGroup } from "@/app/api/__generated__/models/homeRecentWorkGroup";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { formatGroupCounts, getActorKindLabel } from "../helpers";
import { ActorMark } from "./ActorMark";
import { OutcomeRow } from "./OutcomeRow";
import { WorkItemRow } from "./WorkItemRow";

interface Props {
  group: HomeRecentWorkGroup;
  timezone: string;
}

export function WorkGroup({ group, timezone }: Props) {
  const { actor } = group;
  const runs = group.runs ?? [];
  const items = group.items ?? [];
  const header = (
    <div className="flex min-w-0 items-center gap-2">
      <ActorMark actor={actor} />
      <Text
        variant="body-medium"
        className="truncate text-sm font-semibold leading-5 text-zinc-900"
      >
        {actor.name}
      </Text>
      <span className="shrink-0 rounded-full border border-zinc-200 bg-white px-1.5 text-[10px] font-medium uppercase leading-4 tracking-[0.04em] text-zinc-500">
        {getActorKindLabel(actor.kind)}
      </span>
      <span className="ml-auto shrink-0 text-[11px] tabular-nums text-zinc-400">
        {formatGroupCounts(group)}
      </span>
      {actor.link ? (
        <Icon
          icon={ArrowUpRight01Icon}
          size={14}
          className="shrink-0 text-zinc-300 transition-colors group-hover:text-zinc-600"
          aria-hidden="true"
        />
      ) : null}
    </div>
  );

  return (
    <article aria-label={actor.name}>
      <div className="bg-zinc-50/80 px-4 py-2">
        {actor.link ? (
          <Link
            href={actor.link}
            className="group block rounded outline-none focus-visible:ring-2 focus-visible:ring-zinc-300"
          >
            {header}
          </Link>
        ) : (
          header
        )}
      </div>
      {runs.length > 0 ? (
        <div className="divide-y divide-zinc-100 px-4">
          {/* One run tells the story; the rest are a line each, unless they
              failed and need a look. */}
          {runs.map((run, index) => (
            <OutcomeRow
              key={run.id}
              outcome={run}
              timezone={timezone}
              showAgentName={actor.kind === "expert"}
              compact={index > 0 && run.status !== "failed"}
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
