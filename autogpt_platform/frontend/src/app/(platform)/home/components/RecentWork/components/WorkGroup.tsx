import { ArrowUpRight01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import type { HomeRecentWorkGroup } from "@/app/api/__generated__/models/homeRecentWorkGroup";
import type { HomeWorkActor } from "@/app/api/__generated__/models/homeWorkActor";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { getActorIcon, getActorKindLabel } from "../helpers";
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
        className="truncate text-[13px] leading-5 text-zinc-900"
      >
        {actor.name}
      </Text>
      <span className="shrink-0 rounded-full border border-zinc-200 px-1.5 text-[10px] font-medium uppercase leading-4 tracking-[0.04em] text-zinc-500">
        {getActorKindLabel(actor.kind)}
      </span>
      {actor.link ? (
        <Icon
          icon={ArrowUpRight01Icon}
          size={14}
          className="ml-auto shrink-0 text-zinc-300 transition-colors group-hover:text-zinc-600"
          aria-hidden="true"
        />
      ) : null}
    </div>
  );

  return (
    <article className="px-4 py-3" aria-label={actor.name}>
      {actor.link ? (
        <Link
          href={actor.link}
          className="group -mx-1 block rounded px-1 outline-none focus-visible:bg-zinc-50"
        >
          {header}
        </Link>
      ) : (
        header
      )}
      {runs.length > 0 ? (
        <div className="mt-1 divide-y divide-zinc-100">
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
        <div className="mt-1.5 flex flex-col gap-1">
          {items.map((item) => (
            <WorkItemRow key={item.id} item={item} timezone={timezone} />
          ))}
        </div>
      ) : null}
      {group.more_count ? (
        <Text
          variant="small"
          className="mt-1.5 pl-[26px] text-[11px] text-zinc-400"
        >
          Plus {group.more_count} more
        </Text>
      ) : null}
    </article>
  );
}

function ActorMark({ actor }: { actor: HomeWorkActor }) {
  if (actor.kind === "expert" && actor.expert) {
    return (
      <ExpertAvatar
        name={actor.expert.name}
        avatarUrl={actor.expert.avatar_url}
        size={18}
      />
    );
  }
  return (
    <span className="flex size-[18px] shrink-0 items-center justify-center rounded-full bg-zinc-100 text-zinc-500">
      <Icon icon={getActorIcon(actor.kind)} size={11} aria-hidden="true" />
    </span>
  );
}
