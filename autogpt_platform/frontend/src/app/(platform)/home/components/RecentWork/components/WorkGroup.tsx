import { ArrowUpRight01Icon, Robot01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import type { HomeRecentWorkGroup } from "@/app/api/__generated__/models/homeRecentWorkGroup";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { formatWorkTime, getWorkItemIcon } from "../helpers";

interface Props {
  group: HomeRecentWorkGroup;
  timezone: string;
}

export function WorkGroup({ group, timezone }: Props) {
  const header = (
    <div className="flex min-w-0 items-center gap-2">
      {group.actor.expert ? (
        <ExpertAvatar
          name={group.actor.expert.name}
          avatarUrl={group.actor.expert.avatar_url}
          size={22}
        />
      ) : (
        <span className="flex size-[22px] shrink-0 items-center justify-center rounded-full bg-zinc-100 text-zinc-500">
          <Icon icon={Robot01Icon} size={13} aria-hidden="true" />
        </span>
      )}
      <Text variant="body-medium" className="truncate text-zinc-950">
        {group.actor.name}
      </Text>
      {group.session_title ? (
        <>
          <span className="text-zinc-300" aria-hidden="true">
            ·
          </span>
          <Text variant="body" className="truncate text-zinc-500">
            {group.session_title}
          </Text>
        </>
      ) : null}
      {group.link ? (
        <Icon
          icon={ArrowUpRight01Icon}
          size={15}
          className="ml-auto shrink-0 text-zinc-300 transition-colors group-hover:text-zinc-700"
          aria-hidden="true"
        />
      ) : null}
    </div>
  );

  return (
    <article className="px-4 py-3 sm:px-5">
      {group.link ? (
        <Link
          href={group.link}
          className="group block outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-zinc-400"
        >
          {header}
        </Link>
      ) : (
        header
      )}
      <div className="mt-2 flex flex-col gap-1.5">
        {group.items.map((item) => (
          <div key={item.id} className="flex items-start gap-2.5">
            <span className="mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-md bg-zinc-100 text-zinc-500">
              <Icon
                icon={getWorkItemIcon(item.category)}
                size={13}
                aria-hidden="true"
              />
            </span>
            <div className="min-w-0 flex-1">
              <Text variant="body" className="truncate text-zinc-800">
                {item.title}
              </Text>
            </div>
            <span className="shrink-0 text-xs tabular-nums text-zinc-400">
              {item.provider ? `${item.provider} · ` : ""}
              {formatWorkTime(item.occurred_at, timezone)}
            </span>
          </div>
        ))}
        {group.more_count ? (
          <Text variant="small" className="pl-[34px] text-zinc-400">
            Plus {group.more_count} more
          </Text>
        ) : null}
      </div>
    </article>
  );
}
