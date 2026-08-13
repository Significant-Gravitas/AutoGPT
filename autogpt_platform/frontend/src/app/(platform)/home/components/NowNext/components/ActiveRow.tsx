import Link from "next/link";
import type { HomeActiveTask } from "@/app/api/__generated__/models/homeActiveTask";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { formatRunningFor } from "../helpers";

interface Props {
  item: HomeActiveTask;
}

export function ActiveRow({ item }: Props) {
  const content = (
    <>
      <span className="absolute left-0 top-1/2 z-10 flex size-8 -translate-y-1/2 items-center justify-center rounded-full bg-white">
        {item.expert ? (
          <ExpertAvatar
            name={item.expert.name}
            avatarUrl={item.expert.avatar_url}
            size={32}
          />
        ) : (
          <span className="size-2.5 rounded-full bg-primary" />
        )}
      </span>
      <div className="min-w-0 flex-1">
        <Text variant="body-medium" className="truncate text-zinc-900">
          {item.title}
        </Text>
        <Text variant="small" className="truncate text-zinc-500">
          {item.status === "queued"
            ? "Queued"
            : (formatRunningFor(item.started_at) ?? "Running now")}
        </Text>
      </div>
    </>
  );
  const classes = "relative flex items-center gap-3 rounded-xl py-3 pl-12 pr-2";

  if (!item.link) return <div className={classes}>{content}</div>;
  return (
    <Link
      href={item.link}
      className={cn(
        classes,
        "transition-colors hover:bg-zinc-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400",
      )}
    >
      {content}
    </Link>
  );
}
