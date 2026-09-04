import Link from "next/link";
import type { HomeActiveTask } from "@/app/api/__generated__/models/homeActiveTask";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { formatRunningFor } from "../helpers";

interface Props {
  item: HomeActiveTask;
}

const ROW_CLASS = "relative flex items-center gap-3 py-2 pl-[3.75rem] pr-4";

export function ActiveRow({ item }: Props) {
  const content = (
    <>
      <span className="absolute left-4 top-1/2 z-10 flex size-9 -translate-y-1/2 items-center justify-center rounded-full bg-white">
        {item.expert ? (
          <ExpertAvatar
            name={item.expert.name}
            avatarUrl={item.expert.avatar_url}
            size={36}
          />
        ) : (
          <span className="size-2.5 rounded-full bg-primary" />
        )}
      </span>
      <div className="min-w-0 flex-1">
        <Text
          variant="body-medium"
          className="truncate text-[13px] leading-5 text-zinc-900"
        >
          {item.title}
        </Text>
        <Text variant="small" className="truncate text-[11px] text-zinc-500">
          {item.status === "queued"
            ? "Queued"
            : (formatRunningFor(item.started_at) ?? "Running now")}
        </Text>
      </div>
    </>
  );

  if (!item.link) return <div className={ROW_CLASS}>{content}</div>;
  return (
    <Link
      href={item.link}
      className={cn(
        ROW_CLASS,
        "outline-none transition-colors hover:bg-zinc-50 focus-visible:bg-zinc-50",
      )}
    >
      {content}
    </Link>
  );
}
