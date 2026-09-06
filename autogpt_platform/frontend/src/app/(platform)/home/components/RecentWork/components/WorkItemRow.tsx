import Link from "next/link";
import type { HomeRecentWorkItem } from "@/app/api/__generated__/models/homeRecentWorkItem";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { formatWorkTime, getWorkItemIcon } from "../helpers";

interface Props {
  item: HomeRecentWorkItem;
  timezone: string;
}

const ROW_CLASS = "flex items-center gap-2";

export function WorkItemRow({ item, timezone }: Props) {
  const content = (
    <>
      <span className="flex size-[18px] shrink-0 items-center justify-center text-zinc-400">
        <Icon
          icon={getWorkItemIcon(item.category)}
          size={13}
          aria-hidden="true"
        />
      </span>
      <Text
        variant="small"
        className="min-w-0 flex-1 truncate text-[13px] leading-5 text-zinc-700"
      >
        {item.title}
      </Text>
      <span className="shrink-0 text-[11px] tabular-nums text-zinc-400">
        {item.provider ? `${item.provider} · ` : ""}
        {formatWorkTime(item.occurred_at, timezone)}
      </span>
    </>
  );

  if (!item.link) {
    return <div className={ROW_CLASS}>{content}</div>;
  }
  return (
    <Link
      href={item.link}
      title={item.session_title ?? undefined}
      className={cn(
        ROW_CLASS,
        "-mx-1 rounded px-1 outline-none transition-colors hover:bg-zinc-50 focus-visible:bg-zinc-50",
      )}
    >
      {content}
    </Link>
  );
}
