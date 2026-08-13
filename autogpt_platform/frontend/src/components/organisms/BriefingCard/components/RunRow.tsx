import Link from "next/link";
import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import type { BriefingRunItem } from "@/app/api/__generated__/models/briefingRunItem";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { getSafeLink } from "../helpers";

interface Props {
  item: BriefingRunItem;
}

// One row per run: the summary doubles as the row's subtitle, so what ran and
// what it found read as a single line instead of two mirrored lists.
export function RunRow({ item }: Props) {
  const isFailed = item.status !== "COMPLETED";
  const subtitle = item.summary ?? item.expert_name;
  const link = getSafeLink(item.link);

  const body = (
    <>
      <ExpertAvatar
        name={item.expert_name}
        avatarUrl={item.expert_avatar_url}
        size={28}
      />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <Text variant="body-medium" className="truncate text-zinc-900">
            {item.agent_name}
          </Text>
          {isFailed ? (
            <span className="shrink-0 rounded-full bg-red-50 px-2 py-0.5 text-[0.6875rem] font-medium text-red-600">
              Failed
            </span>
          ) : null}
        </div>
        {subtitle ? (
          <Text variant="body" className="line-clamp-2 text-zinc-500">
            {/* Attribution matters once more than one agent reports:
                mirrors the thread markdown's "**{agent}**: {summary}". */}
            {item.summary && item.expert_name ? (
              <span className="text-zinc-400">{item.expert_name} · </span>
            ) : null}
            {subtitle}
          </Text>
        ) : null}
      </div>
      {link ? (
        <Icon
          icon={ArrowRight01Icon}
          size={16}
          className="shrink-0 self-center text-zinc-300 transition-colors group-hover:text-zinc-500"
        />
      ) : null}
    </>
  );

  const rowClassName = "group flex items-start gap-3 px-5 py-4";

  return (
    <li>
      {link ? (
        <Link
          href={link}
          className={cn(rowClassName, "transition-colors hover:bg-zinc-50")}
        >
          {body}
        </Link>
      ) : (
        <div className={rowClassName}>{body}</div>
      )}
    </li>
  );
}
