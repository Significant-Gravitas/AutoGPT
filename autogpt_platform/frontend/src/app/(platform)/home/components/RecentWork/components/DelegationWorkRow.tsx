import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import type { HomeDelegationItem } from "@/app/api/__generated__/models/homeDelegationItem";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { getDelegationStatusBadge } from "../helpers";

interface Props {
  item: HomeDelegationItem;
}

const ROW_CLASS = "flex items-center gap-3 py-2.5";

/** A hand-off Otto made this week: Otto → the expert, what came back. */
export function DelegationWorkRow({ item }: Props) {
  const badge = getDelegationStatusBadge(item.status);
  const content = (
    <>
      <span className="flex shrink-0 items-center gap-1" aria-hidden="true">
        <AutopilotAvatar size={20} />
        <Icon icon={ArrowRight01Icon} size={12} className="text-zinc-900" />
        <ExpertAvatar
          name={item.expert?.name ?? null}
          avatarUrl={item.expert?.avatar_url ?? null}
          size={20}
        />
      </span>
      <div className="min-w-0 flex-1">
        <Text variant="body-medium" tone="primary" className="truncate">
          {item.title}
        </Text>
        <Text variant="small" tone="muted" className="truncate">
          {item.description}
        </Text>
      </div>
      <Badge variant={badge.variant} className="shrink-0">
        {badge.label}
      </Badge>
    </>
  );

  if (!item.link) return <div className={ROW_CLASS}>{content}</div>;
  return (
    <Link
      href={item.link}
      className={cn(
        ROW_CLASS,
        "-mx-1 rounded px-1 outline-none transition-colors hover:bg-zinc-50 focus-visible:bg-zinc-50",
      )}
    >
      {content}
    </Link>
  );
}
