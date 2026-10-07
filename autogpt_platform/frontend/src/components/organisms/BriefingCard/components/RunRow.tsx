import Link from "next/link";
import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import type { BriefingRunItem } from "@/app/api/__generated__/models/briefingRunItem";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { getSafeLink, getSubtitleParts } from "../helpers";

interface Props {
  item: BriefingRunItem;
  isHidden?: boolean;
}

// One row per run: the summary doubles as the row's subtitle, so the agent
// name and its finding read as a single line.
export function RunRow({ item, isHidden = false }: Props) {
  const isFailed = item.status !== "COMPLETED";
  const { attribution, text } = getSubtitleParts(item);
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
          {/* Agent names and run summaries are user data: leave them masked in
              session replays rather than taking Text's static-copy default. */}
          <Text
            variant="body-medium"
            unmask={false}
            className="truncate text-foreground"
          >
            {item.agent_name}
          </Text>
          {isFailed ? (
            <span className="shrink-0 rounded-full bg-destructive/10 px-2 py-0.5 text-[0.6875rem] font-medium text-destructive">
              Failed
            </span>
          ) : null}
        </div>
        {text ? (
          <Text
            variant="body"
            unmask={false}
            className="line-clamp-2 text-muted-foreground"
          >
            {/* Attribution matters once more than one agent reports:
                mirrors the thread markdown's "**{agent}**: {summary}". */}
            {attribution ? (
              <span className="text-muted-foreground">{attribution} · </span>
            ) : null}
            {text}
          </Text>
        ) : null}
      </div>
      {link ? (
        <Icon
          icon={ArrowRight01Icon}
          size={16}
          className="shrink-0 self-center text-zinc-300 transition-colors group-hover:text-muted-foreground"
        />
      ) : null}
    </>
  );

  const rowClassName = "group flex items-start gap-3 px-5 py-4";

  // Collapsed rows stay mounted and are clipped by the list's overflow, so
  // without this they keep their place in the tab order and focus lands on
  // rows the reader cannot see.
  return (
    <li aria-hidden={isHidden || undefined}>
      {link ? (
        <Link
          href={link}
          tabIndex={isHidden ? -1 : undefined}
          className={cn(rowClassName, "transition-colors hover:bg-muted/50")}
        >
          {body}
        </Link>
      ) : (
        <div className={rowClassName}>{body}</div>
      )}
    </li>
  );
}
