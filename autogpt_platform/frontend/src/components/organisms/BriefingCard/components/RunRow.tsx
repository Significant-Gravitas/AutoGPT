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
      <div data-slot="briefing-card-run-row-body" className="min-w-0 flex-1">
        <div
          data-slot="briefing-card-run-row-header"
          className="flex items-center gap-2"
        >
          {/* Agent names and run summaries are user data: never opt them into
              unmasking in session replays. */}
          <Text variant="body-medium" tone="primary" className="truncate">
            {item.agent_name}
          </Text>
          {isFailed ? (
            <span
              data-slot="briefing-card-run-row-status"
              // TODO(tokens): no text role for red-600 (#D93636); error-hover
              // is the hover fill.
              className="shrink-0 rounded-full bg-error-surface px-2 py-0.5 text-caption-sm font-medium leading-inherit text-red-600"
            >
              Failed
            </span>
          ) : null}
        </div>
        {text ? (
          <Text variant="body" tone="muted" className="line-clamp-2">
            {/* Attribution matters once more than one agent reports:
                mirrors the thread markdown's "**{agent}**: {summary}". */}
            {attribution ? (
              <span
                data-slot="briefing-card-run-row-attribution"
                className="text-foreground-subtle"
              >
                {attribution} ·{" "}
              </span>
            ) : null}
            {text}
          </Text>
        ) : null}
      </div>
      {link ? (
        <Icon
          icon={ArrowRight01Icon}
          size={16}
          className="shrink-0 self-center text-foreground-disabled transition-colors group-hover:text-foreground-tertiary"
        />
      ) : null}
    </>
  );

  const rowClassName = "group flex items-start gap-3 px-5 py-4";

  // Collapsed rows stay mounted and are clipped by the list's overflow, so
  // without this they keep their place in the tab order and focus lands on
  // rows the reader cannot see.
  return (
    <li data-slot="briefing-card-run-row" aria-hidden={isHidden || undefined}>
      {link ? (
        <Link
          href={link}
          tabIndex={isHidden ? -1 : undefined}
          data-slot="briefing-card-run-row-content"
          className={cn(
            rowClassName,
            "transition-colors hover:bg-surface-muted",
          )}
        >
          {body}
        </Link>
      ) : (
        <div data-slot="briefing-card-run-row-content" className={rowClassName}>
          {body}
        </div>
      )}
    </li>
  );
}
