import type { ExpertFirstJob } from "@/app/api/__generated__/models/expertFirstJob";
import type { ExpertFirstJobStatus } from "@/app/api/__generated__/models/expertFirstJobStatus";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import NextLink from "next/link";
import {
  getFirstJobHref,
  getStatusConfig,
  getStatusLabel,
  trackFirstJobOpened,
} from "../helpers";

interface Props {
  item: ExpertFirstJob;
}

export function FirstJobCard({ item }: Props) {
  const label = getStatusLabel(item.status);

  return (
    <NextLink
      href={getFirstJobHref(item)}
      onClick={() => trackFirstJobOpened(item)}
      aria-label={`${item.name}: ${item.title}, ${label}`}
      className="flex h-full w-full flex-col gap-3 rounded-2xl border border-border-base bg-surface p-4 text-left transition-[border-color,box-shadow] duration-150 hover:border-border-strong hover:shadow-md"
    >
      <div className="flex items-center gap-3">
        <ExpertAvatar name={item.name} avatarUrl={item.avatar_url} size={32} />
        <div className="flex min-w-0 flex-col">
          <Text variant="body-medium" className="truncate">
            {item.name}
          </Text>
          <Text variant="small" tone="muted" className="truncate">
            {item.role}
          </Text>
        </div>
      </div>
      <div className="flex flex-col gap-1">
        <Text variant="body-medium" tone="primary" className="text-pretty">
          {item.title}
        </Text>
        <Text variant="body" tone="muted" className="line-clamp-2">
          {item.description}
        </Text>
      </div>
      <div className="mt-auto flex flex-wrap items-center gap-x-2 gap-y-1">
        <StatusPill status={item.status} />
        {item.detail && item.detail !== label ? (
          <Text variant="small" as="span" tone="muted">
            {item.detail}
          </Text>
        ) : null}
      </div>
    </NextLink>
  );
}

interface StatusPillProps {
  status: ExpertFirstJobStatus;
}

/** Same dot-and-word look as the Home team's StatusBadge, with this
 *  block's own statuses. */
function StatusPill({ status }: StatusPillProps) {
  const config = getStatusConfig(status);

  return (
    <Text
      variant="small-medium"
      as="span"
      tone="muted"
      className="inline-flex shrink-0 items-center gap-1.5"
    >
      <span
        className={cn(
          "size-1.5 rounded-full",
          config.dot,
          config.pulse && "animate-pulse",
        )}
        aria-hidden="true"
      />
      {config.label}
    </Text>
  );
}
