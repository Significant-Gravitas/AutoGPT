"use client";

import { CornerDownRightIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { RelativeTime } from "@/components/atoms/RelativeTime/RelativeTime";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import type { ExpertOption, TaskActivity } from "../../../../data/types";
import { Button } from "@/components/atoms/Button/Button";
import { splitMentions } from "../../helpers";
import { AssigneeAvatar } from "../AssigneeAvatar/AssigneeAvatar";
import { TurnRow } from "./components/TurnRow/TurnRow";
import { describeActivity } from "./helpers";

interface Props {
  items: TaskActivity[];
  isLoading: boolean;
  experts: ExpertOption[];
  hasMore?: boolean;
  isLoadingMore?: boolean;
  onLoadMore?: () => void;
}

export function ActivityFeed({
  items,
  isLoading,
  experts,
  hasMore = false,
  isLoadingMore = false,
  onLoadMore,
}: Props) {
  const names = ["Otto", ...experts.map((e) => e.name)];

  if (isLoading) {
    return (
      <div className="flex flex-col gap-2">
        <Skeleton className="h-5 w-2/3" />
        <Skeleton className="h-11 w-full" />
        <Skeleton className="h-5 w-1/2" />
      </div>
    );
  }
  if (!items.length) {
    return (
      <Text variant="inbox" tone="faint">
        No activity yet.
      </Text>
    );
  }
  return (
    <div className="flex flex-col gap-2">
      <ol className="flex flex-col gap-2">
        {items.map((item) => (
          <li key={item.id} className="animate-inbox-enter">
            <ActivityRow item={item} names={names} />
          </li>
        ))}
      </ol>
      {hasMore && onLoadMore ? (
        <Button
          variant="ghost-muted"
          size="xs"
          className="self-start"
          loading={isLoadingMore}
          onClick={onLoadMore}
        >
          Load more
        </Button>
      ) : null}
    </div>
  );
}

function ActivityRow({ item, names }: { item: TaskActivity; names: string[] }) {
  if (item.kind === "turn") return <TurnRow item={item} />;

  if (
    item.kind === "comment" ||
    item.kind === "question" ||
    item.kind === "answer"
  ) {
    const isQuestion = item.kind === "question";
    return (
      <div className="flex gap-3 rounded-card bg-surface-wash p-3">
        <AssigneeAvatar assignee={item.actor} size={24} />
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <div className="flex flex-wrap items-center gap-x-2">
            <Text variant="inbox-medium" tone="ink" as="span">
              {item.actor.name}
            </Text>
            {isQuestion ? (
              <span className="inline-flex items-center gap-1.5">
                <span
                  aria-hidden
                  className="size-1.5 shrink-0 rounded-full bg-status-waiting"
                />
                <Text variant="inbox-medium" tone="waiting" as="span">
                  asked
                </Text>
              </span>
            ) : null}
            {item.kind === "answer" ? (
              <span className="inline-flex items-center gap-1">
                <Icon
                  icon={CornerDownRightIcon}
                  tone="quiet"
                  size={12}
                  aria-hidden
                />
                <Text variant="inbox" tone="quiet" as="span">
                  answered
                </Text>
              </span>
            ) : null}
            <Text variant="small" tone="faint" as="span">
              <RelativeTime date={item.created_at} />
            </Text>
          </div>
          <Text
            variant="inbox"
            tone="text"
            className="break-words whitespace-pre-wrap"
          >
            {splitMentions(item.body, names).map((part, index) =>
              part.mention ? (
                <Text key={index} variant="inbox-medium" tone="brand" as="span">
                  {part.text}
                </Text>
              ) : (
                <span key={index}>{part.text}</span>
              ),
            )}
          </Text>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-start gap-2.5 px-3 py-1">
      <AssigneeAvatar assignee={item.actor} size={16} />
      <Text variant="inbox" tone="quiet" className="min-w-0 flex-1 break-words">
        {item.kind === "note" ? null : (
          <>
            <Text variant="inbox-medium" tone="text" as="span">
              {item.actor.name}
            </Text>{" "}
          </>
        )}
        {describeActivity(item)}
      </Text>
      <Text variant="small" tone="faint" as="span" className="shrink-0">
        <RelativeTime date={item.created_at} />
      </Text>
    </div>
  );
}
