"use client";

import {
  AlertCircleIcon,
  CornerDownRightIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { RelativeTime } from "@/components/atoms/RelativeTime/RelativeTime";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import type { ExpertOption, TaskActivity } from "../../../../data/types";
import { splitMentions } from "../../helpers";
import { AssigneeAvatar } from "../AssigneeAvatar/AssigneeAvatar";
import { TurnRow } from "./components/TurnRow/TurnRow";
import { describeActivity } from "./helpers";

interface Props {
  items: TaskActivity[];
  isLoading: boolean;
  experts: ExpertOption[];
}

export function ActivityFeed({ items, isLoading, experts }: Props) {
  const names = ["Otto", ...experts.map((e) => e.name)];

  return (
    <section aria-label="Activity" className="flex flex-col gap-3">
      <Text variant="small-semibold" tone="secondary">
        Activity
      </Text>
      {isLoading ? (
        <div className="flex flex-col gap-2">
          <Skeleton className="h-5 w-2/3" />
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-5 w-1/2" />
        </div>
      ) : !items.length ? (
        <Text variant="small" tone="muted">
          No activity yet.
        </Text>
      ) : (
        <ol className="flex flex-col gap-2">
          {items.map((item) => (
            <li key={item.id}>
              <ActivityRow item={item} names={names} />
            </li>
          ))}
        </ol>
      )}
    </section>
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
      <div
        className={
          isQuestion
            ? "flex gap-3 rounded-lg border border-warning-border bg-warning-surface p-3"
            : "flex gap-3 rounded-lg border border-border-base bg-surface p-3"
        }
      >
        <AssigneeAvatar assignee={item.actor} size={24} />
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <div className="flex flex-wrap items-center gap-x-2">
            <Text variant="small-semibold" tone="primary" as="span">
              {item.actor.name}
            </Text>
            {isQuestion ? (
              <span className="inline-flex items-center gap-1">
                <Icon
                  icon={AlertCircleIcon}
                  tone="warning-bright"
                  aria-hidden
                />
                <Text variant="small" tone="warning" as="span">
                  asked
                </Text>
              </span>
            ) : null}
            {item.kind === "answer" ? (
              <span className="inline-flex items-center gap-1">
                <Icon icon={CornerDownRightIcon} tone="muted" aria-hidden />
                <Text variant="small" tone="muted" as="span">
                  answered
                </Text>
              </span>
            ) : null}
            <Text variant="small" tone="subtle" as="span">
              <RelativeTime date={item.created_at} />
            </Text>
          </div>
          <Text
            variant="body-tight"
            tone="body"
            className="break-words whitespace-pre-wrap"
          >
            {splitMentions(item.body, names).map((part, index) =>
              part.mention ? (
                <Text
                  key={index}
                  variant="body-medium-tight"
                  tone="brand"
                  as="span"
                >
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
    <div className="flex items-center gap-2 px-2">
      <AssigneeAvatar assignee={item.actor} size={16} />
      <Text variant="small" tone="muted" className="min-w-0 flex-1">
        <Text variant="small-medium" tone="secondary" as="span">
          {item.actor.name}
        </Text>{" "}
        {describeActivity(item)}
      </Text>
      <Text variant="small" tone="subtle" as="span" className="shrink-0">
        <RelativeTime date={item.created_at} />
      </Text>
    </div>
  );
}
