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
import { SheetSection } from "../SheetSection/SheetSection";
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
    <SheetSection label="Activity">
      {isLoading ? (
        <div className="flex flex-col gap-2">
          <Skeleton className="h-5 w-2/3" />
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-5 w-1/2" />
        </div>
      ) : !items.length ? (
        <Text variant="small" tone="faint">
          No activity yet.
        </Text>
      ) : (
        <ol className="flex flex-col gap-2">
          {items.map((item) => (
            <li key={item.id} className="animate-inbox-enter">
              <ActivityRow item={item} names={names} />
            </li>
          ))}
        </ol>
      )}
    </SheetSection>
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
            ? "flex gap-3 rounded-xl border border-status-waiting/20 bg-status-waiting/5 p-3"
            : "flex gap-3 rounded-xl bg-surface-tray p-3"
        }
      >
        <AssigneeAvatar assignee={item.actor} size={24} />
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <div className="flex flex-wrap items-center gap-x-2">
            <Text variant="small" weight="medium" tone="ink" as="span">
              {item.actor.name}
            </Text>
            {isQuestion ? (
              <span className="inline-flex items-center gap-1">
                <Icon
                  icon={AlertCircleIcon}
                  tone="waiting"
                  size={12}
                  aria-hidden
                />
                <Text variant="small" tone="waiting" as="span">
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
                <Text variant="small" tone="quiet" as="span">
                  answered
                </Text>
              </span>
            ) : null}
            <Text variant="small" tone="faint" as="span">
              <RelativeTime date={item.created_at} />
            </Text>
          </div>
          <Text
            variant="small"
            tone="text"
            className="break-words whitespace-pre-wrap"
          >
            {splitMentions(item.body, names).map((part, index) =>
              part.mention ? (
                <Text
                  key={index}
                  variant="small"
                  weight="medium"
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
    <div className="flex items-center gap-2.5 px-3 py-1">
      <AssigneeAvatar assignee={item.actor} size={16} />
      <Text variant="small" tone="quiet" className="min-w-0 flex-1">
        <Text variant="small" weight="medium" tone="text" as="span">
          {item.actor.name}
        </Text>{" "}
        {describeActivity(item)}
      </Text>
      <Text variant="small" tone="faint" as="span" className="shrink-0">
        <RelativeTime date={item.created_at} />
      </Text>
    </div>
  );
}
