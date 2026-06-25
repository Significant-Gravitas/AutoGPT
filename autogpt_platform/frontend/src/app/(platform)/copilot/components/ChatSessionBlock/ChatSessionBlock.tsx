"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import {
  CheckCircleIcon,
  CircleNotchIcon,
  HourglassIcon,
} from "@phosphor-icons/react";
import type { ReactNode } from "react";
import { ChatOriginIcon } from "../ChatOriginIcon/ChatOriginIcon";

interface Props {
  title?: string | null;
  titleContent?: ReactNode;
  updatedAt: string;
  sourcePlatform?: string | null;
  isActive?: boolean;
  chatStatus?: string | null;
  showProcessing?: boolean;
  showCompleted?: boolean;
  className?: string;
}

function formatDate(dateString: string) {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return `${diffDays} days ago`;

  const day = date.getDate();
  const ordinal =
    day % 10 === 1 && day !== 11
      ? "st"
      : day % 10 === 2 && day !== 12
        ? "nd"
        : day % 10 === 3 && day !== 13
          ? "rd"
          : "th";
  const month = date.toLocaleDateString("en-US", { month: "short" });
  const year = date.getFullYear();

  return `${day}${ordinal} ${month} ${year}`;
}

export function ChatSessionBlock({
  title,
  titleContent,
  updatedAt,
  sourcePlatform,
  isActive = false,
  chatStatus,
  showProcessing = false,
  showCompleted = false,
  className,
}: Props) {
  const displayTitle = title || "Untitled chat";

  return (
    <div
      className={cn("flex min-w-0 max-w-full items-center gap-2", className)}
    >
      <div className="min-w-0 flex-1">
        <div className="flex min-w-0 items-center">
          <Text
            variant="body"
            className={cn(
              "min-w-0 flex-1 truncate font-normal",
              isActive ? "text-zinc-600" : "text-zinc-800",
            )}
          >
            {titleContent || displayTitle}
          </Text>
        </div>
        <div className="flex items-center gap-1.5">
          <Text variant="small" className="text-neutral-400">
            {formatDate(updatedAt)}
          </Text>
          <ChatOriginIcon sourcePlatform={sourcePlatform} />
        </div>
      </div>
      {chatStatus === "running" ? (
        <span
          aria-label="Session running"
          title="Running"
          data-testid="session-status-running"
          className="inline-flex h-4 w-4 shrink-0 items-center justify-center"
        >
          <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-500" />
        </span>
      ) : null}
      {chatStatus === "queued" ? (
        <span
          aria-label="Session queued"
          title="Queued"
          data-testid="session-status-queued"
          className="inline-flex h-4 w-4 shrink-0 items-center justify-center text-purple-600"
        >
          <HourglassIcon className="h-3.5 w-3.5" weight="bold" />
        </span>
      ) : null}
      {chatStatus !== "running" && showProcessing ? (
        <CircleNotchIcon
          className="h-4 w-4 shrink-0 animate-spin text-zinc-400"
          weight="bold"
        />
      ) : null}
      {showCompleted ? (
        <CheckCircleIcon
          className="h-4 w-4 shrink-0 text-green-500"
          weight="fill"
        />
      ) : null}
    </div>
  );
}
