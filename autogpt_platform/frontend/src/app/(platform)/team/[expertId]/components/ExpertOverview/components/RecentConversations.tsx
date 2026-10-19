"use client";

import Link from "next/link";
import {
  BubbleChatIcon,
  Comment01Icon,
  Message01Icon,
} from "@hugeicons/core-free-icons";
import { formatRelativeDate } from "@/app/(platform)/artifacts/components/ArtifactsList/helpers";
import { PanelEmptyState } from "@/app/(platform)/copilot/components/ContextPanel/components/PanelEmptyState/PanelEmptyState";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { CardError } from "./CardError";
import { OverviewCard } from "./OverviewCard";

interface Props {
  expertName: string;
  sessions: SessionSummaryResponse[];
  isLoading: boolean;
  isError: boolean;
  onRetry: () => unknown;
  onChat: () => void;
}

export function RecentConversations({
  expertName,
  sessions,
  isLoading,
  isError,
  onRetry,
  onChat,
}: Props) {
  return (
    <OverviewCard title="Recent conversations">
      {isLoading ? (
        <div className="flex flex-col gap-2">
          <Skeleton className="h-10 w-full rounded-lg" />
          <Skeleton className="h-10 w-full rounded-lg" />
        </div>
      ) : isError ? (
        <CardError message="Couldn't load conversations." onRetry={onRetry} />
      ) : sessions.length === 0 ? (
        <PanelEmptyState
          tiles={[
            <Icon
              key="a"
              icon={Comment01Icon}
              size={22}
              className="text-sky-500"
            />,
            <Icon
              key="b"
              icon={Message01Icon}
              size={22}
              className="text-amber-500"
            />,
            <Icon
              key="c"
              icon={BubbleChatIcon}
              size={22}
              className="text-violet-500"
            />,
          ]}
          title="No conversations yet"
          description={`Ask ${expertName} for something and the chat shows up here.`}
        >
          <Button
            variant="primary"
            size="small"
            leadingIcon={BubbleChatIcon}
            onClick={onChat}
          >
            Start a chat
          </Button>
        </PanelEmptyState>
      ) : (
        <ul className="-mx-2 flex flex-col">
          {sessions.map((session) => (
            <li key={session.id}>
              <Link
                href={`/home?sessionId=${session.id}`}
                className="flex items-center gap-3 rounded-xl px-2 py-2 transition-colors hover:bg-zinc-50"
              >
                <span className="flex size-9 shrink-0 items-center justify-center rounded-xl border border-zinc-200 bg-white text-zinc-500">
                  <Icon icon={BubbleChatIcon} size={16} aria-hidden />
                </span>
                <span className="flex min-w-0 flex-1 flex-col">
                  <Text
                    variant="body-medium"
                    as="span"
                    className="truncate !text-zinc-900"
                  >
                    {session.title || "Untitled chat"}
                  </Text>
                  <Text variant="small" as="span" className="!text-zinc-500">
                    {session.is_processing
                      ? "Working now"
                      : formatRelativeDate(session.updated_at)}
                  </Text>
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </OverviewCard>
  );
}
