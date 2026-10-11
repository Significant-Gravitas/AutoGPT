"use client";

import type { PausedScheduledChat } from "@/app/api/__generated__/models/pausedScheduledChat";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { PauseIcon, PlayIcon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { usePausedChatListItem } from "./usePausedChatListItem";

interface Props {
  paused: PausedScheduledChat;
}

export function PausedChatListItem({ paused }: Props) {
  const { sessionHref, handleResume, isResuming } = usePausedChatListItem({
    paused,
  });

  const detail = (
    <>
      <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-large border border-slate-50 bg-amber-50">
        <Icon icon={PauseIcon} size={18} className="text-amber-700" />
      </div>
      <div className="flex min-w-0 flex-col gap-1">
        <Text
          variant="body-medium"
          className="block w-full truncate text-ellipsis"
        >
          {paused.title}
        </Text>
        <Text variant="small" className="!text-amber-700">
          {paused.summary}
        </Text>
      </div>
    </>
  );

  return (
    <div
      className="flex w-full flex-col gap-3 rounded-large border border-amber-200 bg-white p-4 sm:flex-row sm:items-center sm:justify-between"
      data-testid="paused-chat-row"
      data-schedule-key={paused.schedule_key}
    >
      {sessionHref ? (
        <Link
          href={sessionHref}
          className="flex min-w-0 flex-1 items-start gap-3 hover:opacity-80"
        >
          {detail}
        </Link>
      ) : (
        <div className="flex min-w-0 flex-1 items-start gap-3">{detail}</div>
      )}
      <div className="flex flex-shrink-0 items-center gap-2">
        <Button
          variant="secondary"
          size="small"
          loading={isResuming}
          onClick={handleResume}
          aria-label={`Resume ${paused.title}`}
        >
          <Icon icon={PlayIcon} className="mr-1 h-4 w-4" />
          Resume
        </Button>
      </div>
    </div>
  );
}
