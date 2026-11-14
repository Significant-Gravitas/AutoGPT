"use client";

import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { formatTimeAgo } from "@/lib/utils/time";
import { Clock01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { chatStatus } from "./helpers";

interface Props {
  /** The recipient's latest chats, from `recentChatsFor`. */
  recent: SessionSummaryResponse[];
}

export function HomeRecentChats({ recent }: Props) {
  return (
    <section
      aria-labelledby="home-recent-chats"
      className="flex w-full flex-col rounded-composer-tray bg-surface-tray p-1 text-left"
    >
      <div className="flex h-9 items-center gap-1.5 pl-2.5 text-foreground-quiet">
        <Icon icon={Clock01Icon} size={14} tone="inherit" />
        <Text id="home-recent-chats" variant="inbox-medium" as="h2" tone="text">
          Recent chats
        </Text>
      </div>

      <ul className="flex flex-1 flex-col rounded-panel bg-surface p-1 shadow-(--inbox-shadow-card)">
        {recent.map((session) => {
          const status = chatStatus(session);
          return (
            <li key={session.id}>
              <Link
                href={`/home?sessionId=${session.id}`}
                className="flex h-9 items-center gap-2.5 rounded-lg px-2.5 text-foreground-faint transition-colors hover:bg-surface-tray focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none"
              >
                <span title={status.label} className="flex shrink-0">
                  <Icon
                    icon={status.icon}
                    size={14}
                    tone={status.tone}
                    spin={status.spin}
                    aria-hidden
                  />
                  <span className="sr-only">{status.label}: </span>
                </span>
                <Text
                  variant="inbox"
                  as="span"
                  tone="ink"
                  className="min-w-0 flex-1 truncate"
                >
                  {session.title || "Untitled chat"}
                </Text>
                <Text
                  variant="inbox"
                  as="span"
                  tone="faint"
                  className="shrink-0"
                >
                  {formatTimeAgo(session.updated_at)}
                </Text>
              </Link>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
