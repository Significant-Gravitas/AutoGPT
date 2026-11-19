"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Loading03Icon } from "@hugeicons/core-free-icons";
import type { ComponentProps } from "react";
import { ChatRowItem } from "./components/ChatRowItem";
import type { ChatRow } from "./helpers";
import { useLoadMoreOnScroll } from "./useLoadMoreOnScroll";

interface Props {
  /** The box's name, such as "Recent chats". */
  title: string;
  icon: ComponentProps<typeof Icon>["icon"];
  /** Ties the heading to the box; one per box on the page. */
  headingID: string;
  /** The recipient's main chats, each with its task chats. */
  rows: ChatRow[];
  hasMore: boolean;
  isLoadingMore: boolean;
  onLoadMore: () => void;
  onTogglePin: (sessionID: string, isPinned: boolean) => void;
  /** The chat whose pin is being changed. */
  pendingPinID: string | null;
}

export function HomeRecentChats({
  title,
  icon,
  headingID,
  rows,
  hasMore,
  isLoadingMore,
  onLoadMore,
  onTogglePin,
  pendingPinID,
}: Props) {
  const { listRef, sentinelRef } = useLoadMoreOnScroll({
    hasMore,
    isLoading: isLoadingMore,
    onLoadMore,
  });

  const hasTaskChats = rows.some((row) => row.taskChats.length > 0);

  return (
    <section
      aria-labelledby={headingID}
      className="flex w-full min-w-0 flex-col rounded-composer-tray bg-surface-tray p-1 text-left"
    >
      <div className="flex h-9 items-center gap-1.5 pl-2.5 text-foreground-quiet">
        <Icon icon={icon} size={14} tone="inherit" />
        <Text id={headingID} variant="inbox-medium" as="h2" tone="text">
          {title}
        </Text>
      </div>

      {/* Exactly six rows tall (6 × 36px + the 8px padding), so no row is
          cut in half; the rest scroll. */}
      <ul
        ref={listRef}
        className="scrollbar-thin flex max-h-56 flex-1 flex-col overflow-y-auto overscroll-contain rounded-panel bg-surface p-1 shadow-(--inbox-shadow-card)"
      >
        {rows.map((row) => (
          <ChatRowItem
            key={row.session.id}
            row={row}
            onTogglePin={onTogglePin}
            isPinPending={pendingPinID === row.session.id}
            keepsCountColumn={hasTaskChats}
          />
        ))}
        {(hasMore || isLoadingMore) && (
          <li
            ref={sentinelRef}
            className="flex h-9 shrink-0 items-center justify-center"
          >
            {isLoadingMore && (
              <Icon
                icon={Loading03Icon}
                size={14}
                tone="faint"
                spin
                aria-label="Loading more chats"
              />
            )}
          </li>
        )}
      </ul>
    </section>
  );
}
