"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import {
  ArrowRight01Icon,
  PinIcon,
  PinOffIcon,
} from "@hugeicons/core-free-icons";
import { AnimatePresence } from "motion/react";
import { useState } from "react";
import { chatTitle, type ChatRow } from "../helpers";
import { ChatLink } from "./ChatLink";
import { TaskChatBranch } from "./TaskChatBranch";

interface Props {
  row: ChatRow;
  onTogglePin: (sessionID: string, isPinned: boolean) => void;
  isPinPending: boolean;
  /** Some chat in the list has task chats, so every row keeps that column
   *  and the times line up. */
  keepsCountColumn: boolean;
}

/** A main chat; with task chats, a count beside it opens them beneath. A
 *  pin button takes no room until the row is hovered or focused, when it
 *  slides in and the time moves over for it; it moves the chat between
 *  Home's pinned and recent boxes. */
export function ChatRowItem({
  row,
  onTogglePin,
  isPinPending,
  keepsCountColumn,
}: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const title = chatTitle(row.session);
  const count = row.taskChats.length;
  const isPinned = !!row.session.is_pinned;
  const branchID = `home-task-chats-${row.session.id}`;

  return (
    <li>
      <div className="group/chat flex items-center gap-0.5">
        <ChatLink session={row.session} title={title} className="flex-1" />
        <button
          type="button"
          aria-label={isPinned ? `Unpin ${title}` : `Pin ${title}`}
          aria-pressed={isPinned}
          disabled={isPinPending}
          onClick={() => onTogglePin(row.session.id, isPinned)}
          className="grid h-9 w-0 shrink-0 place-items-center overflow-hidden rounded-lg text-foreground-faint opacity-0 transition-[width,opacity,background-color,color] duration-200 ease-out group-focus-within/chat:w-9 group-focus-within/chat:opacity-100 group-hover/chat:w-9 group-hover/chat:opacity-100 hover:bg-surface-tray hover:text-foreground-text focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none focus-visible:ring-inset disabled:opacity-40 motion-reduce:transition-none pointer-coarse:w-9 pointer-coarse:opacity-100"
        >
          <Icon
            icon={isPinned ? PinOffIcon : PinIcon}
            size={14}
            tone="inherit"
            aria-hidden
          />
        </button>
        {count > 0 && (
          <button
            type="button"
            aria-expanded={isOpen}
            aria-controls={branchID}
            aria-label={`${count} ${count === 1 ? "task" : "tasks"} from ${title}`}
            onClick={() => setIsOpen((open) => !open)}
            className="flex h-9 w-12 shrink-0 items-center justify-end gap-1 rounded-lg px-2 text-foreground-faint transition-colors hover:bg-surface-tray focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none"
          >
            <Text variant="inbox" as="span" tone="faint" tabular>
              {count}
            </Text>
            <Icon
              icon={ArrowRight01Icon}
              size={14}
              tone="inherit"
              motion="transform"
              aria-hidden
              className={cn(isOpen && "rotate-90")}
            />
          </button>
        )}
        {count === 0 && keepsCountColumn && (
          <span aria-hidden className="w-12 shrink-0" />
        )}
      </div>
      <AnimatePresence initial={false}>
        {isOpen && <TaskChatBranch id={branchID} taskChats={row.taskChats} />}
      </AnimatePresence>
    </li>
  );
}
