"use client";

import type { Task } from "@/app/api/__generated__/models/task";
import { Icon } from "@/components/atoms/Icon/Icon";
import { SidebarRight01Icon } from "@hugeicons/core-free-icons";
import type { TaskChat } from "../helpers";
import { ChatLink } from "./ChatLink";

interface Props {
  chat: TaskChat;
  onOpenTask: (task: Task) => void;
}

/** A task chat: the row opens the chat, the button beside it the task. The
 *  button sits next to the link, never inside it. */
export function TaskChatRow({ chat, onOpenTask }: Props) {
  const { task } = chat;
  return (
    <div className="flex items-center gap-0.5">
      <ChatLink
        session={chat.session}
        title={chat.title}
        isNested
        className="flex-1"
      />
      {task && (
        <button
          type="button"
          aria-label={`Open task details: ${chat.title}`}
          title="Open task details"
          onClick={() => onOpenTask(task)}
          className="flex size-8 shrink-0 items-center justify-center rounded-lg text-foreground-faint transition-colors hover:bg-surface-tray hover:text-foreground-text focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none focus-visible:ring-inset"
        >
          <Icon
            icon={SidebarRight01Icon}
            size={14}
            tone="inherit"
            aria-hidden
          />
        </button>
      )}
    </div>
  );
}
