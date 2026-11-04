"use client";

import {
  AgentInbox,
  type AgentTask,
} from "@/components/atoms/agent-inbox-clean";
import type { Task } from "../../data/types";
import { TaskAssigneeAvatar } from "../TaskAssigneeAvatar/TaskAssigneeAvatar";
import { CardExtra } from "./components/CardExtra";
import { needsExtra, toInboxFields } from "./helpers";
import { useInboxBoard } from "./useInboxBoard";
import { useOverflowRight } from "./useOverflowRight";

interface Props {
  tasks: Task[];
  selectedID: string | null;
  onOpen: (id: string) => void;
}

/** The tasks board: AffectUI's Agent Inbox with our four lanes, the expert
 *  on every card, and what a waiting card needs answered in place. */
export function InboxBoard({ tasks, onOpen }: Props) {
  const board = useInboxBoard();
  const scroller = useOverflowRight<HTMLDivElement>();
  const items: AgentTask[] = tasks.map((task) => ({
    ...toInboxFields(task),
    avatar: task.assignee ? (
      <TaskAssigneeAvatar assignee={task.assignee} size={24} />
    ) : undefined,
    extra:
      (task.state === "needs_you" || task.state === "stuck") &&
      needsExtra(task) ? (
        <CardExtra task={task} />
      ) : undefined,
  }));

  return (
    <div className="relative h-full min-h-0">
      <div
        ref={scroller.ref}
        className="scrollbar-none h-full min-h-0 overflow-auto overscroll-x-contain"
      >
        <AgentInbox
          hideBar
          view="board"
          title="Tasks"
          tasks={items}
          onOpen={(task) => onOpen(task.id)}
          onCancel={(task) => board.cancel(task.id)}
          onApprove={(task) => onOpen(task.id)}
          onReject={(task) => onOpen(task.id)}
          onRetry={(task) => board.retry(task.id)}
        />
      </div>
      {scroller.hiddenRight ? (
        <div
          aria-hidden
          className="pointer-events-none absolute inset-y-0 right-0 w-16 edge-blur-right"
        />
      ) : null}
    </div>
  );
}
