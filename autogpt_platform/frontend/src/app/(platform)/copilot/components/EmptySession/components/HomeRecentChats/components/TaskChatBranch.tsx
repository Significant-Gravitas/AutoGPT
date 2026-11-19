"use client";

import { motion, useReducedMotion } from "motion/react";
import type { TaskChat } from "../helpers";
import { useOpenTaskDetails } from "../useOpenTaskDetails";
import { TaskChatRow } from "./TaskChatRow";

const EASE_OUT = [0.16, 1, 0.3, 1] as const;
const DURATION_S = 0.22;

interface Props {
  id: string;
  taskChats: TaskChat[];
}

/** A main chat's task chats, hung off a trunk under its state icon like the
 *  branches of a git graph: a line down, and a rounded elbow into each. */
export function TaskChatBranch({ id, taskChats }: Props) {
  const reduceMotion = useReducedMotion();
  const { openTaskDetails } = useOpenTaskDetails();
  return (
    <motion.ul
      id={id}
      initial={{ height: 0, opacity: 0 }}
      animate={{ height: "auto", opacity: 1 }}
      exit={{ height: 0, opacity: 0 }}
      transition={
        reduceMotion
          ? { duration: 0 }
          : { duration: DURATION_S, ease: EASE_OUT }
      }
      className="ml-4 overflow-hidden"
    >
      {taskChats.map((chat, index) => (
        <li key={chat.session.id} className="relative pl-3">
          {index < taskChats.length - 1 && (
            <span
              aria-hidden
              className="absolute inset-y-0 left-0 border-l border-border-base"
            />
          )}
          <span
            aria-hidden
            className="absolute top-0 left-0 h-1/2 w-2.5 rounded-bl-lg border-b border-l border-border-base"
          />
          <TaskChatRow chat={chat} onOpenTask={openTaskDetails} />
        </li>
      ))}
    </motion.ul>
  );
}
