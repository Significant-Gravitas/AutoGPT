"use client";

import {
  CheckCircleIcon,
  CircleDashedIcon,
  CircleIcon,
} from "@phosphor-icons/react";
import { motion } from "framer-motion";
import { useCopilotStreamStore } from "../../../../copilotStreamStore";
import { countCompleted, getLatestTaskList, type TodoItem } from "./helpers";

interface Props {
  sessionId: string | null;
}

export function ProgressTab({ sessionId }: Props) {
  const messages = useCopilotStreamStore((s) =>
    sessionId ? s.messageSnapshots[sessionId] : undefined,
  );
  const isStreaming = useCopilotStreamStore((s) => s.isStreaming);

  if (!sessionId) {
    return (
      <p className="p-6 text-center text-sm text-zinc-400">
        Start a conversation to see progress here.
      </p>
    );
  }

  const todos = messages ? getLatestTaskList(messages) : null;

  if (!todos || todos.length === 0) {
    return (
      <div className="flex h-full flex-1 items-center justify-center p-6">
        <p className="text-center text-sm text-zinc-400">
          No task list yet. Autopilot will populate it as work begins.
        </p>
      </div>
    );
  }

  const completed = countCompleted(todos);
  const allDone = completed === todos.length;

  if (allDone) return <AllTasksComplete total={todos.length} />;

  // When the agent is idle, render in_progress entries as pending so the
  // sidebar doesn't display an active spinner/bold treatment for work that
  // isn't actually progressing.
  const isIdle = !isStreaming;
  const displayTodos: TodoItem[] = isIdle
    ? todos.map((t) =>
        t.status === "in_progress" ? { ...t, status: "pending" } : t,
      )
    : todos;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-6 pb-3 pt-4">
      <header className="flex items-baseline justify-between">
        <h3 className="text-sm font-medium text-zinc-900">Task list</h3>
        <span className="text-xs text-zinc-500">
          {completed}/{todos.length} completed
        </span>
      </header>
      <ul className="flex flex-col gap-1.5">
        {displayTodos.map((todo, i) => (
          <TaskRow key={i} todo={todo} />
        ))}
      </ul>
    </div>
  );
}

function TaskRow({ todo }: { todo: TodoItem }) {
  return (
    <li className="flex items-start gap-2 text-xs">
      <span className="mt-0.5 flex-shrink-0">
        <StatusIcon status={todo.status} />
      </span>
      <span
        className={
          todo.status === "completed"
            ? "text-muted-foreground line-through"
            : todo.status === "in_progress"
              ? "font-medium text-foreground"
              : "text-muted-foreground"
        }
      >
        {todo.status === "in_progress" && todo.activeForm
          ? todo.activeForm
          : todo.content}
      </span>
    </li>
  );
}

function StatusIcon({ status }: { status: TodoItem["status"] }) {
  if (status === "completed") {
    return (
      <CheckCircleIcon
        size={14}
        weight="fill"
        className="text-green-500"
        aria-label="completed"
      />
    );
  }
  if (status === "in_progress") {
    return (
      <CircleDashedIcon
        size={14}
        weight="bold"
        className="text-blue-500"
        aria-label="in progress"
      />
    );
  }
  return (
    <CircleIcon
      size={14}
      weight="regular"
      className="text-neutral-400"
      aria-label="pending"
    />
  );
}

function AllTasksComplete({ total }: { total: number }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 py-12 text-center">
      <motion.div
        initial={{ scale: 0, rotate: -30 }}
        animate={{ scale: 1, rotate: 0 }}
        transition={{ type: "spring", stiffness: 260, damping: 18 }}
        className="flex h-14 w-14 items-center justify-center rounded-full bg-emerald-50"
      >
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: [0, 1.15, 1] }}
          transition={{ duration: 0.5, delay: 0.15, ease: "easeOut" }}
        >
          <CheckCircleIcon
            size={32}
            weight="fill"
            className="text-emerald-500"
          />
        </motion.div>
      </motion.div>
      <motion.h3
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.25 }}
        className="text-sm font-medium text-zinc-900"
      >
        All tasks complete
      </motion.h3>
      <motion.p
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.35 }}
        className="max-w-xs text-xs text-zinc-500"
      >
        {total} {total === 1 ? "task" : "tasks"} finished.
        <br />
        Autopilot is ready for the next ask.
      </motion.p>
    </div>
  );
}
