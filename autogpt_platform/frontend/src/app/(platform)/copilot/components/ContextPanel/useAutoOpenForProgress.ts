"use client";

import { useEffect, useRef } from "react";
import { useCopilotStreamStore } from "../../copilotStreamStore";
import { useCopilotUIStore } from "../../store";
import {
  getLatestTaskList,
  type TodoItem,
} from "./components/ProgressTab/helpers";

function hasActiveWork(todos: TodoItem[] | null): boolean {
  if (!todos || todos.length === 0) return false;
  return todos.some(
    (t) => t.status === "in_progress" || t.status === "pending",
  );
}

/**
 * Force-opens the context panel on the Progress tab the moment a session's
 * task list goes "active" (has any incomplete todo). Re-arms when the active
 * state flips back to inactive so a subsequent task list reopens the panel.
 */
export function useAutoOpenForProgress(sessionId: string | null) {
  const openContextPanelForProgress = useCopilotUIStore(
    (s) => s.openContextPanelForProgress,
  );
  const messages = useCopilotStreamStore((s) =>
    sessionId ? s.messageSnapshots[sessionId] : undefined,
  );
  const wasActiveRef = useRef(false);
  const prevSessionIdRef = useRef(sessionId);

  useEffect(() => {
    if (prevSessionIdRef.current !== sessionId) {
      wasActiveRef.current = false;
      prevSessionIdRef.current = sessionId;
    }
  }, [sessionId]);

  useEffect(() => {
    const todos = messages ? getLatestTaskList(messages) : null;
    const active = hasActiveWork(todos);
    if (active && !wasActiveRef.current) {
      wasActiveRef.current = true;
      openContextPanelForProgress();
    } else if (!active && wasActiveRef.current) {
      wasActiveRef.current = false;
    }
  }, [messages, openContextPanelForProgress]);
}
