import { toast } from "@/components/molecules/Toast/use-toast";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import type { WebSocketNotification } from "@/lib/autogpt-server-api/types";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { TASK_EVENTS, type TaskEventName } from "./types";
import { invalidateTasks } from "./useTasksData";

function isTaskEvent(n: WebSocketNotification): n is WebSocketNotification & {
  event: TaskEventName;
} {
  const r = n as { event?: string; type?: string };
  if (r.type === "task_event") return true;
  return TASK_EVENTS.includes((r.event ?? r.type) as TaskEventName);
}

function readTitle(n: WebSocketNotification) {
  const payload = n as Record<string, unknown>;
  const task = payload.task as { title?: string; number?: number } | undefined;
  const title = (payload.title ?? task?.title) as string | undefined;
  const number = task?.number as number | undefined;
  const key =
    (payload.task_key as string | undefined) ?? (number ? `T-${number}` : null);
  return [key, title].filter(Boolean).join(" ");
}

/**
 * Refreshes task queries on any task event from the notification WebSocket,
 * and toasts when a task finishes or needs the user. Mount once per page.
 */
export function useTaskLiveUpdates({ withToasts = true } = {}) {
  const api = useBackendAPI();
  const queryClient = useQueryClient();

  useEffect(() => {
    function handle(n: WebSocketNotification) {
      if (!isTaskEvent(n)) return;
      invalidateTasks(queryClient);
      if (!withToasts) return;
      const name = (n as { event?: string }).event ?? n.type;
      if (name === "task_completed") {
        toast({ title: `${readTitle(n)} is done` });
      } else if (name === "task_needs_input") {
        const activity = (n as { activity?: { body?: string } }).activity;
        toast({
          title: `${readTitle(n)} needs you`,
          description: activity?.body ?? undefined,
        });
      }
    }
    return api.onWebSocketMessage("notification", handle);
  }, [api, queryClient, withToasts]);
}
