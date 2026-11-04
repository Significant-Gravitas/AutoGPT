import type { WebSocketNotification } from "@/lib/autogpt-server-api/types";

interface TaskEventPayload {
  type?: string;
  event?: string;
  task_key?: string | null;
  title?: string | null;
  task?: {
    number?: number;
    title?: string;
    pending_input?: { question?: string } | null;
  } | null;
  activity?: { body?: string | null } | null;
}

function readKey(p: TaskEventPayload) {
  if (p.task_key) return p.task_key;
  if (p.task?.number) return `T-${p.task.number}`;
  return p.title ?? p.task?.title ?? "A task";
}

/** The toast for a task event, or null when the event isn't worth one. */
export function getTaskToast(n: WebSocketNotification) {
  const p = n as unknown as TaskEventPayload;
  const event = p.event ?? p.type;
  if (event === "task_completed") {
    return { title: `${readKey(p)} is done` };
  }
  if (event === "task_needs_input") {
    const question = p.activity?.body ?? p.task?.pending_input?.question;
    return {
      title: question
        ? `${readKey(p)} needs you: ${question}`
        : `${readKey(p)} needs you`,
    };
  }
  return null;
}
