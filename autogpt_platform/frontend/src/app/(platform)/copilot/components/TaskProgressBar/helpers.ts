import type { UIMessage } from "ai";

export interface TodoItem {
  content: string;
  status: "pending" | "in_progress" | "completed";
  activeForm?: string;
}

function isTodoItem(value: unknown): value is TodoItem {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.content === "string" &&
    (v.status === "pending" ||
      v.status === "in_progress" ||
      v.status === "completed")
  );
}

// Walks the message history backwards for the most recent TodoWrite tool call
// and returns its task list (the copilot's live plan).
export function getLatestTaskList(messages: UIMessage[]): TodoItem[] | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const parts = messages[i]?.parts;
    if (!parts) continue;
    for (let j = parts.length - 1; j >= 0; j--) {
      const part = parts[j] as {
        type: string;
        state?: string;
        input?: unknown;
      };
      if (part.type !== "tool-TodoWrite") continue;
      // While a TodoWrite call's arguments are still streaming, the AI SDK
      // exposes a progressively-parsed partial input — often a truncated todos
      // array (e.g. only the first N completed items). Rendering that makes
      // the bar flash "all complete" and jump around, so wait for the full
      // input and keep showing the previous complete list until then.
      if (part.state === "input-streaming") continue;
      const input = part.input;
      if (!input || typeof input !== "object") continue;
      const todos = (input as { todos?: unknown }).todos;
      if (!Array.isArray(todos)) continue;
      const filtered = todos.filter(isTodoItem);
      if (filtered.length > 0) return filtered;
    }
  }
  return null;
}

export function countCompleted(todos: TodoItem[]): number {
  return todos.filter((t) => t.status === "completed").length;
}

export function isAllComplete(todos: TodoItem[]): boolean {
  return todos.length > 0 && countCompleted(todos) === todos.length;
}

// The task the user should focus on: the active one while streaming, otherwise
// the first unfinished item so the collapsed bar always has something to show.
export function getCurrentTask(
  todos: TodoItem[],
  isStreaming: boolean,
): TodoItem | null {
  if (isStreaming) {
    const active = todos.find((t) => t.status === "in_progress");
    if (active) return active;
  }
  const pending = todos.find((t) => t.status !== "completed");
  return pending ?? todos[todos.length - 1] ?? null;
}

export function getCurrentLabel(todo: TodoItem): string {
  if (todo.status === "in_progress" && todo.activeForm) return todo.activeForm;
  return todo.content;
}

// A task that was in_progress when the agent went idle (user Stop, end of turn,
// or error) is "stopped" — shown with a distinct amber icon instead of a spinner
// or a plain pending circle.
export type DisplayStatus = TodoItem["status"] | "stopped";

export function toDisplayStatus(
  status: TodoItem["status"],
  isStreaming: boolean,
): DisplayStatus {
  return !isStreaming && status === "in_progress" ? "stopped" : status;
}
