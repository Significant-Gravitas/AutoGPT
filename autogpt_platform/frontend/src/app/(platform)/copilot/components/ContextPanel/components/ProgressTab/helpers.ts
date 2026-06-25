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

export function getLatestTaskList(messages: UIMessage[]): TodoItem[] | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const parts = messages[i]?.parts;
    if (!parts) continue;
    for (let j = parts.length - 1; j >= 0; j--) {
      const part = parts[j] as { type: string; input?: unknown };
      if (part.type !== "tool-TodoWrite") continue;
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
