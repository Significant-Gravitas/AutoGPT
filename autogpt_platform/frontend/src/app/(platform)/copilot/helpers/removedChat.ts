import { ApiError } from "@/lib/autogpt-server-api/helpers";

export interface RemovedChat {
  expertName: string | null;
}

// Resetting an expert removes its chats, but Otto's cards and Home can still
// link to one. The server answers 410 with a `chat_removed` detail for those,
// so the page can say why instead of "not found".
export function getRemovedChat(error: unknown): RemovedChat | null {
  if (!(error instanceof ApiError) || error.status !== 410) return null;
  const detail: unknown = error.response?.detail;
  if (!detail || typeof detail !== "object") return null;
  const { type, expert_name } = detail as Record<string, unknown>;
  if (type !== "chat_removed") return null;
  return { expertName: typeof expert_name === "string" ? expert_name : null };
}

export function getRemovedChatTitle({ expertName }: RemovedChat) {
  return expertName
    ? `This chat was removed when ${expertName} was reset`
    : "This chat was removed when its expert was reset";
}
