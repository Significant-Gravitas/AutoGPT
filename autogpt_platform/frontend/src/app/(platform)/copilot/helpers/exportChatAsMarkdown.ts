interface SessionChatMessage {
  role: string;
  content: string | null;
  tool_calls: unknown[] | null;
}

interface ToolCall {
  function?: {
    name?: string;
    arguments?: string;
  };
}

function formatToolCalls(toolCalls: unknown[]): string {
  return toolCalls
    .map((tc) => {
      const call = tc as ToolCall;
      const name = call.function?.name ?? "unknown_tool";
      let argsStr = "";
      try {
        const args = JSON.parse(call.function?.arguments ?? "{}") as Record<
          string,
          unknown
        >;
        argsStr = Object.entries(args)
          .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
          .join(", ");
      } catch {
        argsStr = call.function?.arguments ?? "";
      }
      return `> 🔧 \`${name}(${argsStr})\``;
    })
    .join("\n");
}

function sanitizeFilename(name: string): string {
  return name
    .replace(/\.\./g, "_")
    .replace(/[\\/:*?"<>|\x00-\x1f]/g, "_")
    .replace(/^\.+/, "")
    .slice(0, 100);
}

export function exportChatAsMarkdown(
  _sessionId: string,
  title: string | null | undefined,
  messages: SessionChatMessage[],
): void {
  const displayTitle = title || "Untitled chat";
  const date = new Date().toISOString().slice(0, 10);

  const lines: string[] = [`# ${displayTitle}`, `_Exported: ${date}_`, ""];

  for (const msg of messages) {
    if (msg.role === "tool") continue;

    if (msg.role === "user") {
      lines.push("## User", "");
      if (msg.content) lines.push(msg.content, "");
    } else if (msg.role === "assistant") {
      lines.push("## Assistant", "");
      if (msg.tool_calls && msg.tool_calls.length > 0) {
        lines.push(formatToolCalls(msg.tool_calls), "");
      }
      if (msg.content) lines.push(msg.content, "");
    }
  }

  const markdown = lines.join("\n");
  const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `chat-${sanitizeFilename(displayTitle)}-${date}.md`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

const EXPORT_PAGE_SIZE = 200;
const EXPORT_MAX_PAGES = 100;

export async function fetchAndExportChat(
  id: string,
  title: string | null | undefined,
  fetchSession: typeof import("@/app/api/__generated__/endpoints/chat/chat").getV2GetSession,
): Promise<void> {
  const allMessages: SessionChatMessage[] = [];
  let beforeSequence: number | undefined = undefined;
  let truncated = false;

  for (let page = 0; page < EXPORT_MAX_PAGES; page++) {
    const opts: { limit: number; before_sequence?: number } = {
      limit: EXPORT_PAGE_SIZE,
    };
    if (beforeSequence !== undefined) opts.before_sequence = beforeSequence;

    const response = await fetchSession(id, opts);
    if (response.status !== 200) {
      throw new Error(`Failed to fetch session (status: ${response.status})`);
    }

    const pageMessages = (response.data.messages ??
      []) as unknown as SessionChatMessage[];
    allMessages.unshift(...pageMessages);

    const hasMore = !!response.data.has_more_messages;
    const oldestSeq = response.data.oldest_sequence;
    if (!hasMore || oldestSeq == null) break;
    if (page === EXPORT_MAX_PAGES - 1) {
      truncated = true;
      break;
    }
    beforeSequence = oldestSeq;
  }

  if (truncated) {
    throw new Error(
      `Chat export exceeded ${EXPORT_MAX_PAGES * EXPORT_PAGE_SIZE} messages. Please contact support to export this chat.`,
    );
  }

  exportChatAsMarkdown(id, title, allMessages);
}
