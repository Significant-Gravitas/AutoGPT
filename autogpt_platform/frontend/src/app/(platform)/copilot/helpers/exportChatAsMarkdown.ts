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
