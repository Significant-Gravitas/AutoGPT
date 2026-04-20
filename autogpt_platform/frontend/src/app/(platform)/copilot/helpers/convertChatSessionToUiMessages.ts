import { getGetWorkspaceDownloadFileByIdUrl } from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { FileUIPart, UIMessage, UIDataTypes, UITools } from "ai";

interface SessionChatMessage {
  role: string;
  content: string | null;
  tool_call_id: string | null;
  tool_calls: unknown[] | null;
  sequence: number | null;
  duration_ms: number | null;
}

function coerceSessionChatMessages(
  rawMessages: unknown[],
): SessionChatMessage[] {
  return rawMessages
    .map((m) => {
      if (!m || typeof m !== "object") return null;
      const msg = m as Record<string, unknown>;

      const role = typeof msg.role === "string" ? msg.role : null;
      if (!role) return null;

      return {
        role,
        content:
          typeof msg.content === "string"
            ? msg.content
            : msg.content == null
              ? null
              : String(msg.content),
        tool_call_id:
          typeof msg.tool_call_id === "string"
            ? msg.tool_call_id
            : msg.tool_call_id == null
              ? null
              : String(msg.tool_call_id),
        tool_calls: Array.isArray(msg.tool_calls) ? msg.tool_calls : null,
        sequence: typeof msg.sequence === "number" ? msg.sequence : null,
        duration_ms:
          typeof msg.duration_ms === "number" ? msg.duration_ms : null,
      };
    })
    .filter((m): m is SessionChatMessage => m !== null);
}

/**
 * Parse the `[Attached files]` block appended by the backend and return
 * the cleaned text plus reconstructed FileUIPart objects.
 *
 * Backend format:
 * ```
 * \n\n[Attached files]
 * - name.jpg (image/jpeg, 191.0 KB), file_id=<uuid>
 * Use read_workspace_file with the file_id to access file contents.
 * ```
 */
const ATTACHED_FILES_RE =
  /\n?\n?\[Attached files\]\n([\s\S]*?)Use read_workspace_file with the file_id to access file contents\./;
const FILE_LINE_RE = /^- (.+) \(([^,]+),\s*[\d.]+ KB\), file_id=([0-9a-f-]+)$/;

function extractFileParts(content: string): {
  cleanText: string;
  fileParts: FileUIPart[];
} {
  const match = content.match(ATTACHED_FILES_RE);
  if (!match) return { cleanText: content, fileParts: [] };

  const cleanText = content.replace(match[0], "").trim();
  const lines = match[1].trim().split("\n");
  const fileParts: FileUIPart[] = [];

  for (const line of lines) {
    const m = line.trim().match(FILE_LINE_RE);
    if (!m) continue;
    const [, filename, mimeType, fileId] = m;
    const apiPath = getGetWorkspaceDownloadFileByIdUrl(fileId);
    fileParts.push({
      type: "file",
      filename,
      mediaType: mimeType,
      url: `/api/proxy${apiPath}`,
    });
  }

  return { cleanText, fileParts };
}

function safeJsonParse(value: string): unknown {
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return value;
  }
}

function toToolInput(rawArguments: unknown): unknown {
  if (typeof rawArguments === "string") {
    const trimmed = rawArguments.trim();
    return trimmed ? safeJsonParse(trimmed) : {};
  }
  if (rawArguments && typeof rawArguments === "object") return rawArguments;
  return {};
}

/**
 * Concatenate two UIMessage arrays, merging consecutive assistant messages
 * at the join point so that reasoning + response parts stay in a single bubble.
 *
 * Within each page, `convertChatSessionMessagesToUiMessages` already merges
 * consecutive assistant DB rows. This handles the boundary between pages
 * (or between older-pages and the current/streaming page).
 */
export function concatWithAssistantMerge(
  a: UIMessage<unknown, UIDataTypes, UITools>[],
  b: UIMessage<unknown, UIDataTypes, UITools>[],
): UIMessage<unknown, UIDataTypes, UITools>[] {
  if (a.length === 0) return b;
  if (b.length === 0) return a;
  const last = a[a.length - 1];
  const first = b[0];
  if (last.role === "assistant" && first.role === "assistant") {
    return [
      ...a.slice(0, -1),
      { ...last, parts: [...last.parts, ...first.parts] },
      ...b.slice(1),
    ];
  }
  return [...a, ...b];
}

/**
 * Extract a toolCallId → output map from raw API messages.
 * Used to provide cross-page tool output context when converting
 * older pages that may have assistant tool_calls whose results
 * are in a newer page.
 */
export function extractToolOutputsFromRaw(
  rawMessages: unknown[],
): Map<string, unknown> {
  const map = new Map<string, unknown>();
  for (const raw of rawMessages) {
    if (!raw || typeof raw !== "object") continue;
    const msg = raw as Record<string, unknown>;
    if (
      msg.role === "tool" &&
      typeof msg.tool_call_id === "string" &&
      msg.content != null
    ) {
      map.set(
        msg.tool_call_id,
        typeof msg.content === "string" ? msg.content : String(msg.content),
      );
    }
  }
  return map;
}

export function convertChatSessionMessagesToUiMessages(
  sessionId: string,
  rawMessages: unknown[],
  options?: {
    isComplete?: boolean;
    /** Tool outputs from adjacent pages, for cross-page tool_call matching. */
    extraToolOutputs?: Map<string, unknown>;
  },
): {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
  durations: Map<string, number>;
} {
  const messages = coerceSessionChatMessages(rawMessages);
  const toolOutputsByCallId = new Map<string, unknown>();

  // Seed with extra tool outputs from adjacent pages first;
  // outputs from this page will override if present in both.
  if (options?.extraToolOutputs) {
    for (const [id, output] of options.extraToolOutputs) {
      toolOutputsByCallId.set(id, output);
    }
  }

  for (const msg of messages) {
    if (msg.role !== "tool") continue;
    if (!msg.tool_call_id) continue;
    if (msg.content == null) continue;
    toolOutputsByCallId.set(msg.tool_call_id, msg.content);
  }

  const uiMessages: UIMessage<unknown, UIDataTypes, UITools>[] = [];
  const durations = new Map<string, number>();

  messages.forEach((msg, idx) => {
    if (msg.role === "tool") return;
    if (
      msg.role !== "user" &&
      msg.role !== "assistant" &&
      msg.role !== "reasoning"
    )
      return;

    // Role=="reasoning" rows carry extended_thinking content.  Treat them as
    // contributing a reasoning part to the surrounding assistant bubble —
    // the consecutive-assistant merge below then folds them into the same
    // UIMessage as the text that follows.
    const uiRole: "user" | "assistant" =
      msg.role === "reasoning" ? "assistant" : msg.role;

    const parts: UIMessage<unknown, UIDataTypes, UITools>["parts"] = [];

    if (typeof msg.content === "string" && msg.content.trim()) {
      if (msg.role === "reasoning") {
        parts.push({
          type: "reasoning",
          text: msg.content,
          state: "done",
        } as UIMessage<unknown, UIDataTypes, UITools>["parts"][number]);
      } else if (msg.role === "user") {
        const { cleanText, fileParts } = extractFileParts(msg.content);
        if (cleanText) {
          parts.push({ type: "text", text: cleanText, state: "done" });
        }
        for (const fp of fileParts) {
          parts.push(fp);
        }
      } else {
        parts.push({ type: "text", text: msg.content, state: "done" });
      }
    }

    if (uiRole === "assistant" && Array.isArray(msg.tool_calls)) {
      for (const rawToolCall of msg.tool_calls) {
        if (!rawToolCall || typeof rawToolCall !== "object") continue;
        const toolCall = rawToolCall as {
          id?: unknown;
          function?: { name?: unknown; arguments?: unknown };
        };

        const toolCallId = String(toolCall.id ?? "").trim();
        const toolName = String(toolCall.function?.name ?? "").trim();
        if (!toolCallId || !toolName) continue;

        const input = toToolInput(toolCall.function?.arguments);
        const output = toolOutputsByCallId.get(toolCallId);

        if (output !== undefined) {
          parts.push({
            type: `tool-${toolName}`,
            toolCallId,
            state: "output-available",
            input,
            output: typeof output === "string" ? safeJsonParse(output) : output,
          });
        } else if (options?.isComplete) {
          // Session is complete (no active stream) but this tool call has
          // no output in the DB — mark as completed to stop stale spinners.
          parts.push({
            type: `tool-${toolName}`,
            toolCallId,
            state: "output-available",
            input,
            output: "",
          });
        } else {
          parts.push({
            type: `tool-${toolName}`,
            toolCallId,
            state: "input-available",
            input,
          });
        }
      }
    }

    // User messages must always be rendered, even with empty content, so the
    // initial prompt is visible when reloading a session.
    if (parts.length === 0 && msg.role === "user") {
      parts.push({ type: "text", text: "", state: "done" });
    }
    if (parts.length === 0) return;

    // Merge consecutive assistant messages (including reasoning rows) into a
    // single UIMessage to avoid split bubbles on page reload.
    const prevUI = uiMessages[uiMessages.length - 1];
    if (uiRole === "assistant" && prevUI && prevUI.role === "assistant") {
      prevUI.parts.push(...parts);
      // Capture duration on merged message (last assistant msg wins)
      if (msg.duration_ms != null) {
        durations.set(prevUI.id, msg.duration_ms);
      }
      return;
    }

    // Fall back to the loop index when sequence is unexpectedly absent so
    // multiple sequence-less messages don't collide on the same React key.
    const msgId =
      msg.sequence != null
        ? `${sessionId}-seq-${msg.sequence}`
        : `${sessionId}-idx-${idx}`;
    uiMessages.push({
      id: msgId,
      role: uiRole,
      parts,
    });

    if (uiRole === "assistant" && msg.duration_ms != null) {
      durations.set(msgId, msg.duration_ms);
    }
  });

  return { messages: uiMessages, durations };
}
