import { getGetWorkspaceDownloadFileByIdUrl } from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { FileUIPart, UIMessage, UIDataTypes, UITools } from "ai";

export interface TurnStats {
  durationMs?: number;
  createdAt?: string;
  /** Raw ChatMessage.id (UUID).  Carried for the badge's cancel handler. */
  rawMessageId?: string | null;
  /** True iff this is the latest user message in the session.  The
   *  "Queued" badge anchors on this row whenever the OWNING session's
   *  ``chat_status === "queued"`` (checked at render time). */
  isLatestUserMessage?: boolean;
}

export type TurnStatsMap = Map<string, TurnStats>;

interface SessionChatMessage {
  id: string | null;
  role: string;
  content: string | null;
  tool_call_id: string | null;
  tool_calls: unknown[] | null;
  sequence: number | null;
  duration_ms: number | null;
  created_at: string | null;
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
        id: typeof msg.id === "string" ? msg.id : null,
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
        // The API mutator transforms ISO strings to Date objects before
        // the data reaches here, so accept both string and Date.
        created_at:
          typeof msg.created_at === "string"
            ? msg.created_at
            : msg.created_at instanceof Date
              ? msg.created_at.toISOString()
              : null,
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

// Capture trailing sequence number from a hydrated UIMessage id of the
// shape ``<sessionId>-seq-<N>``.  Streaming-path ids (AI SDK uuids) and
// idx-based fallback ids (``-idx-<N>``) don't match — return null so the
// caller refuses the merge in those cases (the safer default).
const HYDRATED_ID_SEQ_RE = /-seq-(\d+)$/;

function extractDbSequence(uiMessage: UIMessage): number | null {
  if (typeof uiMessage.id !== "string") return null;
  const match = HYDRATED_ID_SEQ_RE.exec(uiMessage.id);
  return match ? Number(match[1]) : null;
}

/**
 * Concatenate two UIMessage arrays, merging consecutive assistant messages
 * at the join point so that reasoning + response parts stay in a single bubble.
 *
 * Within each page, `convertChatSessionMessagesToUiMessages` already merges
 * consecutive assistant DB rows.  This handles the boundary between pages
 * (or between older-pages and the current/streaming page).
 *
 * The merge is gated on **DB-sequence adjacency**: extract the trailing
 * ``-seq-N`` from each id and only merge when ``firstSeq === lastSeq + 1``.
 * Without that check, a hydration-race window (where a user/reasoning row
 * exists in the DB but has not yet been hydrated into either page) would
 * silently swallow the missing row by stitching the two surrounding
 * assistant bubbles into one — matching the chip-disappearing /
 * "merged as previous chat" report.
 */
export function concatWithAssistantMerge(
  a: UIMessage<unknown, UIDataTypes, UITools>[],
  b: UIMessage<unknown, UIDataTypes, UITools>[],
): UIMessage<unknown, UIDataTypes, UITools>[] {
  if (a.length === 0) return b;
  if (b.length === 0) return a;
  const last = a[a.length - 1];
  const first = b[0];
  if (last.role !== "assistant" || first.role !== "assistant") {
    return [...a, ...b];
  }
  // Both sides assistant — only merge when the underlying DB sequences are
  // strictly adjacent (no skipped user / reasoning row between them that a
  // hydration race could be hiding).  Streaming-path ids fail extraction and
  // refuse the merge, which is fine: the streaming consumer handles its own
  // assistant continuity inside the active turn.
  const lastSeq = extractDbSequence(last);
  const firstSeq = extractDbSequence(first);
  if (lastSeq === null || firstSeq === null || firstSeq !== lastSeq + 1) {
    return [...a, ...b];
  }
  return [
    ...a.slice(0, -1),
    { ...last, parts: [...last.parts, ...first.parts] },
    ...b.slice(1),
  ];
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
  stats: TurnStatsMap;
} {
  const messages = coerceSessionChatMessages(rawMessages);
  // Find the most-recent user message — when the session is queued, this
  // is the message that's waiting and renders the "Queued" badge.
  const latestUserMessageIndex = messages.findLastIndex(
    (m) => m.role === "user",
  );
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
  const stats: TurnStatsMap = new Map();

  function patchStats(id: string, patch: Partial<TurnStats>) {
    const existing = stats.get(id) ?? {};
    stats.set(id, { ...existing, ...patch });
  }

  messages.forEach((msg, idx) => {
    if (msg.role === "tool") return;
    if (
      msg.role !== "user" &&
      msg.role !== "assistant" &&
      msg.role !== "reasoning"
    )
      return;

    // Cancelled rows stay visible in the conversation as orphan user
    // bubbles (no AI follow-up after them).  We don't emit a separate
    // "Cancelled" indicator — the row's lack of a response, combined
    // with the user remembering they just clicked X, communicates it.

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
    //
    // The merged bubble's ``id`` advances to the LAST DB sequence in the
    // group (i.e. the row we are currently appending) so
    // ``concatWithAssistantMerge`` can read the correct adjacency from the
    // id alone — without that, a merged bubble holding seq=5+6 would still
    // be keyed ``-seq-5``, and a cross-page assistant at seq=7 would fail
    // the ``firstSeq === lastSeq + 1`` check (7 !== 5+1) and split into two
    // bubbles instead of joining the ongoing turn.
    const prevUI = uiMessages[uiMessages.length - 1];
    if (uiRole === "assistant" && prevUI && prevUI.role === "assistant") {
      prevUI.parts.push(...parts);
      const oldId = prevUI.id;
      const newId =
        msg.sequence != null
          ? `${sessionId}-seq-${msg.sequence}`
          : `${sessionId}-idx-${idx}`;
      if (newId !== oldId) {
        // Migrate stats over to the new id and let the old key go.
        const existingStats = stats.get(oldId);
        stats.delete(oldId);
        if (existingStats) stats.set(newId, existingStats);
        prevUI.id = newId;
      }
      // Capture duration on merged message (last assistant msg wins)
      if (msg.duration_ms != null) {
        patchStats(prevUI.id, { durationMs: msg.duration_ms });
      }
      // Advance createdAt to the latest row in the merge so the live
      // "Thinking Xs" counter anchors to the most recent sub-step rather
      // than the turn's first assistant row.
      const existingCreatedAt = stats.get(prevUI.id)?.createdAt;
      if (
        msg.created_at &&
        (!existingCreatedAt || msg.created_at > existingCreatedAt)
      ) {
        patchStats(prevUI.id, { createdAt: msg.created_at });
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

    const patch: Partial<TurnStats> = {};
    if (msg.created_at) patch.createdAt = msg.created_at;
    if (uiRole === "assistant" && msg.duration_ms != null) {
      patch.durationMs = msg.duration_ms;
    }
    if (uiRole === "user") {
      // Queue badge consumes ``rawMessageId`` for its cancel handler and
      // ``isLatestUserMessage`` to pick the anchor row.  The badge's
      // gating on ``session.chat_status === "queued"`` is the consumer's
      // (ChatMessagesContainer's) concern.
      patch.rawMessageId = msg.id;
      patch.isLatestUserMessage = idx === latestUserMessageIndex;
    }
    if (Object.keys(patch).length > 0) patchStats(msgId, patch);
  });

  return { messages: uiMessages, stats };
}
