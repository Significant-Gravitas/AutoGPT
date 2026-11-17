import type { UIMessage } from "ai";

type Part = UIMessage["parts"][number];

interface Row {
  role: string;
  content: string | null;
  toolCallID: string | null;
  toolCalls: unknown[];
  sequence: number | null;
  createdAt: string | null;
  isHidden: boolean;
}

interface Options {
  /** `active_stream.started_at`: rows from then on belong to the running turn. */
  activeTurnStartedAt?: string | null;
}

// The backend appends this block to a user message that carried attachments.
const ATTACHED_FILES_RE = /\n?\n?\[Attached files\]\n[\s\S]*$/;
// One line per file or folder in that block (backend `build_files_block`).
const ATTACHED_LINE_RE =
  /^- (.+) \((?:[^,()]+, [\d.]+ KB|folder, \d+ file\(s\) directly inside)\), (?:file|folder)_id=[0-9a-f-]+$/;

/**
 * Stored chat rows as AI SDK messages. With a turn still running, its rows are
 * left out: resuming the stream replays that turn from its first chunk.
 */
export function toUIMessages(
  sessionID: string,
  rawMessages: unknown[],
  { activeTurnStartedAt }: Options = {},
) {
  const rows = rawMessages.flatMap((raw) => toRow(raw) ?? []);
  const outputs = new Map<string, unknown>();
  for (const row of rows) {
    if (row.role === "tool" && row.toolCallID && row.content !== null) {
      outputs.set(row.toolCallID, row.content);
    }
  }

  const turnStart = activeTurnStartedAt ? Date.parse(activeTurnStartedAt) : NaN;
  const messages: UIMessage[] = [];

  for (const [index, row] of rows.entries()) {
    if (!["user", "assistant", "reasoning"].includes(row.role)) continue;
    // The hire's kickoff is a control turn sent on the user's behalf; the
    // server marks it hidden and the web never shows it either.
    if (row.isHidden) continue;
    const role = row.role === "user" ? "user" : "assistant";
    if (role === "assistant" && isInActiveTurn(row, turnStart)) break;

    const parts = buildParts(row, outputs);
    if (parts.length === 0) continue;

    const id = `${sessionID}-${row.sequence ?? `idx-${index}`}`;
    const previous = messages[messages.length - 1];
    if (role === "assistant" && previous?.role === "assistant") {
      previous.parts.push(...parts);
      previous.id = id;
      continue;
    }
    const fileNames = role === "user" ? attachedFileNames(row.content) : [];
    messages.push({
      id,
      role,
      parts,
      // As a live send keeps them, so an attachment-only bubble still names its files.
      ...(fileNames.length ? { metadata: { fileIDs: [], fileNames } } : {}),
    });
  }
  return messages;
}

function isInActiveTurn(row: Row, turnStart: number) {
  if (Number.isNaN(turnStart) || !row.createdAt) return false;
  return Date.parse(row.createdAt) >= turnStart;
}

function buildParts(row: Row, outputs: Map<string, unknown>) {
  const parts: Part[] = [];
  const text =
    row.role === "user"
      ? (row.content ?? "").replace(ATTACHED_FILES_RE, "")
      : (row.content ?? "");

  if (text.trim()) {
    parts.push(
      row.role === "reasoning"
        ? { type: "reasoning", text, state: "done" }
        : { type: "text", text, state: "done" },
    );
  }
  if (row.role !== "user") {
    for (const toolCall of row.toolCalls) {
      const part = toToolPart(toolCall, outputs);
      if (part) parts.push(part);
    }
  }
  if (parts.length === 0 && row.role === "user") {
    parts.push({ type: "text", text: "", state: "done" });
  }
  return parts;
}

/** The names listed in a user message's `[Attached files]` block. */
function attachedFileNames(content: string | null) {
  const block = ATTACHED_FILES_RE.exec(content ?? "")?.[0] ?? "";
  return block.split("\n").flatMap((line) => {
    const name = ATTACHED_LINE_RE.exec(line.trim())?.[1];
    return name ? [name] : [];
  });
}

function toToolPart(raw: unknown, outputs: Map<string, unknown>) {
  if (!raw || typeof raw !== "object") return null;
  const call = raw as {
    id?: unknown;
    display_name?: unknown;
    function?: { name?: unknown; arguments?: unknown };
  };
  const toolCallId = typeof call.id === "string" ? call.id.trim() : "";
  const name =
    typeof call.function?.name === "string" ? call.function.name.trim() : "";
  if (!toolCallId || !name) return null;

  const base = {
    type: `tool-${name}`,
    toolCallId,
    title: typeof call.display_name === "string" ? call.display_name : undefined,
    input: parseJSON(call.function?.arguments),
  };
  // Stored rows hold no running turn, so a call with no result was cut off.
  if (!outputs.has(toolCallId)) {
    return { ...base, state: "output-error", errorText: "Stopped before it finished" } as Part;
  }
  return {
    ...base,
    state: "output-available",
    output: parseJSON(outputs.get(toolCallId)),
  } as Part;
}

function parseJSON(value: unknown) {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return value;
  }
}

function toRow(raw: unknown): Row | null {
  if (!raw || typeof raw !== "object") return null;
  const row = raw as Record<string, unknown>;
  if (typeof row.role !== "string") return null;
  return {
    role: row.role,
    content: typeof row.content === "string" ? row.content : null,
    toolCallID: typeof row.tool_call_id === "string" ? row.tool_call_id : null,
    toolCalls: Array.isArray(row.tool_calls) ? row.tool_calls : [],
    sequence: typeof row.sequence === "number" ? row.sequence : null,
    createdAt: typeof row.created_at === "string" ? row.created_at : null,
    isHidden: isHiddenMetadata(row.metadata),
  };
}

/** A row the server keeps out of the thread, such as an expert's kickoff. */
export function isHiddenRow(raw: unknown) {
  if (!raw || typeof raw !== "object") return false;
  return isHiddenMetadata((raw as Record<string, unknown>).metadata);
}

/** A user message's own text and the file ids in its `[Attached files]` block. */
export function splitUserContent(content: string) {
  const block = ATTACHED_FILES_RE.exec(content)?.[0] ?? "";
  return {
    text: content.replace(ATTACHED_FILES_RE, ""),
    fileIDs: [...block.matchAll(/file_id=([0-9a-f-]+)/g)].map((match) => match[1]),
  };
}

function isHiddenMetadata(metadata: unknown) {
  if (!metadata || typeof metadata !== "object") return false;
  const value = metadata as Record<string, unknown>;
  return value.hidden === true || value.kind === "expert_kickoff";
}
