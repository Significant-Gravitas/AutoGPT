import { asNumber, asRecord, asString, nonEmptyString } from "../lib/json";
import type { BackendClient } from "./backend";
import { errorMessage, HttpError } from "./errors";
import type { OpenResponse } from "./http";

export interface SessionSummary {
  id: string;
  createdAt: string | null;
  updatedAt: string | null;
  title: string | null;
  chatStatus: string;
  isProcessing: boolean;
  isPinned: boolean;
  expertID: string | null;
}

export interface SessionPage {
  sessions: SessionSummary[];
  total: number;
}

/** Where a running turn's last checkpoint sits: its first `rows` rows start at DB `sequence`. */
export interface TurnCheckpoint {
  entryID: string;
  rows: number;
  sequence: number;
}

export interface ActiveStream {
  turnID: string;
  startedAt: string | null;
  checkpoint: TurnCheckpoint | null;
}

export interface SessionDetail {
  id: string;
  chatStatus: string;
  messages: unknown[];
  activeStream: ActiveStream | null;
  hasMore: boolean;
  oldestSequence: number | null;
  expertID: string | null;
}

export async function listSessions(
  api: BackendClient,
  args: {
    limit: number;
    offset: number;
    expertID?: string;
    pinnedFirst?: boolean;
    signal?: AbortSignal;
  },
): Promise<SessionPage> {
  const body = asRecord(
    await api.json<unknown>("/api/chat/sessions", {
      query: {
        limit: args.limit,
        offset: args.offset,
        expert_id: args.expertID,
        pinned_first: args.pinnedFirst,
      },
      signal: args.signal,
    }),
  );
  const sessions = Array.isArray(body?.sessions) ? body.sessions : [];
  return {
    sessions: sessions
      .map(toSummary)
      .filter((s): s is SessionSummary => s !== null),
    total: asNumber(body?.total) ?? sessions.length,
  };
}

export async function getSessionDetail(
  api: BackendClient,
  sessionID: string,
  args: { limit?: number; beforeSequence?: number; signal?: AbortSignal } = {},
): Promise<SessionDetail> {
  const body = asRecord(
    await api.json<unknown>(
      `/api/chat/sessions/${encodeURIComponent(sessionID)}`,
      {
        query: {
          limit: args.limit ?? 50,
          before_sequence: args.beforeSequence,
        },
        signal: args.signal,
      },
    ),
  );
  const active = asRecord(body?.active_stream);
  const turnID = nonEmptyString(active?.turn_id);
  return {
    id: asString(body?.id) ?? sessionID,
    chatStatus: asString(body?.chat_status) ?? "idle",
    messages: Array.isArray(body?.messages) ? body.messages : [],
    activeStream: turnID
      ? {
          turnID,
          startedAt: asString(active?.started_at),
          checkpoint: parseCheckpoint(active?.checkpoint),
        }
      : null,
    hasMore: body?.has_more_messages === true,
    oldestSequence: asNumber(body?.oldest_sequence),
    expertID: nonEmptyString(body?.expert_id),
  };
}

export async function createSession(
  api: BackendClient,
  args: { expertID: string | null },
): Promise<{ id: string; expertID: string | null }> {
  const body = asRecord(
    await api.json<unknown>("/api/chat/sessions", {
      method: "POST",
      json: args.expertID ? { expert_id: args.expertID } : {},
    }),
  );
  const id = nonEmptyString(body?.id);
  if (!id) throw new Error("The server didn't return a conversation.");
  return { id, expertID: nonEmptyString(body?.expert_id) };
}

/** Starts a turn. `messageID` is the per-send idempotency key; retries reuse it. */
export function openTurnStream(
  api: BackendClient,
  sessionID: string,
  args: { message: string; messageID: string; signal: AbortSignal },
): Promise<OpenResponse> {
  return api.open(
    `/api/chat/sessions/${encodeURIComponent(sessionID)}/stream`,
    {
      method: "POST",
      json: {
        message: args.message,
        is_user_message: true,
        message_id: args.messageID,
      },
      headers: { Accept: "text/event-stream" },
      signal: args.signal,
      timeoutMs: 30_000,
    },
  );
}

/** Reads a turn's stored entries after `after`, then follows it while it runs. */
export function resumeTurnStream(
  api: BackendClient,
  sessionID: string,
  args: { turnID: string; after: string; signal: AbortSignal },
): Promise<OpenResponse> {
  return api.open(
    `/api/chat/sessions/${encodeURIComponent(sessionID)}/stream`,
    {
      query: { turn: args.turnID, after: args.after },
      headers: { Accept: "text/event-stream" },
      signal: args.signal,
    },
  );
}

export async function cancelTurn(api: BackendClient, sessionID: string) {
  const body = asRecord(
    await api.json<unknown>(
      `/api/chat/sessions/${encodeURIComponent(sessionID)}/cancel`,
      {
        method: "POST",
        timeoutMs: 15_000,
      },
    ),
  );
  return {
    cancelled: body?.cancelled === true,
    reason: asString(body?.reason),
  };
}

export function parseCheckpoint(value: unknown): TurnCheckpoint | null {
  const record = asRecord(value);
  const entryID = nonEmptyString(record?.entry_id);
  const rows = asNumber(record?.rows);
  const sequence = asNumber(record?.sequence);
  if (!entryID || rows === null || sequence === null) return null;
  return { entryID, rows, sequence };
}

/** User-facing copy for the chat routes' documented refusals. */
export function describeChatError(error: unknown): string {
  if (!(error instanceof HttpError)) {
    return errorMessage(error, "Something went wrong. Try again.");
  }
  switch (error.code) {
    case "chat_transport_selection_required":
      return "Choose which AI connection new chats use in AutoGPT on the web, then try again.";
    case "chat_transport_not_configured":
      return "This server has no AI connection set up for chat.";
    case "advanced_tier_unavailable":
    case "advanced_tier_not_entitled":
      return "That model tier isn't available on this account.";
  }
  if (error.status === 402) {
    return "A subscription is needed to chat on this server. Subscribing in the app isn't available yet.";
  }
  if (error.status === 404) {
    return "This conversation isn't available. It may have been deleted, or the expert was let go.";
  }
  if (error.status === 429) {
    return error.message.startsWith("Request failed")
      ? "You've reached a usage limit. Try again later."
      : error.message;
  }
  if (error.status === 503) {
    const wait = error.retryAfterSeconds;
    return wait
      ? `The chat service is busy. Try again in ${Math.ceil(wait)} seconds.`
      : "The chat service is busy. Try again in a moment.";
  }
  return error.message;
}

function toSummary(raw: unknown): SessionSummary | null {
  const record = asRecord(raw);
  const id = nonEmptyString(record?.id);
  if (!record || !id) return null;
  return {
    id,
    createdAt: asString(record.created_at),
    updatedAt: asString(record.updated_at),
    title: nonEmptyString(record.title),
    chatStatus: asString(record.chat_status) ?? "idle",
    isProcessing: record.is_processing === true,
    isPinned: record.is_pinned === true,
    expertID: nonEmptyString(record.expert_id),
  };
}
