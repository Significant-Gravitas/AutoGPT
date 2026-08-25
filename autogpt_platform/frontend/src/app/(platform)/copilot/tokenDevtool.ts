import { environment } from "@/services/environment";
import { create } from "zustand";

/** One turn's token usage, captured from the backend's `: usage {...}` SSE
 *  comment. The AI SDK parser drops comment lines, so the transport's fetch
 *  taps the raw stream before parsing (see createUsageCapturingFetch).
 *
 *  The backend accumulates usage across EVERY API step of the turn (each
 *  tool round-trip re-reads the context), so prompt/cache sums overcount
 *  the live context — use estimateContext() for that instead. */
export interface TokenTurn {
  promptTokens: number;
  completionTokens: number;
  cacheReadTokens: number;
  cacheCreationTokens: number;
  /** The stream carried a `context_compaction` tool call this turn — the
   *  backend summarized the transcript before/while answering. */
  compacted: boolean;
  at: number;
}

const KEPT_TURNS = 50;

/** Claude-family window the copilot engines run against. Devtool-only
 *  constants — the real values live in backend config (see
 *  claude_agent_autocompact_pct_override, default 50% of the window). */
export const MODEL_CONTEXT_WINDOW = 200_000;
export const AUTOCOMPACT_TOKENS = 100_000;

/** System prompt + tool definitions the SDK always carries — backend config
 *  notes the post-compaction floor is ≈65-110K, so 65K is the low bound. */
export const BASE_CONTEXT_ESTIMATE = 65_000;

/** Total input across all of the turn's API steps (uncached + cache reads +
 *  cache writes). A cost-side number, not the live context size. */
export function turnInputTokens(turn: TokenTurn): number {
  return turn.promptTokens + turn.cacheReadTokens + turn.cacheCreationTokens;
}

/** Best-effort live context estimate: cache writes are the tokens newly
 *  persisted into the prompt cache, so summing them since the last
 *  compaction approximates the transcript the model currently sees. A
 *  compaction turn restarts the sum (the compacted context is re-written
 *  to cache in that same turn). */
export function estimateContext(turns: TokenTurn[]): number {
  let total = 0;
  for (const turn of turns) {
    total = turn.compacted
      ? turn.cacheCreationTokens
      : total + turn.cacheCreationTokens;
  }
  return total;
}

export function formatTokenCount(count: number): string {
  if (count < 1000) return String(count);
  if (count < 1_000_000) return `${trimZeros((count / 1000).toFixed(1))}k`;
  return `${trimZeros((count / 1_000_000).toFixed(2))}M`;
}

function trimZeros(value: string): string {
  return value.replace(/\.?0+$/, "");
}

/** Dev-only: local/dev environments, and NEXT_PUBLIC_TOKEN_DEVTOOL can turn
 *  it off explicitly (unset = on). */
export function isTokenDevtoolEnabled(): boolean {
  if (process.env.NEXT_PUBLIC_TOKEN_DEVTOOL === "false") return false;
  return (
    environment.isDevelopmentBuild() ||
    environment.isLocal() ||
    environment.isDev()
  );
}

/** Char-based (~4 chars/token) split of the loaded conversation history.
 *  System prompt, tool definitions, and injected context blocks (skills,
 *  memory, team roster) never reach the frontend — they live in the fixed
 *  BASE_CONTEXT_ESTIMATE floor instead. */
export interface ContextBreakdown {
  userTokens: number;
  assistantTokens: number;
  toolTokens: number;
}

export function breakdownTotal(breakdown: ContextBreakdown): number {
  return (
    BASE_CONTEXT_ESTIMATE +
    breakdown.userTokens +
    breakdown.assistantTokens +
    breakdown.toolTokens
  );
}

interface TokenDevtoolState {
  turnsBySession: Record<string, TokenTurn[]>;
  breakdownBySession: Record<string, ContextBreakdown>;
  record: (sessionId: string, turn: TokenTurn) => void;
  setBreakdown: (sessionId: string, breakdown: ContextBreakdown) => void;
}

export const useTokenDevtoolStore = create<TokenDevtoolState>((set) => ({
  turnsBySession: {},
  breakdownBySession: {},
  record(sessionId, turn) {
    set((state) => ({
      turnsBySession: {
        ...state.turnsBySession,
        [sessionId]: [
          ...(state.turnsBySession[sessionId] ?? []),
          turn,
        ].slice(-KEPT_TURNS),
      },
    }));
  },
  setBreakdown(sessionId, breakdown) {
    set((state) => ({
      breakdownBySession: {
        ...state.breakdownBySession,
        [sessionId]: breakdown,
      },
    }));
  },
}));

/** Displayed context: prefer the live cache-write estimate, but until it
 *  can exceed the history estimate (a fresh page load starts the live sum
 *  at zero), keep showing the history number. Once a compaction is observed
 *  the history estimate is stale by definition and the live one wins. */
export function displayContext(
  turns: TokenTurn[] | undefined,
  historyEstimate: number | undefined,
): number | null {
  const live = turns?.length ? estimateContext(turns) : null;
  if (turns?.some((turn) => turn.compacted)) return live;
  if (live === null && historyEstimate === undefined) return null;
  return Math.max(live ?? 0, historyEstimate ?? 0);
}

interface MessageLike {
  role?: string;
  parts?: Array<{ type?: string; text?: string }>;
}

export function computeBreakdown(messages: unknown[]): ContextBreakdown {
  let user = 0;
  let assistant = 0;
  let tools = 0;
  for (const message of messages as MessageLike[]) {
    for (const part of message?.parts ?? []) {
      const chars = partChars(part);
      if (part?.type === "text" || part?.type === "reasoning") {
        if (message.role === "user") user += chars;
        else assistant += chars;
      } else {
        tools += chars;
      }
    }
  }
  return {
    userTokens: Math.ceil(user / 4),
    assistantTokens: Math.ceil(assistant / 4),
    toolTokens: Math.ceil(tools / 4),
  };
}

function partChars(part: unknown): number {
  const text = (part as { text?: unknown })?.text;
  if (typeof text === "string") return text.length;
  try {
    return JSON.stringify(part)?.length ?? 0;
  } catch {
    return 0; // Circular/unserializable part — skip it, this is an estimate.
  }
}

/** Recompute the session's history breakdown from the loaded messages. */
export function updateHistoryBreakdown(sessionId: string, messages: unknown[]) {
  useTokenDevtoolStore
    .getState()
    .setBreakdown(sessionId, computeBreakdown(messages));
}

const USAGE_COMMENT = /^:\s*usage\s+(\{.*\})$/;
const COMPACTION_MARKER = '"context_compaction"';

export function parseUsageComment(line: string): Omit<
  TokenTurn,
  "compacted"
> | null {
  const match = USAGE_COMMENT.exec(line.trim());
  if (!match) return null;
  try {
    const raw = JSON.parse(match[1]) as Record<string, unknown>;
    return {
      promptTokens: toCount(raw.promptTokens),
      completionTokens: toCount(raw.completionTokens),
      cacheReadTokens: toCount(raw.cacheReadTokens),
      cacheCreationTokens: toCount(raw.cacheCreationTokens),
      at: Date.now(),
    };
  } catch {
    return null;
  }
}

function toCount(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : 0;
}

/** Wraps fetch so the copilot SSE stream is teed: the AI SDK consumes one
 *  branch untouched while the other is scanned for usage comments and
 *  compaction tool calls. The tap must never break the chat — every failure
 *  path degrades to "no data". */
export function createUsageCapturingFetch(sessionId: string): typeof fetch {
  return async (input, init) => {
    const response = await fetch(input, init);
    if (!response.ok || !response.body) return response;
    const [main, tap] = response.body.tee();
    void scanForUsage(tap, sessionId);
    return new Response(main, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  };
}

async function scanForUsage(
  stream: ReadableStream<Uint8Array>,
  sessionId: string,
): Promise<void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  // Usage arrives at turn end, after any compaction tool events — so a
  // marker seen earlier in the stream belongs to the next recorded turn.
  let sawCompaction = false;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (line.includes(COMPACTION_MARKER)) sawCompaction = true;
        const usage = parseUsageComment(line);
        if (usage) {
          useTokenDevtoolStore
            .getState()
            .record(sessionId, { ...usage, compacted: sawCompaction });
          sawCompaction = false;
        }
      }
    }
  } catch {
    // Devtool tap only — swallow so it can never surface as a chat error.
  } finally {
    reader.releaseLock();
  }
}
