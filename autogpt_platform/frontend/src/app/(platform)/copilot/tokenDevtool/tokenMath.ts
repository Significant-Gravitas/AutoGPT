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

export function formatTokenCount(count: number): string {
  if (count < 1000) return String(count);
  if (count < 1_000_000) return `${trimZeros((count / 1000).toFixed(1))}k`;
  return `${trimZeros((count / 1_000_000).toFixed(2))}M`;
}

/** Trailing zeros are only noise in the fractional part — "1.0" -> "1". An
 *  integer string is returned untouched, so "100" does not become "1". */
function trimZeros(value: string): string {
  return value.includes(".") ? value.replace(/\.?0+$/, "") : value;
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

/** Displayed context: prefer the live cache-write estimate, but until it
 *  can exceed the history estimate (a fresh page load starts the live sum
 *  at zero), keep showing the history number. Once a compaction is observed
 *  the history estimate is stale by definition and the live one wins: the
 *  compaction turn re-writes the whole summarized context to cache, so that
 *  turn's write IS the new context — clamping it to a floor would over-report
 *  a genuinely small post-compaction context. */
export function displayContext(
  liveContext: number | null,
  compacted: boolean,
  historyEstimate: number | undefined,
): number | null {
  if (compacted) return liveContext;
  if (liveContext === null && historyEstimate === undefined) return null;
  return Math.max(liveContext ?? 0, historyEstimate ?? 0);
}

export interface MessageLike {
  role?: string;
  parts?: Array<{ type?: string; text?: string }>;
}

export function computeBreakdown(
  messages: readonly MessageLike[],
): ContextBreakdown {
  let user = 0;
  let assistant = 0;
  let tools = 0;
  for (const message of messages) {
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

/** Tool parts are the big ones and computeBreakdown re-walks the whole loaded
 *  history on every recompute, so serializing them each time is O(history) per
 *  call. AI SDK part objects are stable references across re-renders, so
 *  caching per object collapses that to O(new parts). */
const partCharCache = new WeakMap<object, number>();

function partChars(part: unknown): number {
  const text = (part as { text?: unknown })?.text;
  if (typeof text === "string") return text.length;
  if (part === null || typeof part !== "object") return 0;
  const cached = partCharCache.get(part);
  if (cached !== undefined) return cached;
  let chars = 0;
  try {
    chars = JSON.stringify(part)?.length ?? 0;
  } catch {
    chars = 0; // Circular/unserializable part — skip it, this is an estimate.
  }
  partCharCache.set(part, chars);
  return chars;
}
