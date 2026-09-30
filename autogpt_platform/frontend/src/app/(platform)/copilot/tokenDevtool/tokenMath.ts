/** One turn's token usage, captured from the backend's `: usage {...}` SSE
 *  comment. The AI SDK parser drops comment lines, so the transport's fetch
 *  taps the raw stream before parsing (see createUsageCapturingFetch).
 *
 *  The backend accumulates usage across EVERY API step of the turn (each
 *  tool round-trip re-reads the context), so prompt/cache sums overcount
 *  the live context — the store's running cache-write sum tracks that. */
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

/** Chars per token for the char-based history estimate. */
const CHARS_PER_TOKEN = 4;

export interface MessageLike {
  role?: string;
  /** `text` is widened to unknown because parts arrive from the AI SDK: tool
   *  parts carry arbitrary payloads and are not guaranteed to hold a string
   *  here, which is what partChars() runtime-checks. */
  parts?: Array<{ type?: string; text?: unknown; state?: string }>;
}

/** Guards the breakdown recompute, which walks the whole loaded history and
 *  so must not run per stream delta. Carries:
 *  - the session, so switching to a thread with an identical message count
 *    still recomputes;
 *  - the last message's part count, because the SDK appends a turn's tool
 *    calls and results into the existing assistant message, not a new one;
 *  - whether the turn has settled, because assistant text grows in place
 *    without adding a part — deltas are deliberately not tracked, so this is
 *    what forces exactly one recompute when the turn finishes. */
export function breakdownCacheKey(
  sessionId: string,
  messages: readonly MessageLike[],
  isStreaming: boolean,
): string {
  const parts = messages.at(-1)?.parts?.length ?? 0;
  return `${sessionId}:${messages.length}:${parts}:${isStreaming ? "live" : "settled"}`;
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
    userTokens: Math.ceil(user / CHARS_PER_TOKEN),
    assistantTokens: Math.ceil(assistant / CHARS_PER_TOKEN),
    toolTokens: Math.ceil(tools / CHARS_PER_TOKEN),
  };
}

/** Tool parts are the big ones and computeBreakdown re-walks the whole loaded
 *  history on every recompute, so serializing them each time is O(history) per
 *  call. AI SDK part objects are stable references across re-renders, so
 *  caching per object collapses that to O(new parts). */
const partCharCache = new WeakMap<object, number>();

function partChars(part: {
  type?: string;
  text?: unknown;
  state?: string;
}): number {
  if (typeof part?.text === "string") return part.text.length;
  if (part === null || typeof part !== "object") return 0;
  // A tool part still filling in its input/output is mutated in place, so
  // caching it would pin a count that is about to change.
  const cacheable =
    part.state === undefined || part.state.endsWith("available");
  const cached = cacheable ? partCharCache.get(part) : undefined;
  if (cached !== undefined) return cached;
  let chars = 0;
  try {
    chars = JSON.stringify(part)?.length ?? 0;
  } catch {
    chars = 0; // Circular/unserializable part — skip it, this is an estimate.
  }
  if (cacheable) partCharCache.set(part, chars);
  return chars;
}
