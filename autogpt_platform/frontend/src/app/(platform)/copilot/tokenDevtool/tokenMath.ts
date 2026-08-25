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
