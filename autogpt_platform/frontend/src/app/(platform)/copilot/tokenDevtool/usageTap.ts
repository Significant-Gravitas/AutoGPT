import { useTokenDevtoolStore } from "./store";
import type { TokenTurn } from "./tokenMath";

const USAGE_COMMENT = /^:\s*usage\s+(\{.*\})$/;
const COMPACTION_TOOL = "context_compaction";
const COLON = 58;
const MAX_BUFFERED_LINE = 1_000_000;

export function parseUsageComment(
  line: string,
): Omit<TokenTurn, "compacted" | "at"> | null {
  const match = USAGE_COMMENT.exec(line.trim());
  if (!match) return null;
  try {
    const raw = JSON.parse(match[1]) as Record<string, unknown>;
    return {
      promptTokens: toCount(raw.promptTokens),
      completionTokens: toCount(raw.completionTokens),
      cacheReadTokens: toCount(raw.cacheReadTokens),
      cacheCreationTokens: toCount(raw.cacheCreationTokens),
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

/** True only for a `data:` chunk that actually calls the compaction tool.
 *  Substring-matching the raw line would also fire on an assistant reply that
 *  merely quotes the tool name, which would wrongly invalidate the session's
 *  history seed for the rest of the thread. */
export function isCompactionLine(line: string): boolean {
  if (!line.includes(COMPACTION_TOOL)) return false;
  const payload = line.slice(line.indexOf(":") + 1);
  try {
    const chunk = JSON.parse(payload) as { toolName?: unknown };
    return chunk?.toolName === COMPACTION_TOOL;
  } catch {
    return false;
  }
}

/** Wraps fetch so the copilot SSE stream is teed: the AI SDK consumes one
 *  branch untouched while the other is scanned for usage comments and
 *  compaction tool calls. The tap must never break the chat — every failure
 *  path degrades to "no data". */
export function createUsageCapturingFetch(sessionId: string): typeof fetch {
  return async (input, init) => {
    const response = await fetch(input, init);
    // Only tap the POST that streams a fresh turn. The AI SDK's reconnect
    // path re-GETs the active turn from "0-0", replaying the same
    // `: usage {...}` comment — recording it again would permanently inflate
    // the running context estimate.
    if (init?.method?.toUpperCase() !== "POST") return response;
    if (!response.ok || !response.body) return response;
    const [main, tap] = response.body.tee();
    void scanForUsage(tap, sessionId, init?.signal);
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
  signal?: AbortSignal | null,
): Promise<void> {
  const reader = stream.getReader();
  // Both tee branches share one source, so a tap that kept reading after the
  // consumer aborted would hold the connection open. Cancel through the
  // reader — the branch is locked, so tap.cancel() would throw.
  const cancelTap = () => void reader.cancel().catch(() => {});
  signal?.addEventListener("abort", cancelTap);
  if (signal?.aborted) cancelTap();
  const decoder = new TextDecoder();
  let buffer = "";
  // Usage arrives at turn end, after any compaction tool events — so a
  // marker seen earlier in the stream belongs to the next recorded turn.
  let sawCompaction = false;
  const scan = (lines: string[]) => {
    for (const line of lines) {
      // Thousands of lines per turn on the app's most latency-sensitive
      // path: reject the `data:` majority on one char before doing any
      // trimming, regex, or JSON work.
      if (line.charCodeAt(0) === COLON) {
        const usage = parseUsageComment(line);
        if (!usage) continue;
        useTokenDevtoolStore.getState().record(sessionId, {
          ...usage,
          compacted: sawCompaction,
          at: Date.now(),
        });
        sawCompaction = false;
      } else if (isCompactionLine(line)) {
        sawCompaction = true;
      }
    }
  };
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // SSE always delimits with newlines, so this only trips on a malformed
      // upstream. Dropping the partial line is fine — this is an estimate.
      if (buffer.length > MAX_BUFFERED_LINE) buffer = "";
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      scan(lines);
    }
    // A stream that ends without a trailing newline leaves the last line —
    // which is where the usage comment lives — sitting in the buffer.
    if (buffer) scan([buffer]);
  } catch {
    // Devtool tap only — swallow so it can never surface as a chat error.
  } finally {
    signal?.removeEventListener("abort", cancelTap);
    // Cancel, not just releaseLock: tee() buffers every remaining chunk for a
    // branch nobody reads, so bailing out mid-stream without cancelling would
    // queue the rest of the turn (tool results run to megabytes) for the life
    // of the request.
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
