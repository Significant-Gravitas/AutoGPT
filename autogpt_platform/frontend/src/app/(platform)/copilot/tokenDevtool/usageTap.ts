import { useTokenDevtoolStore } from "./store";
import type { TokenTurn } from "./tokenMath";

const USAGE_COMMENT = /^:\s*usage\s+(\{.*\})$/;
const COMPACTION_MARKER = '"context_compaction"';

export function parseUsageComment(
  line: string,
): Omit<TokenTurn, "compacted"> | null {
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
    signal?.removeEventListener("abort", cancelTap);
    reader.releaseLock();
  }
}
