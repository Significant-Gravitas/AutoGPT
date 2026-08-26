import { afterEach, describe, expect, it, vi } from "vitest";
import { updateHistoryBreakdown, useTokenDevtoolStore } from "../store";
import {
  BASE_CONTEXT_ESTIMATE,
  breakdownTotal,
  computeBreakdown,
  displayContext,
  formatTokenCount,
  turnInputTokens,
  type TokenTurn,
} from "../tokenMath";
import {
  createUsageCapturingFetch,
  isCompactionLine,
  parseUsageComment,
} from "../usageTap";

const USAGE_LINE =
  ': usage {"type":"usage","promptTokens":1200,"completionTokens":300,"totalTokens":1500,"cacheReadTokens":40000,"cacheCreationTokens":2000}';

function turn(overrides: Partial<TokenTurn> = {}): TokenTurn {
  return {
    promptTokens: 0,
    completionTokens: 0,
    cacheReadTokens: 0,
    cacheCreationTokens: 0,
    compacted: false,
    at: 1,
    ...overrides,
  };
}

afterEach(() => {
  useTokenDevtoolStore.setState({
    turnsBySession: {},
    breakdownBySession: {},
    liveContextBySession: {},
    compactedBySession: {},
  });
  vi.unstubAllGlobals();
});

function liveContextOf(sessionId: string) {
  const state = useTokenDevtoolStore.getState();
  return {
    context: state.liveContextBySession[sessionId] ?? null,
    compacted: state.compactedBySession[sessionId] ?? false,
  };
}

describe("parseUsageComment", () => {
  it("parses a usage comment into turn usage", () => {
    const usage = parseUsageComment(USAGE_LINE);
    expect(usage).not.toBeNull();
    expect(usage!.promptTokens).toBe(1200);
    expect(usage!.completionTokens).toBe(300);
    expect(usage!.cacheReadTokens).toBe(40000);
    expect(usage!.cacheCreationTokens).toBe(2000);
    expect(turnInputTokens({ ...usage!, compacted: false })).toBe(43200);
  });

  it("ignores data lines and other comments", () => {
    expect(parseUsageComment('data: {"type":"text-delta"}')).toBeNull();
    expect(parseUsageComment(": keepalive")).toBeNull();
    // No closing brace — rejected by the regex before JSON.parse is reached.
    expect(parseUsageComment(": usage {broken")).toBeNull();
  });

  it("ignores a well-shaped comment whose payload is invalid JSON", () => {
    expect(parseUsageComment(': usage {"promptTokens":}')).toBeNull();
  });

  it("does not flag a turn because the assistant quoted the tool name", () => {
    expect(
      isCompactionLine(
        'data: {"type":"text-delta","delta":"I will run context_compaction next"}',
      ),
    ).toBe(false);
  });

  it("flags a real compaction tool call", () => {
    expect(
      isCompactionLine(
        'data: {"type":"tool-input-start","toolCallId":"c1","toolName":"context_compaction"}',
      ),
    ).toBe(true);
  });

  it("coerces missing or negative counts to zero", () => {
    const usage = parseUsageComment(': usage {"promptTokens":-5}');
    expect(usage).not.toBeNull();
    expect(usage!.promptTokens).toBe(0);
    expect(usage!.completionTokens).toBe(0);
  });
});

describe("store.record", () => {
  function record(sessionId: string, turns: TokenTurn[]) {
    turns.forEach((t) => useTokenDevtoolStore.getState().record(sessionId, t));
  }

  it("sums cache writes across turns", () => {
    record("s", [
      turn({ cacheCreationTokens: 30000 }),
      turn({ cacheCreationTokens: 5000 }),
    ]);
    expect(liveContextOf("s").context).toBe(35000);
  });

  it("restarts the sum on a compaction turn", () => {
    record("s", [
      turn({ cacheCreationTokens: 90000 }),
      turn({ cacheCreationTokens: 20000, compacted: true }),
      turn({ cacheCreationTokens: 4000 }),
    ]);
    expect(liveContextOf("s").context).toBe(24000);
  });

  // The per-turn list is capped for display. The estimate must not be, or a
  // long session would silently lose both its post-compaction restart point
  // and the fact that it ever compacted.
  it("keeps the estimate exact after the compaction turn is evicted", () => {
    record("s", [turn({ cacheCreationTokens: 20000, compacted: true })]);
    for (let i = 0; i < 55; i++) {
      record("s", [turn({ cacheCreationTokens: 1000 })]);
    }
    const { context, compacted } = liveContextOf("s");
    expect(useTokenDevtoolStore.getState().turnsBySession["s"]).toHaveLength(
      50,
    );
    expect(
      useTokenDevtoolStore.getState().turnsBySession["s"],
    ).not.toContainEqual(expect.objectContaining({ compacted: true }));
    expect(context).toBe(75000);
    expect(compacted).toBe(true);
  });

  it("tracks sessions independently", () => {
    record("a", [turn({ cacheCreationTokens: 10 })]);
    record("b", [turn({ cacheCreationTokens: 20, compacted: true })]);
    expect(liveContextOf("a")).toEqual({ context: 10, compacted: false });
    expect(liveContextOf("b")).toEqual({ context: 20, compacted: true });
  });
});

describe("displayContext", () => {
  it("shows the seed before any live turns", () => {
    expect(displayContext(null, false, 90000)).toBe(90000);
    expect(displayContext(null, false, undefined)).toBeNull();
  });

  it("keeps the seed until the live estimate exceeds it", () => {
    expect(displayContext(2000, false, 90000)).toBe(90000);
    expect(displayContext(95000, false, 90000)).toBe(95000);
  });

  it("drops the seed once a compaction is observed", () => {
    expect(displayContext(70000, true, 90000)).toBe(70000);
  });

  it("reports nothing when a compacted session has no live turns", () => {
    expect(displayContext(null, true, 90000)).toBeNull();
  });
});

describe("computeBreakdown", () => {
  it("splits history by user text, assistant text, and tool parts", () => {
    const toolPart = { type: "tool-run_agent", output: "y".repeat(100) };
    const breakdown = computeBreakdown([
      { role: "user", parts: [{ type: "text", text: "x".repeat(400) }] },
      { role: "assistant", parts: [{ type: "text", text: "z".repeat(200) }] },
      { role: "assistant", parts: [toolPart] },
    ]);
    expect(breakdown.userTokens).toBe(100);
    expect(breakdown.assistantTokens).toBe(50);
    expect(breakdown.toolTokens).toBe(
      Math.ceil(JSON.stringify(toolPart).length / 4),
    );
    expect(breakdownTotal(breakdown)).toBe(
      BASE_CONTEXT_ESTIMATE + 100 + 50 + breakdown.toolTokens,
    );
  });

  it("counts assistant reasoning as assistant text", () => {
    const breakdown = computeBreakdown([
      {
        role: "assistant",
        parts: [{ type: "reasoning", text: "a".repeat(40) }],
      },
    ]);
    expect(breakdown.assistantTokens).toBe(10);
    expect(breakdown.toolTokens).toBe(0);
  });
});

describe("updateHistoryBreakdown", () => {
  it("overwrites the session breakdown on each call", () => {
    updateHistoryBreakdown("s1", [
      { role: "user", parts: [{ type: "text", text: "hi" }] },
    ]);
    updateHistoryBreakdown("s1", [
      { role: "user", parts: [{ type: "text", text: "x".repeat(80) }] },
    ]);
    expect(
      useTokenDevtoolStore.getState().breakdownBySession["s1"].userTokens,
    ).toBe(20);
  });
});

describe("formatTokenCount", () => {
  it("formats counts across magnitudes", () => {
    expect(formatTokenCount(512)).toBe("512");
    expect(formatTokenCount(1000)).toBe("1k");
    expect(formatTokenCount(45200)).toBe("45.2k");
    // Scales to "100.0" — only the fractional zero may be trimmed.
    expect(formatTokenCount(100_000)).toBe("100k");
    expect(formatTokenCount(1_250_000)).toBe("1.25M");
    expect(formatTokenCount(1_000_000)).toBe("1M");
  });
});

describe("createUsageCapturingFetch", () => {
  const POST = { method: "POST" } as const;

  function sseResponse(chunks: string[]) {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)));
        controller.close();
      },
    });
    return new Response(body, { status: 200 });
  }

  async function drain(response: Response) {
    const reader = response.body!.getReader();
    let text = "";
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      text += decoder.decode(value, { stream: true });
    }
    return text;
  }

  it("records usage comments while passing the stream through untouched", async () => {
    const payload = `data: {"type":"text-delta"}\n\n${USAGE_LINE}\n\n`;
    // Split mid-comment to prove line reassembly across chunks.
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => sseResponse([payload.slice(0, 40), payload.slice(40)])),
    );

    const wrapped = createUsageCapturingFetch("session-1");
    const text = await drain(await wrapped("http://x/stream", POST));

    expect(text).toBe(payload);
    await vi.waitFor(() => {
      const turns = useTokenDevtoolStore.getState().turnsBySession["session-1"];
      expect(turns).toHaveLength(1);
      expect(turns[0].promptTokens).toBe(1200);
      expect(turns[0].compacted).toBe(false);
    });
  });

  it("flags the turn when the stream carried a compaction tool call", async () => {
    const payload =
      'data: {"type":"tool-input-start","toolCallId":"c1","toolName":"context_compaction"}\n\n' +
      `${USAGE_LINE}\n\n`;
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => sseResponse([payload])),
    );

    const wrapped = createUsageCapturingFetch("session-2");
    await drain(await wrapped("http://x/stream", POST));

    await vi.waitFor(() => {
      const turns = useTokenDevtoolStore.getState().turnsBySession["session-2"];
      expect(turns).toHaveLength(1);
      expect(turns[0].compacted).toBe(true);
    });
  });

  it("releases the underlying stream when the consumer aborts", async () => {
    let sourceCancelled = false;
    const controller = new AbortController();
    // Never closes on its own — only cancellation can release it.
    const body = new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(new TextEncoder().encode('data: {"type":"text-delta"}\n\n'));
      },
      cancel() {
        sourceCancelled = true;
      },
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(body, { status: 200 })),
    );

    const wrapped = createUsageCapturingFetch("session-abort");
    const response = await wrapped("http://x/stream", {
      ...POST,
      signal: controller.signal,
    });

    // A tee'd source is only released once BOTH branches let go — and the
    // branch's own cancel() promise does not settle until then, so this is
    // deliberately not awaited yet.
    const consumerCancelled = response.body!.cancel();
    expect(sourceCancelled).toBe(false);

    // Without the abort listener the tap would read on and hold the source.
    controller.abort();
    await consumerCancelled;

    await vi.waitFor(() => expect(sourceCancelled).toBe(true));
  });

  // The AI SDK's reconnect path GETs the active turn replayed from "0-0", so
  // tapping it would record the same `: usage {...}` twice and permanently
  // inflate the running estimate.
  it("leaves the reconnect GET untapped", async () => {
    const payload = `${USAGE_LINE}\n\n`;
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => sseResponse([payload])),
    );

    const wrapped = createUsageCapturingFetch("session-resume");
    const text = await drain(
      await wrapped("http://x/stream", { method: "GET" }),
    );

    expect(text).toBe(payload);
    expect(
      useTokenDevtoolStore.getState().turnsBySession["session-resume"],
    ).toBeUndefined();
  });

  it("returns error responses untouched", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("nope", { status: 500 })),
    );
    const wrapped = createUsageCapturingFetch("session-1");
    const response = await wrapped("http://x/stream", POST);
    expect(response.status).toBe(500);
    expect(await response.text()).toBe("nope");
    expect(
      useTokenDevtoolStore.getState().turnsBySession["session-1"],
    ).toBeUndefined();
  });
});
