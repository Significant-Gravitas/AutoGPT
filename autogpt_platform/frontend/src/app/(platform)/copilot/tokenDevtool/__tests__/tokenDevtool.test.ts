import { afterEach, describe, expect, it, vi } from "vitest";
import { updateHistoryBreakdown, useTokenDevtoolStore } from "../store";
import {
  BASE_CONTEXT_ESTIMATE,
  breakdownCacheKey,
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
    sessionOrder: [],
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
    expect(turnInputTokens({ ...usage!, compacted: false, at: 1 })).toBe(43200);
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

  // Turns are capped per session; the session keys need a ceiling too, or a
  // long-lived tab that browses many threads grows monotonically.
  it("evicts the oldest sessions past the retention cap", () => {
    for (let i = 0; i < 25; i++) {
      record(`s${i}`, [turn({ cacheCreationTokens: 1 })]);
    }
    const tracked = Object.keys(useTokenDevtoolStore.getState().turnsBySession);
    expect(tracked).toHaveLength(20);
    expect(tracked).toContain("s24");
    expect(tracked).not.toContain("s0");
  });

  // Retention is driven by one shared LRU, so a session evicted from the turn
  // maps cannot linger in the breakdown map.
  it("prunes every map to the same set of sessions", () => {
    for (let i = 0; i < 25; i++) {
      useTokenDevtoolStore.getState().setBreakdown(`s${i}`, {
        userTokens: 1,
        assistantTokens: 0,
        toolTokens: 0,
      });
      record(`s${i}`, [turn({ cacheCreationTokens: 1 })]);
    }
    const state = useTokenDevtoolStore.getState();
    const keys = Object.keys(state.turnsBySession).sort();
    expect(keys).toHaveLength(20);
    expect(Object.keys(state.breakdownBySession).sort()).toEqual(keys);
    expect(Object.keys(state.liveContextBySession).sort()).toEqual(keys);
    expect(Object.keys(state.compactedBySession).sort()).toEqual(keys);
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

describe("breakdownCacheKey", () => {
  const messages = [{ role: "user", parts: [{ type: "text", text: "hi" }] }];

  it("changes when the session changes at an identical message count", () => {
    expect(breakdownCacheKey("a", messages, false)).not.toBe(
      breakdownCacheKey("b", messages, false),
    );
  });

  it("changes when the last message grows parts in place", () => {
    const before = [{ role: "assistant", parts: [{ type: "text" }] }];
    const after = [
      { role: "assistant", parts: [{ type: "text" }, { type: "tool-x" }] },
    ];
    expect(breakdownCacheKey("a", before, false)).not.toBe(
      breakdownCacheKey("a", after, false),
    );
  });

  // Assistant text grows in place, so counts alone never change — settling
  // the turn is what forces the final recompute.
  it("changes when the turn settles at unchanged counts", () => {
    const streaming = [
      { role: "assistant", parts: [{ type: "text", text: "a" }] },
    ];
    const settled = [
      {
        role: "assistant",
        parts: [{ type: "text", text: "a much longer reply" }],
      },
    ];
    expect(breakdownCacheKey("a", streaming, true)).not.toBe(
      breakdownCacheKey("a", settled, false),
    );
  });

  it("is stable for the same session and the same parts", () => {
    expect(breakdownCacheKey("a", messages, false)).toBe(
      breakdownCacheKey("a", messages, false),
    );
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

  it("skips a circular tool part instead of throwing", () => {
    const part: Record<string, unknown> = { type: "tool-x" };
    part.self = part;
    expect(() =>
      computeBreakdown([{ role: "assistant", parts: [part] }]),
    ).not.toThrow();
    expect(
      computeBreakdown([{ role: "assistant", parts: [part] }]).toolTokens,
    ).toBe(0);
  });

  // The per-part memo must not pin a count for a tool part the SDK is still
  // filling in, or the tool row would freeze at its input-streaming size.
  it("recounts a tool part that is still streaming", () => {
    const part: Record<string, unknown> = {
      type: "tool-x",
      state: "input-streaming",
    };
    const first = computeBreakdown([
      { role: "assistant", parts: [part] },
    ]).toolTokens;
    part.output = "y".repeat(400);
    const second = computeBreakdown([
      { role: "assistant", parts: [part] },
    ]).toolTokens;
    expect(second).toBeGreaterThan(first);
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

  it("records a usage comment that arrives without a trailing newline", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => sseResponse([USAGE_LINE])),
    );

    const wrapped = createUsageCapturingFetch("session-flush");
    await drain(await wrapped("http://x/stream", POST));

    await vi.waitFor(() => {
      const turns =
        useTokenDevtoolStore.getState().turnsBySession["session-flush"];
      expect(turns).toHaveLength(1);
      expect(turns[0].promptTokens).toBe(1200);
    });
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
