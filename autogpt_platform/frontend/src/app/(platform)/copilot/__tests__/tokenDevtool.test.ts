import { afterEach, describe, expect, it, vi } from "vitest";
import {
  BASE_CONTEXT_ESTIMATE,
  breakdownTotal,
  computeBreakdown,
  createUsageCapturingFetch,
  displayContext,
  estimateContext,
  formatTokenCount,
  parseUsageComment,
  turnInputTokens,
  updateHistoryBreakdown,
  useTokenDevtoolStore,
  type TokenTurn,
} from "../tokenDevtool";

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
  useTokenDevtoolStore.setState({ turnsBySession: {}, breakdownBySession: {} });
  vi.unstubAllGlobals();
});

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

  it("ignores data lines, other comments, and malformed JSON", () => {
    expect(parseUsageComment('data: {"type":"text-delta"}')).toBeNull();
    expect(parseUsageComment(": keepalive")).toBeNull();
    expect(parseUsageComment(": usage {broken")).toBeNull();
  });

  it("coerces missing or negative counts to zero", () => {
    const usage = parseUsageComment(': usage {"promptTokens":-5}');
    expect(usage).not.toBeNull();
    expect(usage!.promptTokens).toBe(0);
    expect(usage!.completionTokens).toBe(0);
  });
});

describe("estimateContext", () => {
  it("sums cache writes across turns", () => {
    const turns = [
      turn({ cacheCreationTokens: 30000 }),
      turn({ cacheCreationTokens: 5000 }),
    ];
    expect(estimateContext(turns)).toBe(35000);
  });

  it("restarts the sum on a compaction turn", () => {
    const turns = [
      turn({ cacheCreationTokens: 90000 }),
      turn({ cacheCreationTokens: 20000, compacted: true }),
      turn({ cacheCreationTokens: 4000 }),
    ];
    expect(estimateContext(turns)).toBe(24000);
  });
});

describe("displayContext", () => {
  it("shows the seed before any live turns", () => {
    expect(displayContext(undefined, 90000)).toBe(90000);
    expect(displayContext([], 90000)).toBe(90000);
    expect(displayContext(undefined, undefined)).toBeNull();
  });

  it("keeps the seed until the live estimate exceeds it", () => {
    const turns = [turn({ cacheCreationTokens: 2000 })];
    expect(displayContext(turns, 90000)).toBe(90000);
    const rebuilt = [turn({ cacheCreationTokens: 95000 })];
    expect(displayContext(rebuilt, 90000)).toBe(95000);
  });

  it("drops the seed once a compaction is observed", () => {
    const turns = [turn({ cacheCreationTokens: 70000, compacted: true })];
    expect(displayContext(turns, 90000)).toBe(70000);
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
    expect(formatTokenCount(1_250_000)).toBe("1.25M");
    expect(formatTokenCount(1_000_000)).toBe("1M");
  });
});

describe("createUsageCapturingFetch", () => {
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
    const text = await drain(await wrapped("http://x/stream"));

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
    await drain(await wrapped("http://x/stream"));

    await vi.waitFor(() => {
      const turns = useTokenDevtoolStore.getState().turnsBySession["session-2"];
      expect(turns).toHaveLength(1);
      expect(turns[0].compacted).toBe(true);
    });
  });

  it("returns error responses untouched", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("nope", { status: 500 })),
    );
    const wrapped = createUsageCapturingFetch("session-1");
    const response = await wrapped("http://x/stream");
    expect(response.status).toBe(500);
    expect(await response.text()).toBe("nope");
    expect(
      useTokenDevtoolStore.getState().turnsBySession["session-1"],
    ).toBeUndefined();
  });
});
