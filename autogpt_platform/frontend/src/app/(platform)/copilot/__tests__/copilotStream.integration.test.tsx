import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  copilotResumeHandler,
  copilotStreamErrorHandler,
  copilotStreamHandler,
} from "@/tests/integrations/copilot-sse";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import type { UIMessage, UIMessageChunk } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const TEST_BACKEND_BASE_URL = "http://localhost:18006";

vi.mock("@/services/environment", async (importActual) => {
  const actual = await importActual<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: {
      ...actual.environment,
      getAGPTServerBaseUrl: () => TEST_BACKEND_BASE_URL,
    },
  };
});

vi.mock("../helpers", async (importActual) => {
  const actual = await importActual<typeof import("../helpers")>();
  return {
    ...actual,
    getCopilotAuthHeaders: async () => ({ "x-test-auth": "yes" }),
  };
});

import { resetCopilotChatRegistry } from "../copilotChatRegistry";
import { useCopilotStream } from "../useCopilotStream";

const TEST_SESSION_ID = "test-session-stream-1";

interface HarnessProps {
  hasActiveStream?: boolean;
  hydratedMessages?: UIMessage[];
}

function StreamHarness({
  hasActiveStream = false,
  hydratedMessages = [],
}: HarnessProps) {
  const refetchSession = async () => ({ data: undefined });
  const result = useCopilotStream({
    sessionId: TEST_SESSION_ID,
    hydratedMessages,
    hasActiveStream,
    refetchSession,
    copilotMode: undefined,
    copilotModel: undefined,
  });

  const assistantMessages = result.messages.filter(
    (m) => m.role === "assistant",
  );
  const userMessages = result.messages.filter((m) => m.role === "user");

  const allParts = assistantMessages.flatMap((m) => m.parts);

  const assistantText = allParts
    .filter((p): p is Extract<typeof p, { type: "text" }> => p.type === "text")
    .map((p) => p.text)
    .join("");

  const assistantReasoning = allParts
    .filter(
      (p): p is Extract<typeof p, { type: "reasoning" }> =>
        p.type === "reasoning",
    )
    .map((p) => p.text)
    .join("");

  interface ToolPartSnapshot {
    type: string;
    state?: string;
    toolCallId?: string;
    input?: unknown;
    output?: unknown;
  }
  const toolParts: ToolPartSnapshot[] = allParts
    .filter(
      (p) =>
        typeof p.type === "string" &&
        (p.type.startsWith("tool-") || p.type === "dynamic-tool"),
    )
    .map((p) => {
      const tp = p as ToolPartSnapshot;
      return {
        type: tp.type,
        state: tp.state,
        toolCallId: tp.toolCallId,
        input: tp.input,
        output: tp.output,
      };
    });

  const stepStartCount = allParts.filter((p) => p.type === "step-start").length;

  return (
    <div>
      <div data-testid="status">{result.status}</div>
      <div data-testid="error">{result.error?.message ?? ""}</div>
      <div data-testid="is-reconnecting">{String(result.isReconnecting)}</div>
      <div data-testid="is-user-stopping">{String(result.isUserStopping)}</div>
      <div data-testid="rate-limit">{result.rateLimitMessage ?? ""}</div>
      <div data-testid="assistant-text">{assistantText}</div>
      <div data-testid="assistant-reasoning">{assistantReasoning}</div>
      <div data-testid="tool-parts">{JSON.stringify(toolParts)}</div>
      <div data-testid="assistant-message-count">
        {assistantMessages.length}
      </div>
      <div data-testid="user-message-count">{userMessages.length}</div>
      <div data-testid="step-start-count">{stepStartCount}</div>
      <button
        data-testid="send"
        onClick={() => result.sendMessage({ text: "hello copilot" })}
      >
        send
      </button>
      <button data-testid="stop" onClick={() => result.stop()}>
        stop
      </button>
    </div>
  );
}

async function clickSend() {
  const user = (await import("@testing-library/user-event")).default.setup();
  await user.click(screen.getByTestId("send"));
}

async function clickStop() {
  const user = (await import("@testing-library/user-event")).default.setup();
  await user.click(screen.getByTestId("stop"));
}

function expectTextContent(testId: string, expected: string) {
  expect(screen.getByTestId(testId).textContent).toBe(expected);
}

describe("CoPilot streaming (SSE) — content rendering", () => {
  beforeEach(() => {
    resetCopilotChatRegistry();
  });

  afterEach(() => {
    resetCopilotChatRegistry();
    vi.useRealTimers();
  });

  it("renders assistant text from a single text-delta frame", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: assistantTextChunks("Hello from the copilot."),
      }),
    );

    render(<StreamHarness />);
    await clickSend();

    await waitFor(
      () => expectTextContent("assistant-text", "Hello from the copilot."),
      { timeout: 5000 },
    );
    await waitFor(() => expectTextContent("status", "ready"));
  });

  it("concatenates multiple text-delta frames into the final assistant message", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: [
          { type: "start", messageId: "msg-1" },
          { type: "start-step" },
          { type: "text-start", id: "t1" },
          { type: "text-delta", id: "t1", delta: "Hello " },
          { type: "text-delta", id: "t1", delta: "from " },
          { type: "text-delta", id: "t1", delta: "the copilot." },
          { type: "text-end", id: "t1" },
          { type: "finish-step" },
          { type: "finish" },
        ],
      }),
    );

    render(<StreamHarness />);
    await clickSend();

    await waitFor(
      () => expectTextContent("assistant-text", "Hello from the copilot."),
      { timeout: 5000 },
    );
  });

  it("renders reasoning parts from reasoning-* chunks", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: [
          { type: "start", messageId: "msg-1" },
          { type: "start-step" },
          { type: "reasoning-start", id: "r1" },
          { type: "reasoning-delta", id: "r1", delta: "Thinking " },
          { type: "reasoning-delta", id: "r1", delta: "step by step…" },
          { type: "reasoning-end", id: "r1" },
          { type: "text-start", id: "t1" },
          { type: "text-delta", id: "t1", delta: "Final answer." },
          { type: "text-end", id: "t1" },
          { type: "finish-step" },
          { type: "finish" },
        ],
      }),
    );

    render(<StreamHarness />);
    await clickSend();

    await waitFor(
      () => expectTextContent("assistant-reasoning", "Thinking step by step…"),
      { timeout: 5000 },
    );
    expectTextContent("assistant-text", "Final answer.");
  });

  it("renders dynamic tool input + output as a tool part", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: [
          { type: "start", messageId: "msg-1" },
          { type: "start-step" },
          {
            type: "tool-input-start",
            toolCallId: "call-1",
            toolName: "search",
            dynamic: true,
          },
          {
            type: "tool-input-available",
            toolCallId: "call-1",
            toolName: "search",
            input: { query: "weather" },
            dynamic: true,
          },
          {
            type: "tool-output-available",
            toolCallId: "call-1",
            output: { result: "sunny" },
            dynamic: true,
          },
          { type: "finish-step" },
          { type: "finish" },
        ],
      }),
    );

    render(<StreamHarness />);
    await clickSend();

    await waitFor(
      () => {
        const parsed = JSON.parse(
          screen.getByTestId("tool-parts").textContent ?? "[]",
        );
        expect(parsed).toEqual([
          expect.objectContaining({
            type: "dynamic-tool",
            state: "output-available",
            toolCallId: "call-1",
            input: { query: "weather" },
            output: { result: "sunny" },
          }),
        ]);
      },
      { timeout: 5000 },
    );
    await waitFor(() => expectTextContent("status", "ready"));
  });

  it("handles multi-step turns with two start-step / finish-step pairs", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: [
          { type: "start", messageId: "msg-1" },
          { type: "start-step" },
          { type: "text-start", id: "t1" },
          { type: "text-delta", id: "t1", delta: "First " },
          { type: "text-end", id: "t1" },
          { type: "finish-step" },
          { type: "start-step" },
          { type: "text-start", id: "t2" },
          { type: "text-delta", id: "t2", delta: "Second" },
          { type: "text-end", id: "t2" },
          { type: "finish-step" },
          { type: "finish" },
        ],
      }),
    );

    render(<StreamHarness />);
    await clickSend();

    await waitFor(() => expectTextContent("assistant-text", "First Second"), {
      timeout: 5000,
    });
    // Each `start-step` becomes a `step-start` UI part on the assistant message.
    expect(
      Number(screen.getByTestId("step-start-count").textContent),
    ).toBeGreaterThanOrEqual(2);
  });

  it("reaches ready status on an empty completion (no text/tool content)", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: [
          { type: "start", messageId: "msg-1" },
          { type: "start-step" },
          { type: "finish-step" },
          { type: "finish" },
        ],
      }),
    );

    render(<StreamHarness />);
    await clickSend();

    await waitFor(() => expectTextContent("status", "ready"), {
      timeout: 5000,
    });
    expectTextContent("assistant-text", "");
    expectTextContent("error", "");
  });
});

describe("CoPilot streaming (SSE) — status lifecycle", () => {
  beforeEach(() => {
    resetCopilotChatRegistry();
  });

  afterEach(() => {
    resetCopilotChatRegistry();
    vi.useRealTimers();
  });

  it("transitions status from ready → streaming → ready across a turn", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: assistantTextChunks("Hi."),
        delayMsBetweenChunks: 30,
      }),
    );

    render(<StreamHarness />);
    expectTextContent("status", "ready");

    await clickSend();

    await waitFor(() => {
      expect(["submitted", "streaming"]).toContain(
        screen.getByTestId("status").textContent,
      );
    });
    await waitFor(() => expectTextContent("status", "ready"), {
      timeout: 5000,
    });
    expectTextContent("assistant-text", "Hi.");
  });
});

describe("CoPilot streaming (SSE) — error paths", () => {
  beforeEach(() => {
    resetCopilotChatRegistry();
  });

  afterEach(() => {
    resetCopilotChatRegistry();
    vi.useRealTimers();
  });

  it("propagates an SSE error chunk into the error state", async () => {
    const chunks: UIMessageChunk[] = [
      { type: "start", messageId: "msg-1" },
      { type: "start-step" },
      { type: "error", errorText: "Backend went sideways." },
    ];
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks,
      }),
    );

    render(<StreamHarness />);
    await clickSend();

    await waitFor(
      () =>
        expect(screen.getByTestId("error").textContent).toBe(
          "Backend went sideways.",
        ),
      { timeout: 5000 },
    );
  });

  it("sets rateLimitMessage and drops optimistic user bubble on HTTP 429 'usage limit'", async () => {
    server.use(
      copilotStreamErrorHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        status: 429,
        body: { detail: "You've reached your usage limit. Try again later." },
      }),
    );

    render(<StreamHarness />);
    await clickSend();

    await waitFor(
      () =>
        expect(screen.getByTestId("rate-limit").textContent).toContain(
          "usage limit",
        ),
      { timeout: 5000 },
    );
    // The optimistic user bubble is removed by the rate-limit branch in
    // useCopilotStream so the user can edit + resend.
    await waitFor(() => expectTextContent("user-message-count", "0"));
  });

  it("surfaces an HTTP 500 response as the hook's error state", async () => {
    server.use(
      copilotStreamErrorHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        status: 500,
        body: "kaboom",
      }),
    );

    render(<StreamHarness />);
    await clickSend();

    await waitFor(
      () => expect(screen.getByTestId("error").textContent).toContain("kaboom"),
      { timeout: 5000 },
    );
  });
});

describe("CoPilot streaming (SSE) — stop", () => {
  beforeEach(() => {
    resetCopilotChatRegistry();
  });

  afterEach(() => {
    resetCopilotChatRegistry();
    vi.useRealTimers();
  });

  it("flips is-user-stopping and ends the stream when stop is clicked mid-stream", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: [
          { type: "start", messageId: "msg-1" }, //          0 — fast
          { type: "start-step" }, //                          1 — fast
          { type: "text-start", id: "t1" }, //                2 — fast
          { type: "text-delta", id: "t1", delta: "Streaming " }, //3 — fast
          { type: "text-delta", id: "t1", delta: "in progress " }, //4 — STALL before
          { type: "text-delta", id: "t1", delta: "should be cut off" },
          { type: "text-end", id: "t1" },
          { type: "finish-step" },
          { type: "finish" },
        ],
        // Setup + first delta emit instantly; the stream then stalls before
        // producing chunk 4, giving the stop click a wide window to land.
        perChunkDelaysMs: [0, 0, 0, 0, 5000, 0, 0, 0, 0],
      }),
    );

    render(<StreamHarness />);
    await clickSend();

    // Wait until at least the first delta has landed so the stop button has
    // something live to abort.
    await waitFor(() => {
      expect(screen.getByTestId("assistant-text").textContent).toContain(
        "Streaming ",
      );
    });

    await clickStop();

    await waitFor(() => expectTextContent("status", "ready"), {
      timeout: 5000,
    });
    // Post-stop chunks never reach the message…
    expect(screen.getByTestId("assistant-text").textContent).not.toContain(
      "should be cut off",
    );
    // …and the cancellation marker is appended so the bubble shows the stop.
    expect(screen.getByTestId("assistant-text").textContent).toContain(
      "Operation cancelled",
    );
  });
});

describe("CoPilot streaming (SSE) — resume on mount", () => {
  beforeEach(() => {
    resetCopilotChatRegistry();
  });

  afterEach(() => {
    resetCopilotChatRegistry();
    vi.useRealTimers();
  });

  it("issues a GET resume and renders streamed content when mounted with hasActiveStream=true", async () => {
    server.use(
      copilotResumeHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: assistantTextChunks("Resumed content here."),
      }),
    );

    render(<StreamHarness hasActiveStream hydratedMessages={[]} />);

    await waitFor(
      () => expectTextContent("assistant-text", "Resumed content here."),
      { timeout: 5000 },
    );
    await waitFor(() => expectTextContent("status", "ready"));
  });
});
