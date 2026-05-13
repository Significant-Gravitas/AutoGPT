import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  copilotStreamHandler,
} from "@/tests/integrations/copilot-sse";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import type { UIMessage } from "ai";
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

function StreamHarness() {
  const refetchSession = async () => ({ data: undefined });
  const { messages, sendMessage, status } = useCopilotStream({
    sessionId: TEST_SESSION_ID,
    hydratedMessages: [] as UIMessage[],
    hasActiveStream: false,
    refetchSession,
    copilotMode: undefined,
    copilotModel: undefined,
  });

  const assistantText = messages
    .filter((m) => m.role === "assistant")
    .flatMap((m) => m.parts)
    .filter((p): p is Extract<typeof p, { type: "text" }> => p.type === "text")
    .map((p) => p.text)
    .join("");

  return (
    <div>
      <div data-testid="stream-status">{status}</div>
      <div data-testid="assistant-text">{assistantText}</div>
      <button
        data-testid="send"
        onClick={() => sendMessage({ text: "hello copilot" })}
      >
        send
      </button>
    </div>
  );
}

describe("CoPilot streaming (SSE) — happy path", () => {
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

    const user = (await import("@testing-library/user-event")).default.setup();
    render(<StreamHarness />);

    await user.click(screen.getByTestId("send"));

    await waitFor(
      () => {
        expect(screen.getByTestId("assistant-text").textContent).toBe(
          "Hello from the copilot.",
        );
      },
      { timeout: 5000 },
    );
    await waitFor(() => {
      expect(screen.getByTestId("stream-status").textContent).toBe("ready");
    });
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

    const user = (await import("@testing-library/user-event")).default.setup();
    render(<StreamHarness />);

    await user.click(screen.getByTestId("send"));

    await waitFor(
      () => {
        expect(screen.getByTestId("assistant-text").textContent).toBe(
          "Hello from the copilot.",
        );
      },
      { timeout: 5000 },
    );
  });
});
