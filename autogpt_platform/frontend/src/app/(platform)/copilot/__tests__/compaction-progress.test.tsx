import { server } from "@/mocks/mock-server";
import { copilotStreamHandler } from "@/tests/integrations/copilot-sse";
import { screen, waitFor } from "@testing-library/react";
import type { UIMessageChunk } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resetCopilotChatRegistry } from "../copilotChatRegistry";
import {
  renderHost,
  sessionHandler,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
  typeAndSend,
} from "./sse-helpers";

// Pin the backend host so the CoPilot transport's absolute URL is
// deterministic — the transport bypasses the Next proxy on purpose
// (Vercel function-timeout dodge), so MSW has to match an absolute URL.
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

// Replace the auth token fetch with a static header so we don't need
// real auth in tests.
vi.mock("../helpers", async (importActual) => {
  const actual = await importActual<typeof import("../helpers")>();
  return {
    ...actual,
    getCopilotAuthHeaders: async () => ({ "x-test-auth": "yes" }),
  };
});

// useChatSession depends on useAuth via useCopilotPage's auth gate.
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isUserLoading: false, isLoggedIn: true }),
}));

// Keep mode/model toggles and artifacts off so the chat input renders a
// single, predictable Submit button.
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    ARTIFACTS: "ARTIFACTS",
    CHAT_MODE_OPTION: "CHAT_MODE_OPTION",
    ENABLE_PLATFORM_PAYMENT: "ENABLE_PLATFORM_PAYMENT",
  },
  useGetFlag: () => false,
}));

// Mirrors the backend's emission order after the ordering fix: the row is
// OPEN while the work runs, then closes with JSON stats.
const COMPACTION_TURN: UIMessageChunk[] = [
  { type: "start" },
  { type: "start-step" },
  {
    type: "tool-input-start",
    toolCallId: "compaction-1",
    toolName: "context_compaction",
  },
  {
    type: "tool-input-available",
    toolCallId: "compaction-1",
    toolName: "context_compaction",
    input: {},
  },
  {
    type: "data-compaction",
    data: { phase: "summarizing", tokensBefore: 128000 },
  },
  {
    type: "tool-output-available",
    toolCallId: "compaction-1",
    output: JSON.stringify({
      summary: "Earlier messages were summarized to fit within context limits.",
      tokensBefore: 128000,
      tokensAfter: 31000,
      messagesBefore: 412,
      messagesAfter: 38,
    }),
  },
  { type: "finish-step" },
  { type: "data-compaction", data: { phase: "rebuilding" } },
  { type: "start-step" },
  { type: "text-start", id: "t1" },
  { type: "text-delta", id: "t1", delta: "All caught up." },
  { type: "text-end", id: "t1" },
  { type: "finish-step" },
  { type: "finish" },
];

describe("context compaction progress", () => {
  beforeEach(() => {
    resetCopilotChatRegistry();
    server.use(
      sessionHandler(),
      // A small inter-chunk delay keeps the stream from resolving in a
      // single microtask tick, so the transient "summarizing" phase is
      // actually observable by `waitFor` before the turn settles.
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: COMPACTION_TURN,
        delayMsBetweenChunks: 15,
      }),
    );
  });

  afterEach(() => {
    resetCopilotChatRegistry();
  });

  it("shows a live bar while compacting, not a completed message", async () => {
    renderHost();
    await typeAndSend("summarise this");

    await waitFor(() => {
      expect(screen.getByRole("progressbar")).toBeDefined();
      expect(screen.getByText("Condensing our conversation…")).toBeDefined();
    });
  });

  it("lands on the payoff copy with real numbers", async () => {
    renderHost();
    await typeAndSend("summarise this");

    await waitFor(() => {
      expect(
        screen.getByText(/Condensed the conversation · 128K → 31K tokens/),
      ).toBeDefined();
    });
  });

  it("never shows the old apologetic copy", async () => {
    renderHost();
    await typeAndSend("summarise this");

    await waitFor(() => {
      expect(screen.getByText("All caught up.")).toBeDefined();
    });
    expect(screen.queryByText(/Earlier messages were summarized/)).toBeNull();
  });
});
