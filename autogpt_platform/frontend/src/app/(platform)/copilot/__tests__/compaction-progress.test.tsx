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

// A settled compaction row followed immediately by another completed
// generic tool call, both ahead of the final text — reproduces the
// collapsed-group bug where two adjacent "generic completed tool" parts
// fold into a single `CollapsedToolGroup`, burying the payoff copy.
const COMPACTION_THEN_TOOL_TURN: UIMessageChunk[] = [
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
  {
    type: "tool-input-start",
    toolCallId: "search-1",
    toolName: "web_search",
  },
  {
    type: "tool-input-available",
    toolCallId: "search-1",
    toolName: "web_search",
    input: { query: "latest news" },
  },
  {
    type: "tool-output-available",
    toolCallId: "search-1",
    output: JSON.stringify({ result: "ok" }),
  },
  { type: "finish-step" },
  { type: "start-step" },
  { type: "text-start", id: "t1" },
  { type: "text-delta", id: "t1", delta: "Done." },
  { type: "text-end", id: "t1" },
  { type: "finish-step" },
  { type: "finish" },
];

// `COMPACTION_TURN`'s chunk index of `tool-output-available` — the delay
// applied *before* this chunk is what keeps the turn parked in the
// "summarizing" phase long enough for `waitFor` to observe it (see below).
const SUMMARIZING_HOLD_INDEX = COMPACTION_TURN.findIndex(
  (c) => c.type === "tool-output-available",
);

describe("context compaction progress", () => {
  beforeEach(() => {
    resetCopilotChatRegistry();
    server.use(
      sessionHandler(),
      // A uniform 15ms gap keeps the stream from resolving inside a single
      // microtask tick, but the "summarizing" phase itself only spans the
      // single gap between the `data-compaction` chunk and
      // `tool-output-available` — at 15ms that's narrower than
      // `waitFor`'s ~50ms poll interval, so the assertion below could miss
      // it entirely depending on scheduling. Hold specifically at that
      // gap for long enough to make the phase reliably observable.
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: COMPACTION_TURN,
        perChunkDelaysMs: COMPACTION_TURN.map((_, i) =>
          i === SUMMARIZING_HOLD_INDEX ? 300 : 15,
        ),
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
      // CompactionCard owns its own spinner/label — the generic
      // ThinkingIndicator must not double up alongside it. The tool row
      // (or, before it opens, the step-start marker) keeps `hasInflight`
      // true throughout the compaction turn, so this holds from the very
      // first chunk onward — not just once the bar is visible.
      expect(screen.queryByText("Thinking...")).toBeNull();
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

  it("keeps a settled compaction row out of a collapsed tool group when another tool call follows it", async () => {
    server.use(
      sessionHandler(),
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: COMPACTION_THEN_TOOL_TURN,
        delayMsBetweenChunks: 15,
      }),
    );

    renderHost();
    await typeAndSend("search and summarise");

    await waitFor(() => {
      expect(
        screen.getByText(/Condensed the conversation · 128K → 31K tokens/),
      ).toBeDefined();
    });
    // If the compaction row had folded into a CollapsedToolGroup with the
    // adjacent web_search call, the payoff copy above would be hidden
    // behind a "N tool calls completed" toggle instead of standing alone.
    expect(screen.queryByText(/tool calls/)).toBeNull();
  });
});
