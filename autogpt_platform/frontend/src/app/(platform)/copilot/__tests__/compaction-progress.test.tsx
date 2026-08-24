import type { SessionDetailResponseMessagesItem } from "@/app/api/__generated__/models/sessionDetailResponseMessagesItem";
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

// A pre-check false positive: the row opens, the phase streams, then the
// prediction is retired — the row closes with the abort sentinel (output "")
// and the turn continues with normal text. No compaction happened, so no
// compaction copy may survive the settle.
const ABORTED_COMPACTION_TURN: UIMessageChunk[] = [
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
    output: "",
  },
  { type: "finish-step" },
  { type: "start-step" },
  { type: "text-start", id: "t1" },
  { type: "text-delta", id: "t1", delta: "No condensing needed." },
  { type: "text-end", id: "t1" },
  { type: "finish-step" },
  { type: "finish" },
];

// Two compaction cycles in one assistant message: the first settles with
// JSON stats, then a second cycle opens and streams its `summarizing`
// phase. The live phase belongs to the second row only — the first must
// keep showing its settled payoff copy.
const TWO_CYCLE_TURN: UIMessageChunk[] = [
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
    }),
  },
  { type: "finish-step" },
  { type: "data-compaction", data: { phase: "rebuilding" } },
  { type: "start-step" },
  { type: "text-start", id: "t1" },
  { type: "text-delta", id: "t1", delta: "Continuing." },
  { type: "text-end", id: "t1" },
  { type: "finish-step" },
  { type: "start-step" },
  {
    type: "tool-input-start",
    toolCallId: "compaction-2",
    toolName: "context_compaction",
  },
  {
    type: "tool-input-available",
    toolCallId: "compaction-2",
    toolName: "context_compaction",
    input: {},
  },
  {
    type: "data-compaction",
    data: { phase: "summarizing", tokensBefore: 64000 },
  },
  // The hold sits before this chunk so the second cycle stays live long
  // enough to assert against (see SECOND_CYCLE_HOLD_INDEX).
  {
    type: "tool-output-available",
    toolCallId: "compaction-2",
    output: JSON.stringify({
      summary: "Earlier messages were summarized to fit within context limits.",
      tokensBefore: 64000,
      tokensAfter: 20000,
    }),
  },
  { type: "finish-step" },
  { type: "data-compaction", data: { phase: "rebuilding" } },
  { type: "start-step" },
  { type: "text-start", id: "t2" },
  { type: "text-delta", id: "t2", delta: "Done again." },
  { type: "text-end", id: "t2" },
  { type: "finish-step" },
  { type: "finish" },
];

const SECOND_CYCLE_HOLD_INDEX = TWO_CYCLE_TURN.findIndex(
  (c) => c.type === "tool-output-available" && c.toolCallId === "compaction-2",
);

// A turn whose stream dies right after a `data-compaction` part — the Stop
// button or a terminal error ends the stream with no trailing text, so no
// content part ever lands to null out the phase. The streaming gate must
// retire the bar when the stream closes.
const DEAD_STREAM_TURN: UIMessageChunk[] = [
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
    }),
  },
  { type: "finish-step" },
  { type: "data-compaction", data: { phase: "rebuilding" } },
  // The hold sits before this chunk so the live "Reloading context…" bar
  // is observable before the stream closes (see DEAD_STREAM_HOLD_INDEX).
  { type: "finish" },
];

const DEAD_STREAM_HOLD_INDEX = DEAD_STREAM_TURN.findIndex(
  (c) => c.type === "finish",
);

// A turn interrupted while the compaction row is still OPEN — no
// `tool-output-available` ever arrives, so the row stays `input-available`.
// Nothing may keep animating once the stream closes, and the card must not
// claim a compaction that never reported completion.
const INTERRUPTED_OPEN_ROW_TURN: UIMessageChunk[] = [
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
  { type: "finish" },
];

const INTERRUPTED_HOLD_INDEX = INTERRUPTED_OPEN_ROW_TURN.findIndex(
  (c) => c.type === "finish",
);

// The rebuild phase is emitted AFTER the tool row closes — it covers the
// transcript upload, CLI restart and uncached prefill, which is the silence
// this feature exists to narrate. Holding on the chunk that follows it keeps
// that window observable: settling the row on `output-available` would erase
// the phase entirely.
const REBUILDING_HOLD_INDEX =
  COMPACTION_TURN.findIndex(
    (c) => c.type === "data-compaction" && c.data?.phase === "rebuilding",
  ) + 1;

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

  it("renders nothing for a row retired by an aborted prediction", async () => {
    server.use(
      sessionHandler(),
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: ABORTED_COMPACTION_TURN,
        delayMsBetweenChunks: 15,
      }),
    );

    renderHost();
    await typeAndSend("quick question");

    await waitFor(() => {
      expect(screen.getByText("No condensing needed.")).toBeDefined();
    });
    // The abort sentinel (output "") must not read as a real compaction —
    // neither the live copy nor the settled payoff copy may survive.
    expect(screen.queryByText(/Condensing our conversation/)).toBeNull();
    expect(screen.queryByText(/Condensed the conversation/)).toBeNull();
    expect(screen.queryByRole("progressbar")).toBeNull();
  });

  it("keeps the first cycle settled while a second cycle streams", async () => {
    server.use(
      sessionHandler(),
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: TWO_CYCLE_TURN,
        perChunkDelaysMs: TWO_CYCLE_TURN.map((_, i) =>
          i === SECOND_CYCLE_HOLD_INDEX ? 600 : 15,
        ),
      }),
    );

    renderHost();
    await typeAndSend("summarise twice");

    await waitFor(() => {
      // Second cycle live in `summarizing`…
      expect(screen.getByText("Condensing our conversation…")).toBeDefined();
      // …while the first row keeps its settled payoff copy…
      expect(
        screen.getByText(/Condensed the conversation · 128K → 31K tokens/),
      ).toBeDefined();
      // …and only the live row carries a progress bar. Without the
      // per-row phase gate, the second cycle's phase re-animates the
      // first (closed) row and two bars render.
      expect(screen.getAllByRole("progressbar")).toHaveLength(1);
    });
  });

  it("retires the live bar when the stream dies after a compaction phase", async () => {
    server.use(
      sessionHandler(),
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: DEAD_STREAM_TURN,
        perChunkDelaysMs: DEAD_STREAM_TURN.map((_, i) =>
          i === DEAD_STREAM_HOLD_INDEX ? 600 : 15,
        ),
      }),
    );

    renderHost();
    await typeAndSend("summarise this");

    // Live during the stream: the trailing `rebuilding` phase keeps the
    // bar up while the connection is open.
    await waitFor(() => {
      expect(screen.getByRole("progressbar")).toBeDefined();
    });
    // Once the stream closes with no trailing text, the streaming gate
    // must null the phase — the row settles instead of spinning forever.
    await waitFor(() => {
      expect(screen.queryByRole("progressbar")).toBeNull();
      expect(
        screen.getByText(/Condensed the conversation · 128K → 31K tokens/),
      ).toBeDefined();
    });
  });

  it("keeps the bar live through the rebuild that follows the closed row", async () => {
    server.use(
      sessionHandler(),
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: COMPACTION_TURN,
        perChunkDelaysMs: COMPACTION_TURN.map((_, i) =>
          i === REBUILDING_HOLD_INDEX ? 600 : 15,
        ),
      }),
    );

    renderHost();
    await typeAndSend("summarise this");

    // The tool row has already closed with its JSON output by this point;
    // the row must still be live and narrating the rebuild.
    await waitFor(() => {
      expect(screen.getByText("Reloading context…")).toBeDefined();
    });
    expect(screen.getByRole("progressbar")).toBeDefined();
  });

  it("renders nothing for a row left open when the stream is interrupted", async () => {
    server.use(
      sessionHandler(),
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: INTERRUPTED_OPEN_ROW_TURN,
        perChunkDelaysMs: INTERRUPTED_OPEN_ROW_TURN.map((_, i) =>
          i === INTERRUPTED_HOLD_INDEX ? 600 : 15,
        ),
      }),
    );

    renderHost();
    await typeAndSend("summarise this");

    // Live while the connection is open.
    await waitFor(() => {
      expect(screen.getByRole("progressbar")).toBeDefined();
    });
    // The row never closed, so once the stream is gone it is neither a live
    // bar nor a "Condensed…" claim — it disappears.
    await waitFor(() => {
      expect(screen.queryByRole("progressbar")).toBeNull();
    });
    expect(screen.queryByText(/Condensed/)).toBeNull();
    expect(screen.queryByText(/Condensing/)).toBeNull();
  });
});

// A row restored from the DB never carries a `data-compaction` part — that
// part only ever exists on a live stream. `getLatestCompactionPhase` reads
// null in that case, and the tool part's persisted output makes it
// `output-available`, so the card renders settled — this proves the
// back-compat path for both the new JSON payload and the old plain-sentence
// rows some sessions still have on disk.
describe("compaction rows restored from the database", () => {
  beforeEach(() => {
    resetCopilotChatRegistry();
  });

  afterEach(() => {
    resetCopilotChatRegistry();
  });

  it("renders new JSON rows with their numbers and no bar", async () => {
    const messages: SessionDetailResponseMessagesItem[] = [
      { role: "user", content: "hi", sequence: 0 },
      {
        role: "assistant",
        content: "",
        sequence: 1,
        tool_calls: [
          {
            id: "compaction-1",
            type: "function",
            function: { name: "context_compaction", arguments: "{}" },
          },
        ],
      },
      {
        role: "tool",
        tool_call_id: "compaction-1",
        sequence: 2,
        content: JSON.stringify({
          summary:
            "Earlier messages were summarized to fit within context limits.",
          tokensBefore: 128000,
          tokensAfter: 31000,
        }),
      },
    ];
    renderHost({ sessionOverride: { messages } });

    await waitFor(() => {
      expect(
        screen.getByText("Condensed the conversation · 128K → 31K tokens"),
      ).toBeDefined();
    });
    expect(screen.queryByRole("progressbar")).toBeNull();
  });

  it("renders legacy plain-sentence rows without crashing", async () => {
    const messages: SessionDetailResponseMessagesItem[] = [
      { role: "user", content: "hi", sequence: 0 },
      {
        role: "assistant",
        content: "",
        sequence: 1,
        tool_calls: [
          {
            id: "compaction-1",
            type: "function",
            function: { name: "context_compaction", arguments: "{}" },
          },
        ],
      },
      {
        role: "tool",
        tool_call_id: "compaction-1",
        sequence: 2,
        content:
          "Earlier messages were summarized to fit within context limits.",
      },
    ];
    renderHost({ sessionOverride: { messages } });

    await waitFor(() => {
      expect(
        screen.getByText("Condensed the conversation to keep going"),
      ).toBeDefined();
    });
  });
});
