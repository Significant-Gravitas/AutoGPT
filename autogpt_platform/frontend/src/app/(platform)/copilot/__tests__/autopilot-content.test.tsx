import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  copilotStreamHandler,
} from "@/tests/integrations/copilot-sse";
import { screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resetCopilotChatRegistry } from "../copilotChatRegistry";
import {
  copilotStreamSequenceHandler,
  renderHost,
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

// Replace the Supabase token fetch with a static header so we don't need
// real auth in tests.
vi.mock("../helpers", async (importActual) => {
  const actual = await importActual<typeof import("../helpers")>();
  return {
    ...actual,
    getCopilotAuthHeaders: async () => ({ "x-test-auth": "yes" }),
  };
});

// useChatSession depends on useSupabase via useCopilotPage's auth gate.
vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({ isUserLoading: false, isLoggedIn: true }),
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

beforeEach(() => {
  resetCopilotChatRegistry();
});

afterEach(() => {
  resetCopilotChatRegistry();
});

describe("AutoPilot streaming — content rendering", () => {
  it("renders assistant text from a single text-delta frame", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: assistantTextChunks("Hello from the copilot."),
      }),
    );

    renderHost();
    await typeAndSend("hi");

    expect(
      await screen.findByText("Hello from the copilot.", undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
  });

  it("concatenates multiple text-delta frames into a single rendered message", async () => {
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

    renderHost();
    await typeAndSend("hi");

    expect(
      await screen.findByText("Hello from the copilot.", undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
  });

  it("renders the assistant's final text after reasoning chunks", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: [
          { type: "start", messageId: "msg-1" },
          { type: "start-step" },
          { type: "reasoning-start", id: "r1" },
          { type: "reasoning-delta", id: "r1", delta: "Thinking " },
          { type: "reasoning-delta", id: "r1", delta: "step by step." },
          { type: "reasoning-end", id: "r1" },
          { type: "text-start", id: "t1" },
          { type: "text-delta", id: "t1", delta: "Final answer." },
          { type: "text-end", id: "t1" },
          { type: "finish-step" },
          { type: "finish" },
        ],
      }),
    );

    renderHost();
    await typeAndSend("hi");

    expect(
      await screen.findByText("Final answer.", undefined, { timeout: 5000 }),
    ).toBeDefined();
  });

  it("renders text emitted after a tool call in the same turn", async () => {
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
          { type: "text-start", id: "t1" },
          { type: "text-delta", id: "t1", delta: "The weather is sunny." },
          { type: "text-end", id: "t1" },
          { type: "finish-step" },
          { type: "finish" },
        ],
      }),
    );

    renderHost();
    await typeAndSend("weather?");

    expect(
      await screen.findByText("The weather is sunny.", undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
  });

  it("renders text from both steps of a two-step turn", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: [
          { type: "start", messageId: "msg-1" },
          { type: "start-step" },
          { type: "text-start", id: "t1" },
          { type: "text-delta", id: "t1", delta: "First step text." },
          { type: "text-end", id: "t1" },
          { type: "finish-step" },
          { type: "start-step" },
          { type: "text-start", id: "t2" },
          { type: "text-delta", id: "t2", delta: "Second step text." },
          { type: "text-end", id: "t2" },
          { type: "finish-step" },
          { type: "finish" },
        ],
      }),
    );

    renderHost();
    await typeAndSend("hi");

    // Each step's text part renders in its own element, so assert both
    // individually rather than as a single concatenated string.
    expect(
      await screen.findByText("First step text.", undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
    expect(await screen.findByText("Second step text.")).toBeDefined();
  });

  it("completes the turn cleanly on an empty completion (no content, no error)", async () => {
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

    renderHost();
    await typeAndSend("hi");

    await waitFor(
      () => {
        expect(screen.queryByRole("button", { name: /stop/i })).toBeNull();
        expect(
          screen.queryByRole("button", { name: /submit/i }),
        ).not.toBeNull();
      },
      { timeout: 5000 },
    );
    expect(screen.queryByText(/encountered an error/i)).toBeNull();
  });

  it("renders both assistant replies across two back-to-back turns in the same session", async () => {
    server.use(
      copilotStreamSequenceHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunksPerTurn: [
          assistantTextChunks("First reply.", { messageId: "msg-1" }),
          assistantTextChunks("Second reply.", { messageId: "msg-2" }),
        ],
      }),
    );

    renderHost();
    await typeAndSend("first message");
    expect(
      await screen.findByText("First reply.", undefined, { timeout: 5000 }),
    ).toBeDefined();

    await typeAndSend("second message");
    expect(
      await screen.findByText("Second reply.", undefined, { timeout: 5000 }),
    ).toBeDefined();

    // First reply must still be visible — both turns should accumulate in
    // the chat log, not replace each other.
    expect(screen.getByText("First reply.")).toBeDefined();
  });

  it("renders inline markdown emphasis through the assistant's text pipeline", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: assistantTextChunks("Hello **bold** world."),
      }),
    );

    renderHost();
    await typeAndSend("emphasize this");

    // The raw `**bold**` literal must not appear in the rendered text — if
    // it did, the markdown step was bypassed entirely. (Streamdown wraps
    // inline emphasis runs in spans for its streaming animation rather
    // than a bare <strong>, so we don't pin the tag.)
    await screen.findByText(/bold/i, undefined, { timeout: 5000 });
    expect(screen.queryByText("**bold**", { exact: true })).toBeNull();
    expect(screen.queryByText(/\*\*bold\*\*/)).toBeNull();
  });
});
