import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  copilotResumeHandler,
  copilotStreamHandler,
} from "@/tests/integrations/copilot-sse";
import { screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resetCopilotChatRegistry } from "../copilotChatRegistry";
import {
  clickStop,
  renderHost,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
  typeAndSend,
} from "./sse-helpers";

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

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({ isUserLoading: false, isLoggedIn: true }),
}));

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

describe("AutoPilot streaming — submit / stop lifecycle", () => {
  it("swaps the submit button to Stop while streaming and back to Submit when done", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: assistantTextChunks("Hi."),
        delayMsBetweenChunks: 30,
      }),
    );

    renderHost();
    expect(screen.getByRole("button", { name: /submit/i })).toBeDefined();

    await typeAndSend("hi");

    expect(await screen.findByRole("button", { name: /stop/i })).toBeDefined();

    expect(
      await screen.findByText("Hi.", undefined, { timeout: 5000 }),
    ).toBeDefined();
    await waitFor(() => {
      expect(screen.queryByRole("button", { name: /stop/i })).toBeNull();
    });
  });

  it("ends the stream and shows the manual-stop marker when Stop is clicked mid-stream", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: [
          { type: "start", messageId: "msg-1" },
          { type: "start-step" },
          { type: "text-start", id: "t1" },
          { type: "text-delta", id: "t1", delta: "Streaming " },
          { type: "text-delta", id: "t1", delta: "in progress " },
          { type: "text-delta", id: "t1", delta: "should be cut off" },
          { type: "text-end", id: "t1" },
          { type: "finish-step" },
          { type: "finish" },
        ],
        // Setup + first delta emit instantly; stall before the "should be cut
        // off" delta so the stop click has a wide window to land.
        perChunkDelaysMs: [0, 0, 0, 0, 5000, 0, 0, 0, 0],
      }),
    );

    renderHost();
    await typeAndSend("hi");

    // Wait for the first delta so Stop has something live to abort.
    await screen.findByText(/streaming/i, undefined, { timeout: 5000 });

    await clickStop();

    expect(
      await screen.findByText(/you manually stopped this chat/i, undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
    // Post-stop chunks never reach the bubble.
    expect(screen.queryByText(/should be cut off/i)).toBeNull();
  });
});

describe("AutoPilot streaming — resume on mount", () => {
  it("issues a GET resume and renders streamed content when the session has an active_stream", async () => {
    server.use(
      copilotResumeHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: assistantTextChunks("Resumed content here."),
      }),
    );

    renderHost({
      sessionOverride: {
        active_stream: {
          turn_id: "turn-1",
          last_message_id: "msg-prev",
          started_at: "2026-05-13T00:00:00Z",
        },
      },
    });

    expect(
      await screen.findByText("Resumed content here.", undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
  });
});
