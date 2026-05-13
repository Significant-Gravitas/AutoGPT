import { server } from "@/mocks/mock-server";
import {
  copilotStreamErrorHandler,
  copilotStreamHandler,
} from "@/tests/integrations/copilot-sse";
import { screen, waitFor } from "@testing-library/react";
import type { UIMessageChunk } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resetCopilotChatRegistry } from "../copilotChatRegistry";
import {
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

describe("AutoPilot streaming — error paths", () => {
  it("surfaces an SSE error chunk to the user", async () => {
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

    renderHost();
    await typeAndSend("hi");

    expect(
      await screen.findByText(/backend went sideways\./i, undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
  });

  it("opens the rate-limit dialog on HTTP 429 'usage limit'", async () => {
    server.use(
      copilotStreamErrorHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        status: 429,
        body: { detail: "You've reached your usage limit. Try again later." },
      }),
    );

    renderHost();
    await typeAndSend("rate limited please");

    // useCopilotStream's rate-limit branch sets rateLimitMessage, which the
    // RateLimitGate translates into a Dialog with this title.
    expect(
      await screen.findByText(/daily autopilot limit reached/i, undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
  });

  it("surfaces an HTTP 500 response as a visible error", async () => {
    server.use(
      copilotStreamErrorHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        status: 500,
        body: "kaboom",
      }),
    );

    renderHost();
    await typeAndSend("hi");

    expect(
      await screen.findByText(/kaboom/i, undefined, { timeout: 5000 }),
    ).toBeDefined();
  });

  it("preserves the partial assistant text when an error chunk arrives mid-stream", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: [
          { type: "start", messageId: "msg-1" },
          { type: "start-step" },
          { type: "text-start", id: "t1" },
          { type: "text-delta", id: "t1", delta: "Partial response so far." },
          { type: "text-end", id: "t1" },
          { type: "error", errorText: "Stream blew up." },
        ],
      }),
    );

    renderHost();
    await typeAndSend("hi");

    // Both the partial text and the error must be visible — losing the
    // partial would silently drop work the model already did, and hiding
    // the error would leave the user staring at a stalled bubble.
    expect(
      await screen.findByText("Partial response so far.", undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
    expect(await screen.findByText(/stream blew up\./i)).toBeDefined();
  });

  it("keeps the chat input enabled after an HTTP 500 so the user can retry", async () => {
    server.use(
      copilotStreamErrorHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        status: 500,
        body: "kaboom",
      }),
    );

    renderHost();
    await typeAndSend("hi");

    await screen.findByText(/kaboom/i, undefined, { timeout: 5000 });

    // Input must remain enabled for retries; locking it on stream error
    // would strand the user with no way to send a follow-up.
    await waitFor(() => {
      const input = screen.getByLabelText(
        /chat message input/i,
      ) as HTMLTextAreaElement;
      expect(input.disabled).toBe(false);
    });
    expect(screen.queryByRole("button", { name: /stop/i })).toBeNull();
  });
});
