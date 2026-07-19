import { server } from "@/mocks/mock-server";
import { copilotStreamHandler } from "@/tests/integrations/copilot-sse";
import { waitFor } from "@testing-library/react";
import type { UIMessageChunk } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resetCopilotChatRegistry } from "../copilotChatRegistry";
import { useCopilotUIStore } from "../store";
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
  useCopilotUIStore.getState().setCopilotChatMode("fast");
});

afterEach(() => {
  resetCopilotChatRegistry();
});

describe("AutoPilot streaming — server-initiated engine switch", () => {
  it("flips the chat mode when the stream emits data-mode-changed", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: [
          { type: "start", messageId: "msg-1" },
          { type: "text-start", id: "t1" },
          { type: "text-delta", id: "t1", delta: "Switching to Thinking…" },
          { type: "text-end", id: "t1" },
          {
            type: "data-mode-changed",
            data: { mode: "extended_thinking" },
          } as unknown as UIMessageChunk,
          { type: "finish" },
        ],
      }),
    );

    renderHost();
    await typeAndSend("build me an agent");

    await waitFor(
      () => {
        expect(useCopilotUIStore.getState().copilotChatMode).toBe(
          "extended_thinking",
        );
      },
      { timeout: 5000 },
    );
  });

  it("keeps the selected mode on streams without a mode change", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: [
          { type: "start", messageId: "msg-1" },
          { type: "text-start", id: "t1" },
          { type: "text-delta", id: "t1", delta: "Hi." },
          { type: "text-end", id: "t1" },
          { type: "finish" },
        ],
      }),
    );

    renderHost();
    await typeAndSend("hi");

    await waitFor(() => {
      expect(useCopilotUIStore.getState().copilotChatMode).toBe("fast");
    });
  });
});
