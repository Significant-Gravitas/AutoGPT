import { getGetV2GetSessionMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  copilotStreamHandler,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import { screen, waitFor } from "@testing-library/react";
import { http } from "msw";
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

// Records the `isFinishProbing` prop on every render so the producer-side
// lifecycle (set before the probe loop, reset in `finally`) is observable
// without reaching into `useCopilotStream` internals.
const { isFinishProbingHistory } = vi.hoisted(() => ({
  isFinishProbingHistory: [] as boolean[],
}));

vi.mock("../useHydrateOnStreamEnd", async (importActual) => {
  const actual =
    await importActual<typeof import("../useHydrateOnStreamEnd")>();
  return {
    ...actual,
    useHydrateOnStreamEnd: (
      args: Parameters<typeof actual.useHydrateOnStreamEnd>[0],
    ) => {
      isFinishProbingHistory.push(args.isFinishProbing);
      return actual.useHydrateOnStreamEnd(args);
    },
  };
});

function streamUrl() {
  return `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`;
}

function sessionJson(
  activeStream: { turn_id: string; last_message_id: string } | null,
): SessionDetailResponse {
  return {
    id: TEST_SESSION_ID,
    created_at: "2026-05-13T00:00:00Z",
    updated_at: "2026-05-13T00:00:00Z",
    user_id: "test-user",
    chat_status: "idle",
    messages: [],
    has_more_messages: false,
    oldest_sequence: null,
    active_stream: activeStream,
    metadata: { dry_run: false, builder_graph_id: null },
  };
}

beforeEach(() => {
  resetCopilotChatRegistry();
  isFinishProbingHistory.length = 0;
});

afterEach(() => {
  resetCopilotChatRegistry();
});

describe("useCopilotStream — isFinishProbing lifecycle", () => {
  it("sets the flag during the post-finish probe and resets it when no continuation is pending", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: assistantTextChunks("Hi."),
      }),
    );

    renderHost();
    await typeAndSend("hi");
    await screen.findByText("Hi.", undefined, { timeout: 5000 });

    // The probe loop runs for ~500ms after the SSE finish.
    await waitFor(() => expect(isFinishProbingHistory).toContain(true), {
      timeout: 5000,
    });
    // No active backend stream → the loop exits normally and the finally
    // block resets the flag.
    await waitFor(() => expect(isFinishProbingHistory.at(-1)).toBe(false), {
      timeout: 5000,
    });
  });

  it(
    "resets the flag via finally when the probe finds a continuation stream",
    { timeout: 15000 },
    async () => {
      // Flipped by the stream POST: the session GET only reports a live
      // continuation stream once the first turn has actually run.
      let continuationPending = false;
      let resumeRequested = false;

      renderHost();
      // Registered AFTER renderHost so these take precedence over the
      // default pinned handlers (MSW matches most-recently-added first).
      server.use(
        getGetV2GetSessionMockHandler200(() =>
          sessionJson(
            continuationPending
              ? { turn_id: "turn-2", last_message_id: "msg-2" }
              : null,
          ),
        ),
        http.post(streamUrl(), ({ request }) => {
          continuationPending = true;
          return streamSseResponse(assistantTextChunks("Hi."), {
            abortSignal: request.signal,
          });
        }),
        http.get(streamUrl(), ({ request }) => {
          resumeRequested = true;
          continuationPending = false;
          return streamSseResponse(
            assistantTextChunks(" continued", { messageId: "test-message-2" }),
            { abortSignal: request.signal },
          );
        }),
      );

      await typeAndSend("hi");
      await screen.findByText("Hi.", undefined, { timeout: 5000 });

      // Probe window: the flag is up while the active-stream probe runs.
      await waitFor(() => expect(isFinishProbingHistory).toContain(true), {
        timeout: 5000,
      });
      // The probe sees the continuation stream → early-return through the
      // reconnect path → the finally block must still reset the flag.
      await waitFor(() => expect(isFinishProbingHistory.at(-1)).toBe(false), {
        timeout: 5000,
      });
      // ...and the reconnect actually picked the continuation turn up.
      await waitFor(() => expect(resumeRequested).toBe(true), {
        timeout: 8000,
      });
    },
  );
});
