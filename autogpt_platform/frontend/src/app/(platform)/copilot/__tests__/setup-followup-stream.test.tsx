import { getGetV2GetSessionMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getListExpertIdentitiesMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  createStreamHold,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import { act, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import {
  getOrCreateCopilotChatRuntime,
  resetCopilotChatRegistry,
} from "../copilotChatRegistry";
import { useCopilotStreamStore } from "../copilotStreamStore";
import {
  renderHost,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
  typeAndSend,
} from "./sse-helpers";

const finishProbeStates = vi.hoisted(() => [] as boolean[]);

vi.mock("../useHydrateOnStreamEnd", async (importActual) => {
  const actual =
    await importActual<typeof import("../useHydrateOnStreamEnd")>();
  return {
    ...actual,
    useHydrateOnStreamEnd: (
      args: Parameters<typeof actual.useHydrateOnStreamEnd>[0],
    ) => {
      finishProbeStates.push(args.isFinishProbing);
      return actual.useHydrateOnStreamEnd(args);
    },
  };
});

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

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "test-user" },
    isUserLoading: false,
    isLoggedIn: true,
  }),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: () => false,
    useFlagStatus: () => ({ enabled: false, ready: true }),
  };
});

beforeEach(() => {
  finishProbeStates.length = 0;
  resetCopilotChatRegistry();
  useCopilotStreamStore.getState().resetAll();
});

afterEach(() => {
  resetCopilotChatRegistry();
  vi.restoreAllMocks();
  window.sessionStorage.clear();
});

it.each([false, true])(
  "sends an expert setup reply once after a pending-message 409 (delayed resume: %s)",
  async (delayedResume) => {
    const consoleError = vi.spyOn(console, "error");
    const expertId = "3f8b0f7e-9f30-4a3b-a6a1-000000000001";
    const question = "What should I work on first?";
    const reply = "Prepare the weekly report";
    const first = createStreamHold(5);
    const second = createStreamHold(5);
    const handshake = createStreamHold(0);
    let resumeRequested = false;
    let sessionReadAfterSetup = 0;
    const requests: unknown[] = [];
    const queueRequest = vi.fn(() =>
      HttpResponse.json({ detail: "No active turn" }, { status: 409 }),
    );
    server.use(
      getListExpertIdentitiesMockHandler200([
        {
          id: expertId,
          name: "Setup expert",
          avatar_url: null,
          role: "Reporting",
          is_archived: false,
        },
      ]),
      http.post(
        `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`,
        async ({ request }) => {
          requests.push(await request.json());
          const isSetup = requests.length === 1;
          return streamSseResponse(
            assistantTextChunks(isSetup ? question : "I will prepare it.", {
              messageId: isSetup ? "setup-questions" : "setup-reply",
            }),
            {
              hold: isSetup ? first.hold : second.hold,
              abortSignal: request.signal,
            },
          );
        },
      ),
      http.post(
        `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/messages/pending`,
        queueRequest,
      ),
      http.get(
        `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`,
        async ({ request }) => {
          resumeRequested = true;
          await handshake.hold.released;
          return streamSseResponse([], { abortSignal: request.signal });
        },
      ),
    );

    const host = renderHost({
      sessionResponse: getGetV2GetSessionMockHandler200(() => ({
        id: TEST_SESSION_ID,
        created_at: "2026-05-13T00:00:00Z",
        updated_at: "2026-05-13T00:00:00Z",
        user_id: "test-user",
        chat_status: "idle",
        messages: [],
        has_more_messages: false,
        oldest_sequence: null,
        expert_id: expertId,
        active_stream:
          delayedResume && requests.length > 0 && sessionReadAfterSetup++ === 0
            ? { turn_id: "setup-turn", last_message_id: "setup-questions" }
            : null,
        metadata: { dry_run: false, builder_graph_id: null },
      })),
    });
    try {
      await typeAndSend("Start setup");
      await screen.findByText(question, undefined, { timeout: 5_000 });
      const user = userEvent.setup();
      await user.type(screen.getByLabelText(/chat message input/i), reply);
      await user.click(screen.getByRole("button", { name: "Queue message" }));
      await waitFor(() => expect(queueRequest).toHaveBeenCalledOnce());
      expect(requests).toHaveLength(1);

      first.release();
      if (delayedResume) {
        await waitFor(() => expect(resumeRequested).toBe(true));
        await waitFor(() => {
          expect(finishProbeStates).toContain(true);
          expect(finishProbeStates.at(-1)).toBe(false);
        });
        await act(async () => {
          await new Promise((resolve) => setTimeout(resolve, 100));
        });
        expect(requests).toHaveLength(1);
        handshake.release();
      }
      await waitFor(() => expect(requests).toHaveLength(2), { timeout: 5_000 });
      expect(requests[1]).toMatchObject({ message: reply });
      await screen.findByText("I will prepare it.");
      second.release();
      handshake.release();
      const { chat } = getOrCreateCopilotChatRuntime(TEST_SESSION_ID);
      await waitFor(() => expect(chat.status).toBe("ready"));
      expect(queueRequest).toHaveBeenCalledOnce();
      expect(requests).toHaveLength(2);
      expect(chat.error).toBeUndefined();
      expect(
        consoleError.mock.calls.filter((args) =>
          args.some((arg) => arg instanceof TypeError),
        ),
      ).toEqual([]);
    } finally {
      first.release();
      second.release();
      handshake.release();
      host.unmount();
    }
  },
  15_000,
);
