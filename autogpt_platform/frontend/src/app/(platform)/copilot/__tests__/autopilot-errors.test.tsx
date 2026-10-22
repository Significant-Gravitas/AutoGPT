import {
  getGetV2ListChatConnectionsMockHandler200,
  getGetV2GetCopilotUsageMockHandler200,
  getPutV2ChangeTheConnectionAnExistingChatRunsOnMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  copilotStreamErrorHandler,
  copilotStreamHandler,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { UIMessageChunk } from "ai";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getOrCreateCopilotChatRuntime } from "../copilotChatRegistry";
import { useCopilotStreamStore } from "../copilotStreamStore";
import { getKickoffStatus, kickoffStorageKey } from "../expertKickoff";
import { useCopilotUIStore } from "../store";
import { getTurnRuntime } from "../stream/turnRuntime";
import {
  renderHost,
  resetChatRuntimes,
  STREAM_PATHS,
  type StreamPath,
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

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "test-user" },
    isUserLoading: false,
    isLoggedIn: true,
  }),
}));

const flagState = vi.hoisted(() => ({ billing: false }));

const streamPath = vi.hoisted(() => ({ runtime: false }));

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    CHAT_MODE_OPTION: "CHAT_MODE_OPTION",
    ENABLE_PLATFORM_PAYMENT: "ENABLE_PLATFORM_PAYMENT",
    COPILOT_STREAM_RUNTIME: "copilot-stream-runtime",
  },
  useGetFlag: (flag: string) =>
    flag === "copilot-stream-runtime"
      ? streamPath.runtime
      : flag === "ENABLE_PLATFORM_PAYMENT"
        ? flagState.billing
        : false,
}));

beforeEach(() => {
  resetChatRuntimes();
  useCopilotStreamStore.getState().resetAll();
  useCopilotUIStore.setState({ initialPrompt: null });
  window.localStorage.clear();
  flagState.billing = false;
});

afterEach(() => {
  resetChatRuntimes();
  useCopilotStreamStore.getState().resetAll();
  useCopilotUIStore.setState({ initialPrompt: null });
  window.localStorage.clear();
  flagState.billing = false;
});

describe.each(STREAM_PATHS)("on the %s path", (path) => {
  beforeEach(() => {
    streamPath.runtime = path === "stream runtime";
  });

  describe("Otto streaming — error paths", () => {
    it.each(["ready", "error"] as const)(
      "retries an empty response with a fresh stream when status is %s",
      async (status) => {
        const errorText = "The model returned an empty response.";
        const failureChunks = assistantTextChunks(
          `[__COPILOT_RETRYABLE_ERROR_a9c2__] ${errorText}`,
          { messageId: "failed-response" },
        );
        if (status === "error") {
          failureChunks.splice(-1, 1, { type: "error", errorText });
        }
        const requests: unknown[] = [];
        const queueRequest = vi.fn(() =>
          HttpResponse.json(
            {
              detail:
                "Session has no active turn. Start a new turn with POST /stream.",
            },
            { status: 409 },
          ),
        );
        server.use(
          http.post(
            `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`,
            async ({ request }) => {
              requests.push(await request.json());
              return streamSseResponse(
                requests.length === 1
                  ? failureChunks
                  : assistantTextChunks("Retry succeeded.", {
                      messageId: "retried-response",
                    }),
                { abortSignal: request.signal },
              );
            },
          ),
          http.post(
            `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/messages/pending`,
            queueRequest,
          ),
        );

        renderHost();
        await typeAndSend("Please try this task");
        await screen.findByText(errorText);
        await waitFor(() =>
          expect(settledStatus(path)).toBe(
            path === "stream runtime" ? RUNTIME_PHASE[status] : status,
          ),
        );

        await userEvent
          .setup()
          .click(screen.getByRole("button", { name: /try again/i }));

        await waitFor(() => expect(requests).toHaveLength(2));
        expect(requests[1]).toMatchObject({ message: "Please try this task" });
        expect(await screen.findByText("Retry succeeded.")).toBeDefined();
        expect(queueRequest).not.toHaveBeenCalled();
      },
      15_000,
    );

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
        await screen.findByText(/We couldn’t check your usage/i, undefined, {
          timeout: 5000,
        }),
      ).toBeDefined();
    });

    it("clears kickoff recovery state after an HTTP 429", async () => {
      const expertId = "3f8b0f7e-9f30-4a3b-a6a1-000000000001";
      server.use(
        http.get("*/api/experts/identities", () =>
          HttpResponse.json([
            {
              id: expertId,
              name: "Maria",
              avatar_url: null,
              role: "Operations expert",
              is_archived: false,
            },
          ]),
        ),
        copilotStreamErrorHandler({
          baseUrl: TEST_BACKEND_BASE_URL,
          sessionId: TEST_SESSION_ID,
          status: 429,
          body: { detail: "You've reached your usage limit. Try again later." },
        }),
      );

      renderHost({
        searchParams: `?sessionId=${TEST_SESSION_ID}&expertId=${expertId}&kickoff=1`,
        sessionOverride: { expert_id: expertId },
      });

      expect(
        await screen.findByText(/We couldn’t check your usage/i, undefined, {
          timeout: 5000,
        }),
      ).toBeDefined();
      await waitFor(() =>
        expect(getKickoffStatus("test-user", expertId)).toBe("idle"),
      );
      expect(
        window.localStorage.getItem(kickoffStorageKey("test-user", expertId)),
      ).toBeNull();
      expect(
        useCopilotStreamStore.getState().getCoord(TEST_SESSION_ID),
      ).toEqual({
        lastSubmittedMessageText: null,
        lastSubmittedKickoffExpertId: null,
        lastSubmittedKickoffAttemptToken: null,
      });
      expect(useCopilotUIStore.getState().initialPrompt).toBeNull();
      const userRows =
        path === "stream runtime"
          ? getTurnRuntime(TEST_SESSION_ID)
              .getSnapshot()
              .segments.filter((seg) => seg.kind === "user")
          : useCopilotStreamStore
              .getState()
              .getMessageSnapshot(TEST_SESSION_ID)
              .filter((message) => message.role === "user");
      expect(userRows).toHaveLength(0);
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
      expect(screen.queryByRole("button", { name: "Stop" })).toBeNull();
    });
  });

  describe("AutoPilot streaming — our usage cap next to a linked subscription", () => {
    function platformOffer(): AIConnectionOffer {
      return {
        offer_id: "platform:deployment",
        provider_family: "autogpt",
        auth_provider: "platform",
        display_name: "AutoGPT Platform",
        auth_method: "deployment",
        credential_id: null,
        backed_by_label: "Your AutoGPT plan",
        description: "Runs on your AutoGPT plan.",
        state: "ready",
        selectable: true,
        is_default: true,
        tiers: [],
        limitations: [],
        lock_reason: null,
        unlock_href: null,
      };
    }

    function chatgptOffer(): AIConnectionOffer {
      return {
        ...platformOffer(),
        offer_id: "codex:cred-1",
        provider_family: "openai",
        auth_provider: "codex",
        display_name: "ChatGPT",
        auth_method: "chatgpt_oauth",
        credential_id: "cred-1",
        is_default: false,
      };
    }

    // What the backend's admission check sends once it refuses on our own
    // daily budget: a 429 whose body is the typed envelope, not a bare string.
    const ourCap = {
      detail: {
        kind: "usage_limit",
        message: "You've reached your daily usage limit. Resets in 1h 0m.",
        authProvider: "platform",
        credentialId: null,
        resetsAt: null,
        retryable: false,
        reconnectFixesIt: false,
      },
    };

    function usageLimitAdmission() {
      let refused = false;
      server.use(
        getGetV2GetCopilotUsageMockHandler200(() => ({
          tier: "PRO",
          daily: {
            percent_used: refused ? 100 : 0,
            resets_at: new Date("2099-10-08T00:00:00Z"),
          },
          weekly: {
            percent_used: 40,
            resets_at: new Date("2099-10-12T00:00:00Z"),
          },
        })),
      );
      return http.post(
        `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`,
        () => {
          refused = true;
          return HttpResponse.json(ourCap, { status: 429 });
        },
      );
    }

    it("keeps the plan dialog for our cap and offers the linked subscription beside the upgrade", async () => {
      flagState.billing = true;
      let switchedTo: unknown = null;
      server.use(
        getGetV2ListChatConnectionsMockHandler200({
          offers: [platformOffer(), chatgptOffer()],
        }),
        getPutV2ChangeTheConnectionAnExistingChatRunsOnMockHandler200(
          async ({ request }) => {
            switchedTo = await request.json();
            return {};
          },
        ),
      );

      renderHost();
      server.use(usageLimitAdmission());
      await typeAndSend("over the cap");

      // Our own cap, so our own dialog: the upgrade path survives ...
      const dialog = await screen.findByRole("dialog");
      expect(
        await within(dialog).findByText(
          /You’ve hit today’s limit/i,
          undefined,
          {
            timeout: 5000,
          },
        ),
      ).toBeDefined();
      expect(screen.queryByText(/hit this connection's limit/i)).toBeNull();
      expect(
        await within(dialog).findByRole("button", { name: /upgrade to max/i }),
      ).toBeDefined();

      // ... and the linked subscription is offered next to it, because the cap
      // does not apply to a turn billed to the user's own credential.
      const user = userEvent.setup();
      await user.click(
        await within(dialog).findByRole(
          "button",
          { name: /continue on chatgpt/i },
          { timeout: 5000 },
        ),
      );

      await waitFor(() =>
        expect(switchedTo).toEqual({
          llm_auth_provider: "codex",
          llm_credential_id: "cred-1",
        }),
      );
      await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    });

    it("offers nothing to continue on when the platform is the only connection", async () => {
      flagState.billing = true;
      server.use(
        getGetV2ListChatConnectionsMockHandler200({
          offers: [platformOffer()],
        }),
      );

      renderHost();
      server.use(usageLimitAdmission());
      await typeAndSend("over the cap");

      const dialog = await screen.findByRole("dialog");
      expect(
        await within(dialog).findByText(
          /You’ve hit today’s limit/i,
          undefined,
          {
            timeout: 5000,
          },
        ),
      ).toBeDefined();
      expect(
        await within(dialog).findByRole("button", { name: /upgrade to max/i }),
      ).toBeDefined();
      // The connection that just refused is never offered back. Were it not
      // excluded, "Continue on AutoGPT Platform" would appear once the offers
      // load, so give that long enough to have happened.
      await expect(
        screen.findByRole(
          "button",
          { name: /continue on/i },
          { timeout: 1500 },
        ),
      ).rejects.toThrow();
    });

    it("treats a mid-turn limit on the platform route as the provider's, not ours", async () => {
      // A self-host runs its own OpenRouter or local gateway on the "platform"
      // route, so its upstream 429 arrives with the same authProvider as our
      // cap. It came on the stream, mid-turn, which our cap never does -- and
      // it must not be answered with "upgrade your plan" for an account we
      // do not bill.
      const chunks: UIMessageChunk[] = [
        { type: "start", messageId: "msg-1" },
        { type: "start-step" },
        {
          type: "data-provider-failure",
          data: {
            kind: "usage_limit",
            message: "OpenRouter rate-limited this deployment.",
            authProvider: "platform",
            credentialId: null,
            resetsAt: null,
            retryable: false,
            reconnectFixesIt: false,
          },
        },
        {
          type: "error",
          errorText: "OpenRouter rate-limited this deployment.",
        },
      ];
      server.use(
        copilotStreamHandler({
          baseUrl: TEST_BACKEND_BASE_URL,
          sessionId: TEST_SESSION_ID,
          chunks,
        }),
        getGetV2ListChatConnectionsMockHandler200({
          offers: [platformOffer()],
        }),
      );

      renderHost();
      await typeAndSend("hi");

      expect(
        await screen.findByText(/hit this connection's limit/i, undefined, {
          timeout: 5000,
        }),
      ).toBeDefined();
      expect(screen.queryByText(/You’ve hit today’s limit/i)).toBeNull();
    });
  });
});

// The runtime settles a turn in its own phase, which the page maps to a status.
const RUNTIME_PHASE = { ready: "finished", error: "failed" } as const;

function settledStatus(path: StreamPath) {
  return path === "stream runtime"
    ? getTurnRuntime(TEST_SESSION_ID).getSnapshot().phase
    : getOrCreateCopilotChatRuntime(TEST_SESSION_ID).chat.status;
}

describe.each(STREAM_PATHS)(
  "AutoPilot reopened on the %s path — a failure stored on the chat",
  (path) => {
    beforeEach(() => {
      streamPath.runtime = path === "stream runtime";
    });

    function failedChat(kind: string) {
      return [
        {
          id: "row-0",
          role: "user",
          content: "Summarize my inbox.",
          sequence: 0,
        },
        {
          id: "row-1",
          role: "assistant",
          content: "[__COPILOT_ERROR_f7a1__] The connection refused this turn.",
          sequence: 1,
          metadata: {
            provider_failure: {
              kind,
              message: "The connection refused this turn.",
              authProvider: "codex",
              credentialId: "cred-1",
              resetsAt: null,
              retryable: false,
              reconnectFixesIt: kind === "auth_expired",
            },
          },
        },
      ];
    }

    // Still on the connection that failed, so the failure is still live.
    const onThatConnection = {
      dry_run: false,
      builder_graph_id: null,
      llm_auth_provider: "codex",
      llm_credential_id: "cred-1",
    } as const;

    beforeEach(() => {
      server.use(getGetV2ListChatConnectionsMockHandler200({ offers: [] }));
    });

    it("reopens the continue-elsewhere dialog for a spent quota", async () => {
      renderHost({
        sessionOverride: {
          messages: failedChat("usage_limit"),
          metadata: onThatConnection,
        },
      });

      expect(
        await screen.findByText(/hit this connection's limit/i, undefined, {
          timeout: 5000,
        }),
      ).toBeDefined();
    }, 15_000);

    it("leaves it closed for a failure that is not a limit", async () => {
      renderHost({
        sessionOverride: {
          messages: failedChat("auth_expired"),
          metadata: onThatConnection,
        },
      });

      await screen.findByText("Summarize my inbox.", undefined, {
        timeout: 5000,
      });
      await expect(
        screen.findByText(/hit this connection's limit/i, undefined, {
          timeout: 1500,
        }),
      ).rejects.toThrow();
    }, 15_000);
  },
);
