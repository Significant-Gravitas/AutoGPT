import {
  getGetV2ListChatConnectionsMockHandler200,
  getPutV2ChangeTheConnectionAnExistingChatRunsOnMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import { server } from "@/mocks/mock-server";
import {
  copilotStreamErrorHandler,
  copilotStreamHandler,
} from "@/tests/integrations/copilot-sse";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { UIMessageChunk } from "ai";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resetCopilotChatRegistry } from "../copilotChatRegistry";
import { useCopilotStreamStore } from "../copilotStreamStore";
import { getKickoffStatus, kickoffStorageKey } from "../expertKickoff";
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

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "test-user" },
    isUserLoading: false,
    isLoggedIn: true,
  }),
}));

const flagState = vi.hoisted(() => ({ experts: false }));

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    CHAT_MODE_OPTION: "CHAT_MODE_OPTION",
    ENABLE_PLATFORM_PAYMENT: "ENABLE_PLATFORM_PAYMENT",
    HIRE_EXPERTS: "HIRE_EXPERTS",
  },
  useGetFlag: (flag: string) =>
    flag === "HIRE_EXPERTS" ? flagState.experts : false,
}));

beforeEach(() => {
  resetCopilotChatRegistry();
  useCopilotStreamStore.getState().resetAll();
  useCopilotUIStore.setState({ initialPrompt: null });
  window.localStorage.clear();
  flagState.experts = false;
});

afterEach(() => {
  resetCopilotChatRegistry();
  useCopilotStreamStore.getState().resetAll();
  useCopilotUIStore.setState({ initialPrompt: null });
  window.localStorage.clear();
  flagState.experts = false;
});

describe("Otto streaming — error paths", () => {
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
      await screen.findByText(/daily usage limit reached/i, undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
  });

  it("clears kickoff recovery state after an HTTP 429", async () => {
    const expertId = "3f8b0f7e-9f30-4a3b-a6a1-000000000001";
    flagState.experts = true;
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
      await screen.findByText(/daily usage limit reached/i, undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
    await waitFor(() =>
      expect(getKickoffStatus("test-user", expertId)).toBe("idle"),
    );
    expect(
      window.localStorage.getItem(kickoffStorageKey("test-user", expertId)),
    ).toBeNull();
    expect(useCopilotStreamStore.getState().getCoord(TEST_SESSION_ID)).toEqual({
      lastSubmittedMessageText: null,
      lastSubmittedKickoffExpertId: null,
      lastSubmittedKickoffAttemptToken: null,
    });
    expect(useCopilotUIStore.getState().initialPrompt).toBeNull();
    expect(
      useCopilotStreamStore
        .getState()
        .getMessageSnapshot(TEST_SESSION_ID)
        .filter((message) => message.role === "user"),
    ).toHaveLength(0);
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

  it("keeps the plan dialog for our cap and offers the linked subscription beside the upgrade", async () => {
    let switchedTo: unknown = null;
    server.use(
      copilotStreamErrorHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        status: 429,
        body: ourCap,
      }),
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
    await typeAndSend("over the cap");

    // Our own cap, so our own dialog: the upgrade path survives ...
    expect(
      await screen.findByText(/daily usage limit reached/i, undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
    expect(screen.queryByText(/hit this connection's limit/i)).toBeNull();
    expect(
      screen.getByRole("button", { name: /upgrade plan|contact us/i }),
    ).toBeDefined();

    // ... and the linked subscription is offered next to it, because the cap
    // does not apply to a turn billed to the user's own credential.
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole(
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
    await waitFor(() =>
      expect(screen.queryByText(/daily usage limit reached/i)).toBeNull(),
    );
  });

  it("offers nothing to continue on when the platform is the only connection", async () => {
    server.use(
      copilotStreamErrorHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        status: 429,
        body: ourCap,
      }),
      getGetV2ListChatConnectionsMockHandler200({
        offers: [platformOffer()],
      }),
    );

    renderHost();
    await typeAndSend("over the cap");

    expect(
      await screen.findByText(/daily usage limit reached/i, undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: /upgrade plan|contact us/i }),
    ).toBeDefined();
    // The connection that just refused is never offered back. Were it not
    // excluded, "Continue on AutoGPT Platform" would appear once the offers
    // load, so give that long enough to have happened.
    await expect(
      screen.findByRole("button", { name: /continue on/i }, { timeout: 1500 }),
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
      { type: "error", errorText: "OpenRouter rate-limited this deployment." },
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
    expect(screen.queryByText(/daily usage limit reached/i)).toBeNull();
  });
});
