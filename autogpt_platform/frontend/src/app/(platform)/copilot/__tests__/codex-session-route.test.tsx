import type { CredentialsMetaResponse } from "@/lib/autogpt-server-api";
import {
  CredentialsProvidersContext,
  type CredentialsProviderData,
  type CredentialsProvidersContextType,
} from "@/providers/agent-credentials/credentials-provider";
import { server } from "@/mocks/mock-server";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useCopilotUIStore } from "../store";
import { useChatSession } from "../useChatSession";

const testState = vi.hoisted(() => ({
  isCodexEnabled: true,
  toast: vi.fn(),
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => testState.toast(...args),
  useToast: () => ({ toast: testState.toast, dismiss: vi.fn() }),
}));

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { CODEX_SUBSCRIPTION_COPILOT: "codex-subscription-copilot" },
  useGetFlag: () => testState.isCodexEnabled,
}));

const codexCredential: CredentialsMetaResponse = {
  id: "codex-credential-1",
  provider: "codex",
  type: "oauth2",
  title: "Personal ChatGPT",
  scopes: [],
};

function makeCodexProvider(
  savedCredentials: CredentialsMetaResponse[],
): CredentialsProviderData {
  return {
    provider: "codex",
    providerName: "Codex",
    savedCredentials,
    isSystemProvider: false,
    oAuthCallback: async () => codexCredential,
    mcpOAuthCallback: async () => codexCredential,
    createAPIKeyCredentials: async () => codexCredential,
    createUserPasswordCredentials: async () => codexCredential,
    createHostScopedCredentials: async () => codexCredential,
    deleteCredentials: async () => ({ deleted: true, revoked: null }),
  };
}

function SessionHarness() {
  const { createSession, setSessionId, sessionLlmAuthProvider } =
    useChatSession();
  return (
    <>
      <div data-testid="session-llm-route">
        {sessionLlmAuthProvider ?? "unresolved"}
      </div>
      <button
        type="button"
        onClick={() => void createSession().catch(() => {})}
      >
        Create session
      </button>
      <button type="button" onClick={() => void setSessionId("existing")}>
        Open existing session
      </button>
    </>
  );
}

function renderSessionHarness(providers: CredentialsProvidersContextType) {
  return render(
    <CredentialsProvidersContext.Provider value={providers}>
      <SessionHarness />
    </CredentialsProvidersContext.Provider>,
  );
}

function captureCreateRequest() {
  let requestBody: unknown = null;
  server.use(
    http.post("*/api/chat/sessions", async ({ request }) => {
      requestBody = await request.json();
      return HttpResponse.json({
        id: "new-session-1",
        created_at: "2026-01-01T00:00:00Z",
        user_id: "user-1",
      });
    }),
  );
  return () => requestBody;
}

afterEach(() => {
  cleanup();
  server.resetHandlers();
  testState.isCodexEnabled = true;
  testState.toast.mockClear();
  useCopilotUIStore.getState().setCopilotLlmAuth({
    authProvider: "platform",
    credentialId: null,
  });
});

describe("useChatSession Codex route", () => {
  it("creates a platform session without a credential ID by default", async () => {
    const getRequestBody = captureCreateRequest();
    renderSessionHarness({});

    fireEvent.click(screen.getByRole("button", { name: "Create session" }));

    await waitFor(() => {
      expect(getRequestBody()).toEqual({ llm_auth_provider: "platform" });
    });
  });

  it("sends the explicitly selected saved Codex credential", async () => {
    useCopilotUIStore.getState().setCopilotLlmAuth({
      authProvider: "codex",
      credentialId: "codex-credential-1",
    });
    const getRequestBody = captureCreateRequest();
    renderSessionHarness({ codex: makeCodexProvider([codexCredential]) });

    fireEvent.click(screen.getByRole("button", { name: "Create session" }));

    await waitFor(() => {
      expect(getRequestBody()).toEqual({
        llm_auth_provider: "codex",
        llm_credential_id: "codex-credential-1",
      });
    });
  });

  it("refuses to create a session after the selected credential disappears", async () => {
    useCopilotUIStore.getState().setCopilotLlmAuth({
      authProvider: "codex",
      credentialId: "codex-credential-1",
    });
    const getRequestBody = captureCreateRequest();
    renderSessionHarness({ codex: makeCodexProvider([]) });

    fireEvent.click(screen.getByRole("button", { name: "Create session" }));

    await waitFor(() => {
      expect(testState.toast).toHaveBeenCalledWith(
        expect.objectContaining({
          variant: "destructive",
          title: "ChatGPT/Codex connection unavailable",
        }),
      );
    });
    expect(getRequestBody()).toBeNull();
  });

  it("refuses the Codex route when the feature flag is disabled", async () => {
    testState.isCodexEnabled = false;
    useCopilotUIStore.getState().setCopilotLlmAuth({
      authProvider: "codex",
      credentialId: "codex-credential-1",
    });
    const getRequestBody = captureCreateRequest();
    renderSessionHarness({ codex: makeCodexProvider([codexCredential]) });

    fireEvent.click(screen.getByRole("button", { name: "Create session" }));

    await waitFor(() => {
      expect(testState.toast).toHaveBeenCalled();
    });
    expect(getRequestBody()).toBeNull();
  });

  it("restores the immutable route from existing session metadata", async () => {
    server.use(
      http.get("*/api/chat/sessions/existing", () =>
        HttpResponse.json({
          id: "existing",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z",
          user_id: "user-1",
          metadata: { llm_auth_provider: "codex" },
          messages: [],
        }),
      ),
    );
    renderSessionHarness({ codex: makeCodexProvider([codexCredential]) });

    fireEvent.click(
      screen.getByRole("button", { name: "Open existing session" }),
    );

    await waitFor(() => {
      expect(screen.getByTestId("session-llm-route").textContent).toBe("codex");
    });
  });
});
