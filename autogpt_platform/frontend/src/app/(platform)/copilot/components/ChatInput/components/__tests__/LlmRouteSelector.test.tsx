import type { CredentialsMetaResponse } from "@/lib/autogpt-server-api";
import {
  CredentialsProvidersContext,
  type CredentialsProviderData,
} from "@/providers/agent-credentials/credentials-provider";
import {
  cleanup,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useCopilotUIStore } from "../../../../store";
import { LlmRouteSelector } from "../LlmRouteSelector";

const mockToast = vi.fn();
let isCodexFlagEnabled = true;

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => mockToast(...args),
  useToast: () => ({ toast: mockToast, dismiss: vi.fn() }),
}));

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { CODEX_SUBSCRIPTION_COPILOT: "codex-subscription-copilot" },
  useGetFlag: () => isCodexFlagEnabled,
}));

const codexCredential: CredentialsMetaResponse = {
  id: "codex-credential-1",
  provider: "codex",
  type: "oauth2",
  title: "Personal ChatGPT",
  username: "person@example.com",
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

function SelectorHarness({
  credentials,
}: {
  credentials: CredentialsMetaResponse[];
}) {
  return (
    <CredentialsProvidersContext.Provider
      value={{ codex: makeCodexProvider(credentials) }}
    >
      <LlmRouteSelector />
    </CredentialsProvidersContext.Provider>
  );
}

afterEach(() => {
  cleanup();
  mockToast.mockClear();
  isCodexFlagEnabled = true;
  useCopilotUIStore.getState().setCopilotLlmAuth({
    authProvider: "platform",
    credentialId: null,
  });
});

describe("LlmRouteSelector", () => {
  it("does not offer Codex without both the feature flag and a saved OAuth credential", () => {
    const { rerender } = render(<SelectorHarness credentials={[]} />);
    expect(screen.queryByLabelText(/AI connection/i)).toBeNull();

    isCodexFlagEnabled = false;
    rerender(<SelectorHarness credentials={[codexCredential]} />);
    expect(screen.queryByLabelText(/AI connection/i)).toBeNull();
  });

  it("stores an explicit Codex credential selection", async () => {
    const user = userEvent.setup();
    render(<SelectorHarness credentials={[codexCredential]} />);

    await user.click(
      screen.getByLabelText(
        "AI connection: AutoGPT platform — change connection",
      ),
    );
    await user.click(await screen.findByText(/Personal ChatGPT/));

    expect(useCopilotUIStore.getState().copilotLlmAuth).toEqual({
      authProvider: "codex",
      credentialId: "codex-credential-1",
    });
    expect(
      screen.getByLabelText("AI connection: ChatGPT/Codex — change connection"),
    ).toBeTruthy();
  });

  it("keeps a missing selection fail-closed until the user chooses platform", async () => {
    const user = userEvent.setup();
    useCopilotUIStore.getState().setCopilotLlmAuth({
      authProvider: "codex",
      credentialId: "codex-credential-1",
    });
    const { rerender } = render(
      <SelectorHarness credentials={[codexCredential]} />,
    );

    rerender(<SelectorHarness credentials={[]} />);

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          variant: "destructive",
          title: "ChatGPT/Codex connection unavailable",
        }),
      );
    });
    expect(useCopilotUIStore.getState().copilotLlmAuth.authProvider).toBe(
      "codex",
    );
    await user.click(
      screen.getByLabelText(
        "AI connection: ChatGPT/Codex unavailable — change connection",
      ),
    );
    await user.click(await screen.findByText("AutoGPT platform"));

    expect(useCopilotUIStore.getState().copilotLlmAuth).toEqual({
      authProvider: "platform",
      credentialId: null,
    });
  });
});
