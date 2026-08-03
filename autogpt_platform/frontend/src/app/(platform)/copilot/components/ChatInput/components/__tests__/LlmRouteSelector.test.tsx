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
const transportTestState = vi.hoisted(() => ({
  transports: null as unknown[] | null,
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => mockToast(...args),
  useToast: () => ({ toast: mockToast, dismiss: vi.fn() }),
}));

vi.mock("../../../../helpers/copilotLlmAuth", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("../../../../helpers/copilotLlmAuth")>();
  return {
    ...actual,
    getConnectedSubsidizedLlmTransports: (
      providers: Parameters<
        typeof actual.getConnectedSubsidizedLlmTransports
      >[0],
    ) =>
      transportTestState.transports ??
      actual.getConnectedSubsidizedLlmTransports(providers),
  };
});

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
  transportTestState.transports = null;
  useCopilotUIStore.getState().setCopilotLlmAuth({
    authProvider: "platform",
    credentialId: null,
  });
});

describe("LlmRouteSelector", () => {
  it("keeps platform selected when no subsidized transport is connected", () => {
    render(<SelectorHarness credentials={[]} />);

    expect(screen.queryByLabelText(/AI connection/i)).toBeNull();
    expect(useCopilotUIStore.getState().copilotLlmAuth).toEqual({
      authProvider: "platform",
      credentialId: null,
    });
  });

  it("automatically selects the sole subsidized transport without showing a selector", async () => {
    render(<SelectorHarness credentials={[codexCredential]} />);

    await waitFor(() => {
      expect(useCopilotUIStore.getState().copilotLlmAuth).toEqual({
        authProvider: "codex",
        credentialId: "codex-credential-1",
      });
    });
    expect(screen.queryByLabelText(/AI connection/i)).toBeNull();
  });

  it("returns a disconnected next-session selection to platform visibly", async () => {
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
          title: "AI connections changed",
        }),
      );
    });
    expect(useCopilotUIStore.getState().copilotLlmAuth).toEqual({
      authProvider: "platform",
      credentialId: null,
    });
    expect(screen.queryByLabelText(/AI connection/i)).toBeNull();
  });

  it("offers only connected subsidized transports when more than one is available", async () => {
    const user = userEvent.setup();
    transportTestState.transports = [
      {
        authProvider: "codex",
        provider: "codex",
        credentialType: "oauth2",
        label: "ChatGPT/Codex",
        description: "Uses your ChatGPT plan",
        credentials: [codexCredential],
      },
      {
        authProvider: "grok",
        provider: "grok",
        credentialType: "oauth2",
        label: "Grok",
        description: "Uses your Grok plan",
        credentials: [
          {
            ...codexCredential,
            id: "grok-credential-1",
            provider: "grok",
            title: "Personal Grok",
          },
        ],
      },
    ];

    render(<SelectorHarness credentials={[codexCredential]} />);

    await user.click(
      screen.getByLabelText(/AI connection: Choose connection/i),
    );
    expect(screen.queryByText("AutoGPT platform")).toBeNull();
    expect(screen.getByText("ChatGPT/Codex")).toBeTruthy();
    expect(screen.getByText("Grok")).toBeTruthy();

    await user.click(screen.getByText("ChatGPT/Codex"));
    expect(useCopilotUIStore.getState().copilotLlmAuth).toEqual({
      authProvider: "codex",
      credentialId: "codex-credential-1",
    });
  });
});
