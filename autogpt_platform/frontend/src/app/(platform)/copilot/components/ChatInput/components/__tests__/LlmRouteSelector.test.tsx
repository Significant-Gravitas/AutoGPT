import type { ChatTransportResponse } from "@/app/api/__generated__/models/chatTransportResponse";
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

const testState = vi.hoisted(() => ({
  transports: [] as ChatTransportResponse[] | null,
  isError: false,
  queryOptions: vi.fn(),
  toast: vi.fn(),
}));

vi.mock(
  "@/app/api/__generated__/endpoints/chat/chat",
  async (importOriginal) => {
    const actual =
      await importOriginal<
        typeof import("@/app/api/__generated__/endpoints/chat/chat")
      >();
    return {
      ...actual,
      useGetV2ListChatTransports: (options: unknown) => {
        testState.queryOptions(options);
        return {
          data:
            testState.transports === null
              ? undefined
              : { status: 200, data: { transports: testState.transports } },
          isPending: testState.transports === null && !testState.isError,
          isError: testState.isError,
        };
      },
    };
  },
);

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => testState.toast(...args),
  useToast: () => ({ toast: testState.toast, dismiss: vi.fn() }),
}));

const codexCredential: CredentialsMetaResponse = {
  id: "codex-credential-1",
  provider: "codex",
  type: "oauth2",
  title: "Personal ChatGPT",
  username: "person@example.com",
  scopes: [],
};

const hostedPlatform: ChatTransportResponse = {
  auth_provider: "platform",
  credential_id: null,
  label: "AutoGPT Platform",
  available: true,
  default: true,
};

const configuredSelfHosted: ChatTransportResponse = {
  auth_provider: "platform",
  credential_id: null,
  label: "Self-hosted chat",
  available: true,
  default: true,
};

const unconfiguredSelfHosted: ChatTransportResponse = {
  ...configuredSelfHosted,
  available: false,
  default: false,
};

const codexTransport: ChatTransportResponse = {
  auth_provider: "codex",
  credential_id: codexCredential.id,
  label: "ChatGPT/Codex",
  available: true,
  default: false,
};

function makeCodexProvider(): CredentialsProviderData {
  return {
    provider: "codex",
    providerName: "Codex",
    savedCredentials: [codexCredential],
    isSystemProvider: false,
    oAuthCallback: async () => codexCredential,
    mcpOAuthCallback: async () => codexCredential,
    createAPIKeyCredentials: async () => codexCredential,
    createUserPasswordCredentials: async () => codexCredential,
    createHostScopedCredentials: async () => codexCredential,
    deleteCredentials: async () => ({ deleted: true, revoked: null }),
  };
}

function SelectorHarness() {
  return (
    <CredentialsProvidersContext.Provider
      value={{ codex: makeCodexProvider() }}
    >
      <LlmRouteSelector />
    </CredentialsProvidersContext.Provider>
  );
}

afterEach(() => {
  cleanup();
  testState.transports = [];
  testState.isError = false;
  testState.queryOptions.mockClear();
  testState.toast.mockClear();
  useCopilotUIStore.getState().setCopilotLlmAuth({
    authProvider: "platform",
    credentialId: null,
  });
});

describe("LlmRouteSelector", () => {
  it("hides when the hosted platform is the only available route", () => {
    testState.transports = [hostedPlatform];

    render(<SelectorHarness />);

    expect(screen.queryByLabelText(/AI connection/i)).toBeNull();
    expect(useCopilotUIStore.getState().copilotLlmAuth).toEqual({
      authProvider: "platform",
      credentialId: null,
    });
  });

  it("offers hosted platform and Codex while defaulting to hosted", async () => {
    testState.transports = [hostedPlatform, codexTransport];
    const user = userEvent.setup();

    render(<SelectorHarness />);

    await user.click(screen.getByLabelText(/AI connection: AutoGPT Platform/i));
    expect(screen.getAllByText("AutoGPT Platform").length).toBeGreaterThan(0);
    expect(screen.getByText("ChatGPT/Codex")).toBeTruthy();
    expect(screen.getByText(/Personal ChatGPT/)).toBeTruthy();

    await user.click(screen.getByText("ChatGPT/Codex"));
    expect(useCopilotUIStore.getState().copilotLlmAuth).toEqual({
      authProvider: "codex",
      credentialId: codexCredential.id,
    });
  });

  it("refreshes a changed transport inventory immediately after pairing", () => {
    testState.transports = [hostedPlatform];
    const { rerender } = render(<SelectorHarness />);

    expect(screen.queryByLabelText(/AI connection/i)).toBeNull();
    expect(testState.queryOptions).toHaveBeenLastCalledWith(
      expect.objectContaining({
        query: expect.objectContaining({
          refetchOnWindowFocus: true,
          staleTime: 0,
        }),
      }),
    );

    testState.transports = [hostedPlatform, codexTransport];
    rerender(<SelectorHarness />);
    expect(
      screen.getByLabelText(/AI connection: AutoGPT Platform/i),
    ).toBeTruthy();

    testState.transports = [hostedPlatform];
    rerender(<SelectorHarness />);
    expect(screen.queryByLabelText(/AI connection/i)).toBeNull();
  });

  it("hides when configured self-hosted chat is the only route", () => {
    testState.transports = [configuredSelfHosted];

    render(<SelectorHarness />);

    expect(screen.queryByLabelText(/AI connection/i)).toBeNull();
  });

  it("offers configured self-hosted chat alongside Codex", async () => {
    testState.transports = [configuredSelfHosted, codexTransport];
    const user = userEvent.setup();

    render(<SelectorHarness />);

    await user.click(screen.getByLabelText(/AI connection: Self-hosted chat/i));
    expect(screen.getAllByText("Self-hosted chat").length).toBeGreaterThan(0);
    expect(screen.getByText("ChatGPT/Codex")).toBeTruthy();
  });

  it("automatically selects sole Codex on a keyless self-host", async () => {
    testState.transports = [
      unconfiguredSelfHosted,
      { ...codexTransport, default: true },
    ];

    render(<SelectorHarness />);

    await waitFor(() => {
      expect(useCopilotUIStore.getState().copilotLlmAuth).toEqual({
        authProvider: "codex",
        credentialId: codexCredential.id,
      });
    });
    expect(screen.queryByLabelText(/AI connection/i)).toBeNull();
  });

  it("links a keyless self-host with no Codex connection to setup", () => {
    testState.transports = [unconfiguredSelfHosted];

    render(<SelectorHarness />);

    const setupLink = screen.getByLabelText("Set up an AI connection");
    expect(setupLink.getAttribute("href")).toBe("/settings/integrations");
  });

  it("shows inventory failures without inventing a route", () => {
    testState.transports = null;
    testState.isError = true;

    render(<SelectorHarness />);

    expect(screen.getByLabelText("AI connections unavailable")).toBeTruthy();
  });
});
