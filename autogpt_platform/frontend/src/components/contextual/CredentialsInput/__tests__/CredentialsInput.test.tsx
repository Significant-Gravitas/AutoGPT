import {
  render,
  screen,
  cleanup,
  fireEvent,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import { CredentialsInput } from "../CredentialsInput";

vi.mock("@/hooks/useCredentials", () => ({ default: vi.fn() }));
vi.mock("@/lib/autogpt-server-api/context", () => ({
  useBackendAPI: vi.fn(),
  BackendAPIProvider: ({ children }: { children: React.ReactNode }) => children,
}));
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: vi.fn(),
  useToast: () => ({ toast: vi.fn(), dismiss: vi.fn(), toasts: [] }),
}));
vi.mock("@/lib/oauth-popup", () => ({
  openOAuthPopup: vi.fn(),
  OAUTH_ERROR_WINDOW_CLOSED: "Sign-in window was closed",
  OAUTH_ERROR_FLOW_CANCELED: "OAuth flow was canceled",
  OAUTH_ERROR_FLOW_TIMED_OUT: "OAuth flow timed out",
}));
vi.mock("@/app/api/__generated__/endpoints/mcp/mcp", () => ({
  postV2InitiateOauthLoginForAnMcpServer: vi.fn(),
}));

import useCredentials from "@/hooks/useCredentials";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { openOAuthPopup } from "@/lib/oauth-popup";

const mockUseCredentials = useCredentials as unknown as ReturnType<
  typeof vi.fn
>;
const mockUseBackendAPI = useBackendAPI as unknown as ReturnType<typeof vi.fn>;
const mockOpenOAuthPopup = openOAuthPopup as unknown as ReturnType<
  typeof vi.fn
>;

const baseSchema: BlockIOCredentialsSubSchema = {
  credentials_provider: ["google"],
  credentials_types: ["oauth2"],
  credentials_scopes: ["drive.file", "drive.metadata"],
} as BlockIOCredentialsSubSchema;

type CredentialsReturn = ReturnType<typeof useCredentials>;
type BackendAPI = ReturnType<typeof useBackendAPI>;

function makeCredentialsReturn(overrides: Partial<CredentialsReturn> = {}) {
  return {
    provider: "google",
    providerName: "Google",
    savedCredentials: [],
    upgradeableCredentials: [],
    supportsApiKey: false,
    supportsOAuth2: true,
    supportsUserPassword: false,
    supportsHostScoped: false,
    isLoading: false,
    isSystemProvider: false,
    schema: baseSchema,
    oAuthCallback: vi.fn().mockResolvedValue({
      id: "new-cred",
      type: "oauth2",
      title: "Connected Google",
      provider: "google",
      scopes: ["drive.file", "drive.metadata"],
    }),
    mcpOAuthCallback: vi.fn(),
    createAPIKeyCredentials: vi.fn(),
    createUserPasswordCredentials: vi.fn(),
    createHostScopedCredentials: vi.fn(),
    deleteCredentials: vi.fn(),
    discriminatorValue: undefined,
    ...overrides,
  };
}

function makeBackendAPI(overrides: Partial<BackendAPI> = {}) {
  return {
    oAuthLogin: vi.fn().mockResolvedValue({
      login_url: "https://accounts.google.com/o/oauth2/auth",
      state_token: "state-xyz",
    }),
    onWebSocketMessage: vi.fn().mockReturnValue(() => {}),
    connectWebSocket: vi.fn().mockResolvedValue(undefined),
    sendWebSocketMessage: vi.fn(),
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  mockUseBackendAPI.mockReturnValue(makeBackendAPI());
});

afterEach(() => {
  cleanup();
});

describe("CredentialsInput – OAuth flow", () => {
  it("clicking the Add account button calls oAuthLogin without a credentialID", async () => {
    const oAuthLoginMock = vi.fn().mockResolvedValue({
      login_url: "https://accounts.google.com/o/oauth2/auth",
      state_token: "state-xyz",
    });
    mockUseBackendAPI.mockReturnValue(
      makeBackendAPI({ oAuthLogin: oAuthLoginMock }),
    );

    mockUseCredentials.mockReturnValue(makeCredentialsReturn());

    mockOpenOAuthPopup.mockReturnValue({
      promise: Promise.resolve({ code: "code-2", state: "state-xyz" }),
      cleanup: { abort: vi.fn() },
    });

    render(
      <CredentialsInput
        schema={baseSchema}
        onSelectCredentials={vi.fn()}
        showTitle={false}
      />,
    );

    const addAccountButton = await screen.findByRole("button", {
      name: /add account/i,
    });
    fireEvent.click(addAccountButton);

    await waitFor(() => {
      expect(oAuthLoginMock).toHaveBeenCalledWith(
        "google",
        ["drive.file", "drive.metadata"],
        undefined,
      );
    });
  });
});
