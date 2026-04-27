import { renderHook, act } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import type {
  BlockIOCredentialsSubSchema,
  CredentialsMetaResponse,
} from "@/lib/autogpt-server-api";

vi.mock("@/hooks/useCredentials", () => ({ default: vi.fn() }));
vi.mock("@/lib/autogpt-server-api/context", () => ({
  useBackendAPI: vi.fn(),
}));
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: vi.fn(),
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
import { useCredentialsInput } from "../useCredentialsInput";

const mockUseCredentials = useCredentials as unknown as ReturnType<
  typeof vi.fn
>;
const mockUseBackendAPI = useBackendAPI as unknown as ReturnType<typeof vi.fn>;
const mockOpenOAuthPopup = openOAuthPopup as unknown as ReturnType<
  typeof vi.fn
>;

function makeCred(
  partial: Partial<CredentialsMetaResponse>,
): CredentialsMetaResponse {
  return {
    id: "cred-id",
    provider: "google",
    type: "oauth2",
    title: "Test",
    scopes: [],
    ...partial,
  } as CredentialsMetaResponse;
}

const baseSchema: BlockIOCredentialsSubSchema = {
  credentials_provider: ["google"],
  credentials_types: ["oauth2"],
  credentials_scopes: ["drive.file", "drive.metadata"],
} as BlockIOCredentialsSubSchema;

function makeCredentialsReturn(overrides: Record<string, any> = {}) {
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
    oAuthCallback: vi.fn().mockResolvedValue(makeCred({ id: "new-cred" })),
    mcpOAuthCallback: vi.fn(),
    createAPIKeyCredentials: vi.fn(),
    createUserPasswordCredentials: vi.fn(),
    createHostScopedCredentials: vi.fn(),
    deleteCredentials: vi.fn(),
    discriminatorValue: undefined,
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  mockUseBackendAPI.mockReturnValue({
    oAuthLogin: vi.fn().mockResolvedValue({
      login_url: "https://accounts.google.com/o/oauth2/auth",
      state_token: "state-123",
    }),
  });
});

describe("useCredentialsInput – upgradeableCredentials", () => {
  it("exposes userUpgradeableCredentials filtered from upgradeableCredentials", () => {
    const upgradeable = makeCred({
      id: "narrow",
      title: "Narrow Cred",
      scopes: ["drive.file"],
    });
    const systemUpgradeable = makeCred({
      id: "sys",
      title: "Use credits for Google",
      scopes: ["drive.file"],
      is_system: true,
    });

    mockUseCredentials.mockReturnValue(
      makeCredentialsReturn({
        savedCredentials: [],
        upgradeableCredentials: [upgradeable, systemUpgradeable],
      }),
    );

    const onSelect = vi.fn();
    const { result } = renderHook(() =>
      useCredentialsInput({
        schema: baseSchema,
        onSelectCredential: onSelect,
      }),
    );

    expect(result.current.isLoading).toBe(false);
    if (result.current.isLoading) return;

    // System credentials should be filtered out
    expect(result.current.userUpgradeableCredentials).toHaveLength(1);
    expect(result.current.userUpgradeableCredentials![0].id).toBe("narrow");
  });
});

describe("useCredentialsInput – handleScopeUpgrade", () => {
  it("passes credentialID through executeOAuthFlow to oAuthLogin", async () => {
    const oAuthLoginMock = vi.fn().mockResolvedValue({
      login_url: "https://accounts.google.com/o/oauth2/auth",
      state_token: "state-abc",
    });
    mockUseBackendAPI.mockReturnValue({ oAuthLogin: oAuthLoginMock });

    const oAuthCallback = vi.fn().mockResolvedValue(
      makeCred({
        id: "upgraded-cred",
        scopes: ["drive.file", "drive.metadata"],
      }),
    );

    mockUseCredentials.mockReturnValue(
      makeCredentialsReturn({ oAuthCallback }),
    );

    mockOpenOAuthPopup.mockReturnValue({
      promise: Promise.resolve({ code: "auth-code", state: "state-abc" }),
      cleanup: { abort: vi.fn() },
    });

    const onSelect = vi.fn();
    const { result } = renderHook(() =>
      useCredentialsInput({
        schema: baseSchema,
        onSelectCredential: onSelect,
      }),
    );

    expect(result.current.isLoading).toBe(false);
    if (result.current.isLoading) return;

    await act(async () => {
      await result.current.handleScopeUpgrade!("existing-cred-id");
    });

    expect(oAuthLoginMock).toHaveBeenCalledWith(
      "google",
      ["drive.file", "drive.metadata"],
      "existing-cred-id",
    );
  });

  it("handleOAuthLogin calls executeOAuthFlow without credentialID", async () => {
    const oAuthLoginMock = vi.fn().mockResolvedValue({
      login_url: "https://accounts.google.com/o/oauth2/auth",
      state_token: "state-xyz",
    });
    mockUseBackendAPI.mockReturnValue({ oAuthLogin: oAuthLoginMock });

    const oAuthCallback = vi.fn().mockResolvedValue(
      makeCred({
        id: "new-cred",
        scopes: ["drive.file", "drive.metadata"],
      }),
    );

    mockUseCredentials.mockReturnValue(
      makeCredentialsReturn({ oAuthCallback }),
    );

    mockOpenOAuthPopup.mockReturnValue({
      promise: Promise.resolve({ code: "code-2", state: "state-xyz" }),
      cleanup: { abort: vi.fn() },
    });

    const onSelect = vi.fn();
    const { result } = renderHook(() =>
      useCredentialsInput({
        schema: baseSchema,
        onSelectCredential: onSelect,
      }),
    );

    expect(result.current.isLoading).toBe(false);
    if (result.current.isLoading) return;

    await act(async () => {
      await result.current.handleOAuthLogin!();
    });

    // credentialID should be undefined (not passed)
    expect(oAuthLoginMock).toHaveBeenCalledWith(
      "google",
      ["drive.file", "drive.metadata"],
      undefined,
    );
  });
});
