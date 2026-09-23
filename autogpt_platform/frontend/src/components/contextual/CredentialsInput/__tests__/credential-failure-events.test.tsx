import {
  render,
  screen,
  cleanup,
  fireEvent,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import React from "react";
import { CredentialsInput } from "../CredentialsInput";

const capture = vi.hoisted(() => vi.fn());
vi.mock("posthog-js", () => ({ default: { capture } }));

vi.mock("@/hooks/useCredentials", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/hooks/useCredentials")>()),
  default: vi.fn(),
}));
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
  preOpenOAuthPopup: vi.fn(() => null),
  OAUTH_ERROR_WINDOW_CLOSED: "Sign-in window was closed",
  OAUTH_ERROR_FLOW_CANCELED: "OAuth flow was canceled",
  OAUTH_ERROR_FLOW_TIMED_OUT: "OAuth flow timed out",
  OAUTH_ERROR_POPUP_BLOCKED: "Popup blocked — opened in a new tab instead.",
  OAUTH_ERROR_POPUP_BLOCKED_NO_TAB:
    "Popup blocked. Allow popups for this site and retry.",
}));
vi.mock("@/app/api/__generated__/endpoints/mcp/mcp", () => ({
  postV2InitiateOauthLoginForAnMcpServer: vi.fn(),
}));

import useCredentials from "@/hooks/useCredentials";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import {
  openOAuthPopup,
  OAUTH_ERROR_FLOW_TIMED_OUT,
  OAUTH_ERROR_POPUP_BLOCKED_NO_TAB,
} from "@/lib/oauth-popup";

const mockUseCredentials = useCredentials as unknown as ReturnType<
  typeof vi.fn
>;
const mockUseBackendAPI = useBackendAPI as unknown as ReturnType<typeof vi.fn>;
const mockOpenOAuthPopup = openOAuthPopup as unknown as ReturnType<
  typeof vi.fn
>;

const schema: BlockIOCredentialsSubSchema = {
  type: "object",
  properties: {},
  credentials_provider: ["google"],
  credentials_types: ["oauth2"],
  credentials_scopes: ["drive.file", "drive.metadata"],
};

function makeCredentialsReturn(overrides: Record<string, unknown> = {}) {
  return {
    provider: "google",
    providerName: "Google",
    savedCredentials: [],
    allProviderCredentials: [],
    upgradeableCredentials: [],
    supportsApiKey: false,
    supportsOAuth2: true,
    supportsUserPassword: false,
    supportsHostScoped: false,
    isLoading: false,
    isSystemProvider: false,
    schema,
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

function renderCard() {
  render(
    <CredentialsInput
      schema={schema}
      onSelectCredentials={vi.fn()}
      showTitle
      variant="node"
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: /add account/i }));
}

// A promise rejected at construction is unhandled until the flow reaches
// its await, which vitest reports as an error.
function rejectLater(message: string) {
  return new Promise<never>((_resolve, reject) =>
    setTimeout(() => reject(new Error(message)), 0),
  );
}

function capturedFailures() {
  return capture.mock.calls.filter(([event]) =>
    String(event).startsWith("credential_"),
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  mockUseBackendAPI.mockReturnValue({
    oAuthLogin: vi.fn().mockResolvedValue({
      login_url: "https://accounts.google.com/o/oauth2/auth",
      state_token: "state-xyz",
    }),
    onWebSocketMessage: vi.fn().mockReturnValue(() => {}),
    connectWebSocket: vi.fn().mockResolvedValue(undefined),
    sendWebSocketMessage: vi.fn(),
  });
  mockUseCredentials.mockReturnValue(makeCredentialsReturn());
});

afterEach(() => {
  cleanup();
});

describe("credential connect failures the card reports instead of throwing", () => {
  it("counts a blocked popup whose new-tab fallback was blocked too", async () => {
    mockOpenOAuthPopup.mockReturnValue({
      promise: rejectLater(OAUTH_ERROR_POPUP_BLOCKED_NO_TAB),
      cleanup: { abort: vi.fn() },
      popupBlocked: true,
      fallbackBlocked: true,
    });

    renderCard();

    await waitFor(() =>
      expect(capture).toHaveBeenCalledWith("credential_oauth_popup_blocked", {
        failure_class: "class_05_browser_channel_broken",
        provider: "google",
      }),
    );
  });

  it("counts a flow that ran out of time", async () => {
    mockOpenOAuthPopup.mockReturnValue({
      promise: rejectLater(OAUTH_ERROR_FLOW_TIMED_OUT),
      cleanup: { abort: vi.fn() },
      popupBlocked: false,
      fallbackBlocked: false,
    });

    renderCard();

    await waitFor(() =>
      expect(capture).toHaveBeenCalledWith("credential_oauth_flow_timed_out", {
        failure_class: "class_05_browser_channel_broken",
        provider: "google",
      }),
    );
  });

  it("counts a credential stored with narrower scopes than the card requires", async () => {
    mockUseCredentials.mockReturnValue(
      makeCredentialsReturn({
        oAuthCallback: vi.fn().mockResolvedValue({
          id: "new-cred",
          type: "oauth2",
          title: "Connected Google",
          provider: "google",
          scopes: ["drive.file"],
        }),
      }),
    );
    mockOpenOAuthPopup.mockReturnValue({
      promise: Promise.resolve({ code: "code-1", state: "state-xyz" }),
      cleanup: { abort: vi.fn() },
      popupBlocked: false,
      fallbackBlocked: false,
    });

    renderCard();

    await waitFor(() =>
      expect(capture).toHaveBeenCalledWith(
        "credential_scope_shortfall_blocked_selection",
        {
          failure_class: "class_08_scopes_too_narrow",
          provider: "google",
        },
      ),
    );
  });

  it("counts nothing when the sign-in completes with the scopes it asked for", async () => {
    mockOpenOAuthPopup.mockReturnValue({
      promise: Promise.resolve({ code: "code-1", state: "state-xyz" }),
      cleanup: { abort: vi.fn() },
      popupBlocked: false,
      fallbackBlocked: false,
    });

    renderCard();

    await waitFor(() =>
      expect(
        mockUseCredentials.mock.results[0].value.oAuthCallback,
      ).toHaveBeenCalled(),
    );
    expect(capturedFailures()).toHaveLength(0);
  });
});
