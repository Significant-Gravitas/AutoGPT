import {
  render,
  screen,
  cleanup,
  fireEvent,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
  CredentialsMetaResponse,
} from "@/lib/autogpt-server-api";
import React from "react";
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
  preOpenOAuthPopup: vi.fn(() => null),
  OAUTH_ERROR_WINDOW_CLOSED: "Sign-in window was closed",
  OAUTH_ERROR_FLOW_CANCELED: "OAuth flow was canceled",
  OAUTH_ERROR_FLOW_TIMED_OUT: "OAuth flow timed out",
  OAUTH_ERROR_POPUP_BLOCKED:
    "Popup blocked — the sign-in window opened in a new tab instead. If you don't see it, allow popups for this site and retry.",
}));
vi.mock("@/app/api/__generated__/endpoints/mcp/mcp", () => ({
  postV2InitiateOauthLoginForAnMcpServer: vi.fn(),
}));

import { toast } from "@/components/molecules/Toast/use-toast";
import useCredentials from "@/hooks/useCredentials";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { openOAuthPopup, preOpenOAuthPopup } from "@/lib/oauth-popup";

const mockUseCredentials = useCredentials as unknown as ReturnType<
  typeof vi.fn
>;
const mockUseBackendAPI = useBackendAPI as unknown as ReturnType<typeof vi.fn>;
const mockOpenOAuthPopup = openOAuthPopup as unknown as ReturnType<
  typeof vi.fn
>;
const mockPreOpenOAuthPopup = preOpenOAuthPopup as unknown as ReturnType<
  typeof vi.fn
>;
const mockToast = toast as unknown as ReturnType<typeof vi.fn>;

const baseSchema: BlockIOCredentialsSubSchema = {
  type: "object",
  properties: {},
  credentials_provider: ["codex"],
  credentials_types: ["oauth2"],
  credentials_scopes: ["drive.file", "drive.metadata"],
};

type CredentialsReturn = NonNullable<ReturnType<typeof useCredentials>>;
type BackendAPI = ReturnType<typeof useBackendAPI>;

function makeCredentialsReturn(overrides: Partial<CredentialsReturn> = {}) {
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

// These cover useCredentialsInput's direct OAuth flow (pre-opened popup,
// abort/supersede handling). variant="node" wires the add-credential button
// straight to that flow; the default variant routes through
// ConnectCredentialDialog instead.
describe("CredentialsInput – OAuth flow", () => {
  it("clears a credential retained from a different transport provider", async () => {
    const onSelectCredentials = vi.fn();
    mockUseCredentials.mockReturnValue(
      makeCredentialsReturn({
        provider: "codex",
        providerName: "Codex",
        savedCredentials: [],
      }),
    );

    render(
      <CredentialsInput
        schema={baseSchema}
        selectedCredentials={{
          id: "openai-key",
          provider: "openai",
          type: "api_key",
          title: "OpenAI key",
        }}
        onSelectCredentials={onSelectCredentials}
        showTitle={false}
      />,
    );

    await waitFor(() =>
      expect(onSelectCredentials).toHaveBeenCalledWith(undefined),
    );
  });

  it("shows OpenAI branding while routing ChatGPT sign-in through Codex", async () => {
    const oAuthLoginMock = vi.fn().mockResolvedValue({
      login_url: "https://auth.openai.com/codex/device",
      state_token: "state-codex",
    });
    mockUseBackendAPI.mockReturnValue(
      makeBackendAPI({ oAuthLogin: oAuthLoginMock }),
    );
    mockUseCredentials.mockReturnValue(
      makeCredentialsReturn({
        provider: "codex",
        providerName: "OpenAI",
        schema: {
          ...baseSchema,
          credentials_provider: ["codex"],
          credentials_scopes: [],
        },
      }),
    );
    mockOpenOAuthPopup.mockReturnValue({
      promise: Promise.resolve({ code: "login-id", state: "state-codex" }),
      cleanup: { abort: vi.fn() },
      popupBlocked: false,
      fallbackBlocked: false,
    });

    render(
      <CredentialsInput
        schema={{
          ...baseSchema,
          credentials_provider: ["codex"],
          credentials_scopes: [],
        }}
        onSelectCredentials={vi.fn()}
        showTitle
        variant="node"
      />,
    );

    expect(await screen.findByText("OpenAI credentials")).toBeDefined();
    const signInButton = screen.getByRole("button", {
      name: "Sign in with ChatGPT",
    });
    fireEvent.click(signInButton);

    await waitFor(() => {
      expect(oAuthLoginMock).toHaveBeenCalledWith("codex", [], undefined);
    });
  });

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
        variant="node"
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

  it("pre-opens the window before the oAuthLogin await and passes it to openOAuthPopup", async () => {
    const fakeWindow = { closed: false, close: vi.fn() };
    const callOrder: string[] = [];
    mockPreOpenOAuthPopup.mockImplementation(() => {
      callOrder.push("preOpen");
      return fakeWindow;
    });

    const oAuthLoginMock = vi.fn().mockImplementation(async () => {
      callOrder.push("initiate");
      return {
        login_url: "https://accounts.google.com/o/oauth2/auth",
        state_token: "state-xyz",
      };
    });
    mockUseBackendAPI.mockReturnValue(
      makeBackendAPI({ oAuthLogin: oAuthLoginMock }),
    );
    mockUseCredentials.mockReturnValue(makeCredentialsReturn());
    mockOpenOAuthPopup.mockReturnValue({
      promise: Promise.resolve({ code: "code-2", state: "state-xyz" }),
      cleanup: { abort: vi.fn() },
      popupBlocked: false,
      fallbackBlocked: false,
    });

    render(
      <CredentialsInput
        schema={baseSchema}
        onSelectCredentials={vi.fn()}
        showTitle={false}
        variant="node"
      />,
    );

    fireEvent.click(
      await screen.findByRole("button", { name: /add account/i }),
    );
    await waitFor(() => expect(mockOpenOAuthPopup).toHaveBeenCalled());

    // The window must be opened synchronously, before the login-URL request —
    // after an await iOS Safari blocks every window.open().
    expect(callOrder).toEqual(["preOpen", "initiate"]);
    expect(mockOpenOAuthPopup).toHaveBeenCalledWith(
      "https://accounts.google.com/o/oauth2/auth",
      expect.objectContaining({
        stateToken: "state-xyz",
        preOpenedWindow: fakeWindow,
        useCrossOriginListeners: true,
      }),
    );
  });

  it("shows the blocked-popup toast and modal copy when the popup is blocked", async () => {
    mockPreOpenOAuthPopup.mockReturnValue(null);
    mockUseCredentials.mockReturnValue(makeCredentialsReturn());
    mockOpenOAuthPopup.mockReturnValue({
      promise: new Promise(() => {}), // flow stays in flight
      cleanup: { abort: vi.fn() },
      popupBlocked: true,
      fallbackBlocked: false,
    });

    render(
      <CredentialsInput
        schema={baseSchema}
        onSelectCredentials={vi.fn()}
        showTitle={false}
        variant="node"
      />,
    );

    fireEvent.click(
      await screen.findByRole("button", { name: /add account/i }),
    );

    await waitFor(() =>
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Popup blocked" }),
      ),
    );
    // The waiting modal must direct the user to the fallback tab instead of
    // to a popup that doesn't exist.
    expect(
      await screen.findByText(/blocked the sign-in window/i),
    ).toBeDefined();
  });

  it("closes the pre-opened window when the login-URL request fails", async () => {
    const fakeWindow = { closed: false, close: vi.fn() };
    mockPreOpenOAuthPopup.mockReturnValue(fakeWindow);

    const oAuthLoginMock = vi
      .fn()
      .mockRejectedValue(new Error("provider not configured"));
    mockUseBackendAPI.mockReturnValue(
      makeBackendAPI({ oAuthLogin: oAuthLoginMock }),
    );
    mockUseCredentials.mockReturnValue(makeCredentialsReturn());

    render(
      <CredentialsInput
        schema={baseSchema}
        onSelectCredentials={vi.fn()}
        showTitle={false}
        variant="node"
      />,
    );

    fireEvent.click(
      await screen.findByRole("button", { name: /add account/i }),
    );

    // The failure happens before openOAuthPopup adopts the window, so the
    // flow still owns the dangling about:blank window and must close it.
    await waitFor(() => expect(fakeWindow.close).toHaveBeenCalled());
    expect(mockOpenOAuthPopup).not.toHaveBeenCalled();
  });

  it("closes the pre-opened window and skips adoption when unmounted mid-initiation", async () => {
    const fakeWindow = { closed: false, close: vi.fn() };
    mockPreOpenOAuthPopup.mockReturnValue(fakeWindow);

    let resolveLogin: (value: unknown) => void = () => {};
    const oAuthLoginMock = vi.fn().mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveLogin = resolve;
        }),
    );
    mockUseBackendAPI.mockReturnValue(
      makeBackendAPI({ oAuthLogin: oAuthLoginMock }),
    );
    mockUseCredentials.mockReturnValue(makeCredentialsReturn());
    mockOpenOAuthPopup.mockReturnValue({
      promise: Promise.resolve({ code: "code-2", state: "state-xyz" }),
      cleanup: { abort: vi.fn() },
    });

    const { unmount } = render(
      <CredentialsInput
        schema={baseSchema}
        onSelectCredentials={vi.fn()}
        showTitle={false}
        variant="node"
      />,
    );

    fireEvent.click(
      await screen.findByRole("button", { name: /add account/i }),
    );
    await waitFor(() => expect(oAuthLoginMock).toHaveBeenCalled());

    // Unmount while the login-URL request is still in flight — the cleanup
    // must close the pre-opened window immediately.
    unmount();
    expect(fakeWindow.close).toHaveBeenCalled();

    // When the request resolves, the stale continuation must not adopt the
    // window into a new OAuth popup.
    resolveLogin({
      login_url: "https://accounts.google.com/o/oauth2/auth",
      state_token: "state-xyz",
    });
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(mockOpenOAuthPopup).not.toHaveBeenCalled();
  });

  it("a superseded flow does not clear the newer flow's abort handler", async () => {
    const windowA = { closed: false, close: vi.fn() };
    const windowB = { closed: false, close: vi.fn() };
    mockPreOpenOAuthPopup
      .mockReturnValueOnce(windowA)
      .mockReturnValueOnce(windowB);

    let resolveA: (value: unknown) => void = () => {};
    let resolveB: (value: unknown) => void = () => {};
    const oAuthLoginMock = vi
      .fn()
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveA = resolve;
          }),
      )
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveB = resolve;
          }),
      );
    mockUseBackendAPI.mockReturnValue(
      makeBackendAPI({ oAuthLogin: oAuthLoginMock }),
    );
    mockUseCredentials.mockReturnValue(makeCredentialsReturn());

    const abortB = vi.fn();
    mockOpenOAuthPopup.mockReturnValue({
      promise: new Promise(() => {}), // flow stays in flight
      cleanup: { abort: abortB },
    });

    const { unmount } = render(
      <CredentialsInput
        schema={baseSchema}
        onSelectCredentials={vi.fn()}
        showTitle={false}
        variant="node"
      />,
    );

    const addAccountButton = await screen.findByRole("button", {
      name: /add account/i,
    });

    // Flow A starts, then flow B supersedes it while A's request is pending.
    fireEvent.click(addAccountButton);
    await waitFor(() => expect(oAuthLoginMock).toHaveBeenCalledTimes(1));
    fireEvent.click(addAccountButton);
    await waitFor(() => expect(oAuthLoginMock).toHaveBeenCalledTimes(2));

    // Starting B closes A's still-pending pre-opened window.
    expect(windowA.close).toHaveBeenCalled();

    // B's request resolves first and registers its abort handler.
    resolveB({
      login_url: "https://accounts.google.com/o/oauth2/auth",
      state_token: "state-xyz",
    });
    await waitFor(() => expect(mockOpenOAuthPopup).toHaveBeenCalledTimes(1));

    // A's request resolves late — its continuation must bail without
    // nulling B's abort handler in its finally block.
    resolveA({
      login_url: "https://accounts.google.com/o/oauth2/auth",
      state_token: "state-xyz",
    });
    await new Promise((resolve) => setTimeout(resolve, 0));

    // Unmount must still reach B's abort — i.e. A's finally did not clear it.
    unmount();
    expect(abortB).toHaveBeenCalled();
  });
});

describe("CredentialsInput – device auth", () => {
  const deviceSchema = {
    credentials_provider: ["stripe_link"],
    // Both: a device-code grant yields an oauth2-shaped credential, so the
    // block accepts oauth2 while device_code is how it is obtained.
    credentials_types: ["oauth2", "device_code"],
    credentials_scopes: ["userinfo:read"],
  } as unknown as BlockIOCredentialsSubSchema;

  function renderDeviceInput() {
    mockUseCredentials.mockReturnValue(
      makeCredentialsReturn({
        provider: "stripe_link",
        providerName: "Stripe Link",
        schema: deviceSchema,
        savedCredentials: [],
        supportsOAuth2: false,
        supportsDeviceCode: true,
      } as any),
    );
    return render(
      <CredentialsInput
        schema={deviceSchema}
        selectedCredentials={undefined}
        onSelectCredentials={vi.fn()}
      />,
    );
  }

  // The regression: this path had no device_code branch, so "Add account"
  // fell through to OAuth and the backend answered "Provider 'stripe_link'
  // does not support OAuth" — the blocks were unconnectable from the builder.
  it("opens the device auth flow instead of an OAuth popup", async () => {
    renderDeviceInput();

    fireEvent.click(screen.getByText("Add account"));

    await waitFor(() => {
      expect(screen.getByText(/device authorization/i)).toBeTruthy();
    });
    expect(mockOpenOAuthPopup).not.toHaveBeenCalled();
    expect(mockPreOpenOAuthPopup).not.toHaveBeenCalled();
  });
});
function StatefulCredentialsInput({
  initial,
  onSelectionChange,
}: {
  initial?: CredentialsMetaInput;
  onSelectionChange?: (credential?: CredentialsMetaInput) => void;
}) {
  const [selected, setSelected] = React.useState<
    CredentialsMetaInput | undefined
  >(initial);

  function handleSelectionChange(credential?: CredentialsMetaInput) {
    setSelected(credential);
    onSelectionChange?.(credential);
  }

  return (
    <CredentialsInput
      schema={baseSchema}
      selectedCredentials={selected}
      onSelectCredentials={handleSelectionChange}
      showTitle={false}
    />
  );
}

describe("CredentialsInput – a removed connection", () => {
  const codexCredential = {
    id: "codex-1",
    provider: "codex",
    type: "oauth2" as const,
    title: "ChatGPT for Codex",
    scopes: [],
  };

  const deleted = {
    id: "deleted-cred",
    provider: "codex",
    type: "oauth2" as const,
    title: "Old ChatGPT connection",
  };

  function mockProvider(
    savedCredentials: CredentialsMetaResponse[],
    allProviderCredentials: CredentialsMetaResponse[] = savedCredentials,
  ) {
    mockUseCredentials.mockReturnValue(
      makeCredentialsReturn({
        provider: "codex",
        providerName: "Codex",
        schema: baseSchema,
        savedCredentials,
        allProviderCredentials,
      }),
    );
  }

  it("heals the node and reports the swap", async () => {
    // Healing is the point: reconnecting mints a new id, and adopting it is
    // what makes disconnect/reconnect just work. Refusing to adopt would force
    // a manual re-pick in the common case (same account) to warn about the
    // rare one (a different account). So heal — and say so.
    mockProvider([codexCredential]);

    render(<StatefulCredentialsInput initial={deleted} />);

    expect(
      await screen.findByText(
        /Old ChatGPT connection was removed — now using ChatGPT for Codex\./i,
      ),
    ).toBeDefined();
  });

  it("asks the user to choose when nothing is left to heal with", async () => {
    // Deleting your only connection: nothing to adopt, so say what happened
    // and ask. Previously an empty list was read as "still loading", so this
    // case was skipped entirely — a stale selection with no warning.
    mockProvider([]);

    render(<StatefulCredentialsInput initial={deleted} />);

    expect(
      await screen.findByText(
        /Old ChatGPT connection was removed\. Choose a connection/i,
      ),
    ).toBeDefined();
  });

  it("does not announce an external removal while deleting the selected credential", async () => {
    const rerenderRef: {
      current?: ReturnType<typeof render>["rerender"];
    } = {};
    let resolveDelete:
      | ((result: { deleted: true; revoked: null }) => void)
      | undefined;
    const deleteCredentials = vi.fn(async () => {
      mockProvider([]);
      if (!rerenderRef.current)
        throw new Error("expected the input to be rendered");
      rerenderRef.current(
        <StatefulCredentialsInput
          initial={{ ...codexCredential, type: "oauth2" }}
        />,
      );

      return new Promise<{ deleted: true; revoked: null }>((resolve) => {
        resolveDelete = resolve;
      });
    });
    mockUseCredentials.mockReturnValue(
      makeCredentialsReturn({
        provider: "codex",
        providerName: "Codex",
        schema: baseSchema,
        savedCredentials: [codexCredential],
        allProviderCredentials: [codexCredential],
        deleteCredentials,
      }),
    );

    const view = render(
      <StatefulCredentialsInput
        initial={{ ...codexCredential, type: "oauth2" }}
      />,
    );
    rerenderRef.current = view.rerender;

    const menuTrigger = screen
      .getAllByRole("button")
      .find((button) => button.getAttribute("aria-haspopup") === "menu");
    if (!menuTrigger) throw new Error("expected the credential actions menu");
    fireEvent.pointerDown(menuTrigger, { button: 0 });
    fireEvent.click(await screen.findByRole("menuitem", { name: "Delete" }));
    fireEvent.click(await screen.findByRole("button", { name: "Delete" }));

    await waitFor(() => expect(deleteCredentials).toHaveBeenCalled());
    await waitFor(() => expect(screen.queryByText(/was removed/i)).toBeNull());

    if (!resolveDelete) throw new Error("expected the deletion to be pending");
    resolveDelete({ deleted: true, revoked: null });
    await waitFor(() => expect(screen.queryByText(/was removed/i)).toBeNull());
  });

  it("stays quiet while the provider list is still loading", async () => {
    mockUseCredentials.mockReturnValue(null);

    const view = render(<StatefulCredentialsInput initial={deleted} />);

    await waitFor(() => expect(screen.queryByText(/was removed/i)).toBeNull());

    mockProvider([codexCredential]);
    view.rerender(<StatefulCredentialsInput initial={deleted} />);

    expect(
      await screen.findByText(
        /Old ChatGPT connection was removed — now using ChatGPT for Codex/i,
      ),
    ).toBeDefined();
  });

  it("says nothing when the configured connection still resolves", async () => {
    mockProvider([codexCredential]);

    render(
      <StatefulCredentialsInput
        initial={{ ...codexCredential, type: "oauth2" }}
      />,
    );

    await waitFor(() => expect(screen.queryByText(/was removed/i)).toBeNull());
  });

  it("clears an existing credential when filtering excludes it", async () => {
    const onSelectionChange = vi.fn();
    mockProvider([], [{ ...codexCredential }]);

    render(
      <StatefulCredentialsInput
        initial={{ ...codexCredential, type: "oauth2" }}
        onSelectionChange={onSelectionChange}
      />,
    );

    await waitFor(() =>
      expect(onSelectionChange).toHaveBeenCalledWith(undefined),
    );
    await waitFor(() => expect(screen.queryByText(/was removed/i)).toBeNull());
  });

  it("heals again after an auto-selected connection is itself deleted", async () => {
    // The auto-select flag latches after the first run, so without resetting it
    // the field stayed empty instead of adopting whatever remained.
    const second = {
      ...codexCredential,
      id: "codex-2",
      title: "Second ChatGPT",
    };
    mockProvider([second], [second]);

    render(
      <StatefulCredentialsInput
        initial={{ ...codexCredential, type: "oauth2" }}
      />,
    );

    expect(await screen.findByText(/now using Second ChatGPT/i)).toBeDefined();
  });

  it("still auto-selects a lone connection when nothing was configured before", async () => {
    const onSelectionChange = vi.fn();
    mockProvider([codexCredential]);

    render(<StatefulCredentialsInput onSelectionChange={onSelectionChange} />);

    await waitFor(() =>
      expect(onSelectionChange).toHaveBeenCalledWith({
        id: codexCredential.id,
        provider: codexCredential.provider,
        title: codexCredential.title,
        type: codexCredential.type,
      }),
    );
    await waitFor(() => expect(screen.queryByText(/was removed/i)).toBeNull());
  });

  it("dismisses the notice after an explicit dropdown selection", async () => {
    const replacement = {
      ...codexCredential,
      id: "codex-2",
      title: "Replacement ChatGPT",
    };
    const onSelectionChange = vi.fn();
    mockProvider([codexCredential, replacement]);

    render(
      <StatefulCredentialsInput
        initial={deleted}
        onSelectionChange={onSelectionChange}
      />,
    );

    expect(
      await screen.findByText(
        /Old ChatGPT connection was removed\. Choose a connection/i,
      ),
    ).toBeDefined();

    fireEvent.change(
      screen.getByRole("combobox", { name: /Select OpenAI credential/i }),
      { target: { value: replacement.id } },
    );

    await waitFor(() =>
      expect(onSelectionChange).toHaveBeenCalledWith({
        id: replacement.id,
        provider: replacement.provider,
        title: replacement.title,
        type: replacement.type,
      }),
    );
    await waitFor(() => expect(screen.queryByText(/was removed/i)).toBeNull());
  });

  it("names an adopted title-less credential with the existing fallback", async () => {
    mockProvider([
      {
        ...codexCredential,
        title: undefined,
      },
    ]);

    render(<StatefulCredentialsInput initial={deleted} />);

    expect(
      await screen.findByText(/now using Your OpenAI account\./i),
    ).toBeDefined();
  });

  it("dismisses the notice once OAuth reconnects the account", async () => {
    // Reconnecting is an explicit choice, same as picking from the dropdown,
    // so the notice has done its job. Only `handleCredentialSelect` used to
    // clear it, which left the warning up after the user had already fixed
    // the thing it was warning about.
    // The provider list starts empty and gains the connection when the
    // callback resolves, which is what really happens: `oAuthCallback` mints
    // the credential and refreshes the list.
    const reconnected = {
      id: "new-cred",
      provider: "codex",
      type: "oauth2" as const,
      title: "Reconnected ChatGPT",
      scopes: ["drive.file", "drive.metadata"],
    };
    const list: CredentialsMetaResponse[] = [];
    mockUseCredentials.mockReturnValue(
      makeCredentialsReturn({
        provider: "codex",
        providerName: "Codex",
        schema: baseSchema,
        savedCredentials: list,
        allProviderCredentials: list,
        oAuthCallback: vi.fn().mockImplementation(async () => {
          list.push(reconnected);
          return reconnected;
        }),
      }),
    );
    mockOpenOAuthPopup.mockReturnValue({
      promise: Promise.resolve({ code: "code-1", state: "state-xyz" }),
      cleanup: { abort: vi.fn() },
    });

    render(<StatefulCredentialsInput initial={deleted} />);

    expect(
      await screen.findByText(
        /Old ChatGPT connection was removed\. Choose a connection/i,
      ),
    ).toBeDefined();

    fireEvent.click(
      screen.getByRole("button", { name: /sign in with chatgpt/i }),
    );

    await waitFor(() => expect(screen.queryByText(/was removed/i)).toBeNull());
  });
});
