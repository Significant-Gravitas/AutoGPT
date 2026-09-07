import type { ExistingCredentialsOffer } from "@/components/contextual/CredentialsInput/components/ConnectCredentialDialog/helpers";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ConnectorRow } from "../ConnectorRow";
import type { ConnectorRow as Row } from "../helpers";

const mockGrant = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/experts/experts", () => ({
  useGrantExpertCredentials: () => ({
    mutateAsync: mockGrant,
    isPending: false,
  }),
}));

vi.mock(
  "@/components/contextual/CredentialsInput/components/ConnectCredentialDialog/ConnectCredentialDialog",
  () => ({
    ConnectCredentialDialog: ({
      open,
      existing,
      onClose,
      onConnected,
    }: {
      open: boolean;
      existing?: ExistingCredentialsOffer;
      onClose: () => void;
      onConnected?: () => void;
    }) =>
      open ? (
        <div data-testid="connect-dialog">
          {existing?.credentials.map((credential) => (
            <button
              key={credential.id}
              onClick={() => void existing.onUse(credential)}
            >
              {`use-${credential.title}`}
            </button>
          ))}
          {existing?.error && <span>{existing.error}</span>}
          <button
            onClick={() => {
              onConnected?.();
              onClose();
            }}
          >
            finish sign-in
          </button>
        </div>
      ) : null,
  }),
);

afterEach(() => {
  cleanup();
  mockGrant.mockReset();
});

const savedGithub = {
  id: "spare-cred",
  provider: "github",
  type: "oauth2",
  title: "GH spare",
  scopes: [],
};

function providersWithGithub(saved: (typeof savedGithub)[] = [savedGithub]) {
  return {
    github: {
      provider: "github",
      providerName: "GitHub",
      savedCredentials: saved,
      oAuthCallback: vi.fn(),
      createAPIKeyCredentials: vi.fn(),
      createUserPasswordCredentials: vi.fn(),
      createHostScopedCredentials: vi.fn(),
      deleteCredentials: vi.fn(),
    },
  } as unknown as React.ContextType<typeof CredentialsProvidersContext>;
}

function row(overrides: Partial<Row> = {}): Row {
  return {
    provider: "github",
    displayName: "GitHub",
    description: "Connect your GitHub account",
    schema: { credentials_provider: ["github"], credentials_types: ["oauth2"] },
    selected: undefined,
    select: vi.fn(),
    onConnected: vi.fn(),
    ...overrides,
  };
}

describe("ConnectorRow in an expert chat", () => {
  it("does not treat the account's credential as connected", () => {
    const current = row({
      expertGrant: { expertId: "expert-a", credentials: [] },
    });
    render(
      <CredentialsProvidersContext.Provider value={providersWithGithub()}>
        <ConnectorRow row={current} />
      </CredentialsProvidersContext.Provider>,
    );
    expect(current.select).not.toHaveBeenCalled();
    expect(screen.queryByText("Connected")).toBeNull();
    expect(screen.getByText("This expert needs its own access")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));
    expect(screen.queryByRole("button", { name: /^use-/ })).toBeNull();
  });

  it("offers the account's credential in the dialog and grants it", async () => {
    mockGrant.mockResolvedValue([]);
    const current = row({
      expertGrant: {
        expertId: "expert-a",
        credentials: [{ id: "spare-cred", title: "GH spare", type: "oauth2" }],
      },
    });
    render(
      <CredentialsProvidersContext.Provider value={providersWithGithub()}>
        <ConnectorRow row={current} />
      </CredentialsProvidersContext.Provider>,
    );
    expect(screen.getByText("Needs this expert's access")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));
    fireEvent.click(screen.getByRole("button", { name: "use-GH spare" }));
    await waitFor(() =>
      expect(mockGrant).toHaveBeenCalledWith({
        expertId: "expert-a",
        data: { credential_ids: ["spare-cred"] },
      }),
    );
    expect(current.select).toHaveBeenCalledWith(
      expect.objectContaining({ id: "spare-cred", provider: "github" }),
    );
    expect(current.onConnected).toHaveBeenCalled();
  });

  it("lists every grantable account and still lets the user connect a new one", () => {
    const current = row({
      expertGrant: {
        expertId: "expert-a",
        credentials: [
          { id: "cred-1", title: "Work GitHub", type: "oauth2" },
          { id: "cred-2", title: "Personal GitHub", type: "oauth2" },
        ],
      },
    });
    render(
      <CredentialsProvidersContext.Provider value={providersWithGithub()}>
        <ConnectorRow row={current} />
      </CredentialsProvidersContext.Provider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));
    expect(
      screen.getByRole("button", { name: "use-Work GitHub" }),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: "use-Personal GitHub" }),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: "finish sign-in" }),
    ).toBeDefined();
  });

  it("shows a retryable error when the grant fails", async () => {
    mockGrant.mockRejectedValue(new Error("nope"));
    const current = row({
      expertGrant: {
        expertId: "expert-a",
        credentials: [{ id: "spare-cred", title: "GH spare", type: "oauth2" }],
      },
    });
    render(
      <CredentialsProvidersContext.Provider value={providersWithGithub()}>
        <ConnectorRow row={current} />
      </CredentialsProvidersContext.Provider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));
    fireEvent.click(screen.getByRole("button", { name: "use-GH spare" }));
    expect(
      await screen.findByText("Couldn't grant access. Try again."),
    ).toBeDefined();
    expect(screen.getByTestId("connect-dialog")).toBeDefined();
    expect(current.onConnected).not.toHaveBeenCalled();
  });
});

describe("ConnectorRow after a sign-in in an expert chat", () => {
  const existing = { ...savedGithub, id: "old-cred", title: "Old GitHub" };
  const added = { ...savedGithub, id: "new-cred", title: "New GitHub" };

  it("grants the account that appeared after Connect, not a pre-existing one", async () => {
    mockGrant.mockResolvedValue([]);
    const current = row({
      expertGrant: { expertId: "expert-a", credentials: [] },
    });
    const { rerender } = render(
      <CredentialsProvidersContext.Provider
        value={providersWithGithub([existing])}
      >
        <ConnectorRow row={current} />
      </CredentialsProvidersContext.Provider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));
    fireEvent.click(screen.getByRole("button", { name: "finish sign-in" }));
    rerender(
      <CredentialsProvidersContext.Provider
        value={providersWithGithub([existing, added])}
      >
        <ConnectorRow row={current} />
      </CredentialsProvidersContext.Provider>,
    );
    await waitFor(() =>
      expect(mockGrant).toHaveBeenCalledWith({
        expertId: "expert-a",
        data: { credential_ids: ["new-cred"] },
      }),
    );
    expect(mockGrant).toHaveBeenCalledTimes(1);
  });

  it("keeps the new account available for a retry when the grant fails", async () => {
    mockGrant.mockRejectedValueOnce(new Error("nope")).mockResolvedValue([]);
    const current = row({
      expertGrant: { expertId: "expert-a", credentials: [] },
    });
    const { rerender } = render(
      <CredentialsProvidersContext.Provider value={providersWithGithub([])}>
        <ConnectorRow row={current} />
      </CredentialsProvidersContext.Provider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));
    fireEvent.click(screen.getByRole("button", { name: "finish sign-in" }));
    rerender(
      <CredentialsProvidersContext.Provider
        value={providersWithGithub([added])}
      >
        <ConnectorRow row={current} />
      </CredentialsProvidersContext.Provider>,
    );
    expect(
      await screen.findByText("Couldn't grant access. Try again."),
    ).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));
    fireEvent.click(screen.getByRole("button", { name: "use-New GitHub" }));
    await waitFor(() => expect(mockGrant).toHaveBeenCalledTimes(2));
    expect(current.onConnected).toHaveBeenCalledTimes(1);
  });
});

describe("ConnectorRow in a personal chat", () => {
  it("still auto-selects the account's credential", async () => {
    const current = row();
    render(
      <CredentialsProvidersContext.Provider value={providersWithGithub()}>
        <ConnectorRow row={current} />
      </CredentialsProvidersContext.Provider>,
    );
    await waitFor(() =>
      expect(current.select).toHaveBeenCalledWith(
        expect.objectContaining({ id: "spare-cred" }),
      ),
    );
  });
});
