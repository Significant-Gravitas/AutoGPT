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
    ConnectCredentialDialog: ({ open }: { open: boolean }) =>
      open ? <div data-testid="connect-dialog" /> : null,
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

function providersWithGithub() {
  return {
    github: {
      provider: "github",
      providerName: "GitHub",
      savedCredentials: [savedGithub],
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
    expect(screen.getByRole("button", { name: "Connect" })).toBeDefined();
  });

  it("offers Grant access for a credential the account already has", async () => {
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
    expect(screen.getByText(/this expert needs access to it/)).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Grant access" }));
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

  it("lets the user pick among several grantable accounts and still connect a new one", () => {
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
    expect(screen.getByText("Work GitHub")).toBeDefined();
    expect(screen.getByRole("button", { name: "Grant access" })).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Connect another" }));
    expect(screen.getByTestId("connect-dialog")).toBeDefined();
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
    fireEvent.click(screen.getByRole("button", { name: "Grant access" }));
    expect(
      await screen.findByText("Couldn't grant access. Try again."),
    ).toBeDefined();
    expect(current.onConnected).not.toHaveBeenCalled();
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
