import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import type { ExistingCredentialsOffer } from "@/components/contextual/CredentialsInput/components/ConnectCredentialDialog/helpers";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ConnectorRow } from "../ConnectorRow";
import type { ConnectorRow as Row } from "../helpers";

const mockGrant = vi.fn();
// What the mocked dialog reports as the credential a sign-in produced.
const mockReported: { current?: CredentialsMetaResponse } = {};
const mockGrants = vi.fn((_expertId?: string, _options?: unknown) =>
  grantsResult(),
);
vi.mock("@/app/api/__generated__/endpoints/experts/experts", () => ({
  useListExpertCredentials: (expertId: string, options?: unknown) =>
    mockGrants(expertId, options),
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
      onConnected?: (credential?: CredentialsMetaResponse) => void;
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
              onConnected?.(mockReported.current);
              onClose();
            }}
          >
            finish sign-in
          </button>
        </div>
      ) : null,
  }),
);

interface GrantsResult {
  data?: { credential_id: string }[];
  isPending: boolean;
  isError: boolean;
  refetch: () => Promise<{ data?: { credential_id: string }[] } | undefined>;
}

/** Credential ids the grant mutation was asked for, i.e. what the server
 *  would report once a successful grant is refetched. */
function grantedByMutation() {
  return mockGrant.mock.calls.flatMap(
    (call) =>
      (call[0] as { data: { credential_ids: string[] } }).data.credential_ids,
  );
}

function grantsResult(
  ids: string[] = [],
  overrides: Partial<GrantsResult> = {},
): GrantsResult {
  return {
    data: ids.map((id) => ({ credential_id: id })),
    isPending: false,
    isError: false,
    refetch: vi.fn(async () => ({
      data: [...ids, ...grantedByMutation()].map((id) => ({
        credential_id: id,
      })),
    })),
    ...overrides,
  };
}

afterEach(() => {
  cleanup();
  mockGrant.mockReset();
  mockReported.current = undefined;
  mockGrants.mockReturnValue(grantsResult());
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

/** The card rebuilds the row from its own state on every render, so a row
 *  that only hydrates once needs its selection fed back in. */
function StatefulRow({ current }: { current: Row }) {
  const [selected, setSelected] = useState(current.selected);
  return (
    <ConnectorRow
      row={{
        ...current,
        selected,
        select: (value) => {
          current.select(value);
          setSelected(value);
        },
      }}
    />
  );
}

describe("ConnectorRow in an expert chat", () => {
  it("restores a persisted expert grant without triggering another run", async () => {
    mockGrants.mockReturnValue(grantsResult(["spare-cred"]));
    const current = row({
      expertGrant: { expertId: "expert-a", credentials: [] },
    });
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
    expect(mockGrant).not.toHaveBeenCalled();
    expect(current.onConnected).not.toHaveBeenCalled();
  });

  it("shows Granted once the expert holds the selected credential", async () => {
    mockGrants.mockReturnValue(grantsResult(["spare-cred"]));
    const current = row({
      expertGrant: { expertId: "expert-a", credentials: [] },
    });
    render(
      <CredentialsProvidersContext.Provider value={providersWithGithub()}>
        <StatefulRow current={current} />
      </CredentialsProvidersContext.Provider>,
    );
    expect(await screen.findByText("Granted")).toBeDefined();
  });

  it("hydrates the first granted account when several of them match", async () => {
    mockGrants.mockReturnValue(grantsResult(["other-cred", "spare-cred"]));
    const current = row({
      expertGrant: { expertId: "expert-a", credentials: [] },
    });
    render(
      <CredentialsProvidersContext.Provider
        value={providersWithGithub([
          savedGithub,
          { ...savedGithub, id: "other-cred" },
        ])}
      >
        <StatefulRow current={current} />
      </CredentialsProvidersContext.Provider>,
    );
    expect(await screen.findByText("Granted")).toBeDefined();
    expect(current.select).toHaveBeenCalledWith(
      expect.objectContaining({ id: "other-cred" }),
    );
  });

  it("does not show Granted while the grant list is unavailable", () => {
    mockGrants.mockReturnValue(
      grantsResult([], { data: undefined, isError: true }),
    );
    const current = row({
      expertGrant: { expertId: "expert-a", credentials: [] },
      selected: { id: "spare-cred", provider: "github", type: "oauth2" },
    });
    render(
      <CredentialsProvidersContext.Provider value={providersWithGithub()}>
        <ConnectorRow row={current} />
      </CredentialsProvidersContext.Provider>,
    );
    expect(screen.queryByText("Granted")).toBeNull();
    expect(screen.getByRole("button", { name: "Connect" })).toBeDefined();
    expect(current.onConnected).not.toHaveBeenCalled();
  });

  it("drops a personal-mode selection when a later card makes the row an expert's", async () => {
    const current = row();
    const { rerender } = render(
      <CredentialsProvidersContext.Provider value={providersWithGithub()}>
        <StatefulRow current={current} />
      </CredentialsProvidersContext.Provider>,
    );
    expect(await screen.findByText("Connected")).toBeDefined();
    rerender(
      <CredentialsProvidersContext.Provider value={providersWithGithub()}>
        <StatefulRow
          current={{
            ...current,
            expertGrant: { expertId: "expert-a", credentials: [] },
          }}
        />
      </CredentialsProvidersContext.Provider>,
    );
    expect(
      await screen.findByRole("button", { name: "Connect" }),
    ).toBeDefined();
    expect(screen.queryByText("Connected")).toBeNull();
    expect(screen.queryByText("Granted")).toBeNull();
  });

  it("clears a selected credential after the grant is revoked", async () => {
    const current = row({
      expertGrant: { expertId: "expert-a", credentials: [] },
      selected: { id: "spare-cred", provider: "github", type: "oauth2" },
    });
    render(
      <CredentialsProvidersContext.Provider value={providersWithGithub()}>
        <ConnectorRow row={current} />
      </CredentialsProvidersContext.Provider>,
    );
    await waitFor(() => expect(current.select).toHaveBeenCalledWith(undefined));
  });

  it("preserves the chosen account when multiple matching accounts are granted", () => {
    mockGrants.mockReturnValue(grantsResult(["spare-cred", "other-cred"]));
    const current = row({
      expertGrant: { expertId: "expert-a", credentials: [] },
      selected: { id: "other-cred", provider: "github", type: "oauth2" },
    });
    render(
      <CredentialsProvidersContext.Provider
        value={providersWithGithub([
          savedGithub,
          { ...savedGithub, id: "other-cred" },
        ])}
      >
        <ConnectorRow row={current} />
      </CredentialsProvidersContext.Provider>,
    );
    expect(current.select).not.toHaveBeenCalled();
  });

  it("does not hydrate a granted credential lacking the required scopes", () => {
    mockGrants.mockReturnValue(grantsResult(["spare-cred"]));
    const current = row({
      expertGrant: { expertId: "expert-a", credentials: [] },
      schema: {
        credentials_provider: ["github"],
        credentials_types: ["oauth2"],
        credentials_scopes: ["repo"],
      },
    });
    render(
      <CredentialsProvidersContext.Provider value={providersWithGithub()}>
        <ConnectorRow row={current} />
      </CredentialsProvidersContext.Provider>,
    );
    expect(current.select).not.toHaveBeenCalled();
  });

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
    fireEvent.click(screen.getByRole("button", { name: "Grant access" }));
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

  it("grants before telling the chain the row connected", async () => {
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
    fireEvent.click(screen.getByRole("button", { name: "Grant access" }));
    fireEvent.click(screen.getByRole("button", { name: "use-GH spare" }));
    await waitFor(() => expect(current.onConnected).toHaveBeenCalled());
    expect(mockGrant.mock.invocationCallOrder[0]).toBeLessThan(
      vi.mocked(current.onConnected).mock.invocationCallOrder[0],
    );
  });

  it("does not connect the row when the refreshed list lacks the grant", async () => {
    mockGrant.mockResolvedValue([]);
    mockGrants.mockReturnValue(
      grantsResult([], { refetch: vi.fn(async () => ({ data: [] })) }),
    );
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
    fireEvent.click(screen.getByRole("button", { name: "use-GH spare" }));
    expect(
      await screen.findByText("Couldn't grant access. Try again."),
    ).toBeDefined();
    expect(current.select).not.toHaveBeenCalled();
    expect(current.onConnected).not.toHaveBeenCalled();
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
    fireEvent.click(screen.getByRole("button", { name: "Grant access" }));
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
    fireEvent.click(screen.getByRole("button", { name: "Grant access" }));
    fireEvent.click(screen.getByRole("button", { name: "use-GH spare" }));
    expect(
      await screen.findByText("Couldn't grant access. Try again."),
    ).toBeDefined();
    expect(screen.getByTestId("connect-dialog")).toBeDefined();
    expect(current.onConnected).not.toHaveBeenCalled();
  });

  it("surfaces the reason the backend refused the grant", async () => {
    const { ApiError } = await import("@/lib/autogpt-server-api/helpers");
    mockGrant.mockRejectedValue(
      new ApiError("Expert is out of credentials seats", 403, {}),
    );
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
    fireEvent.click(screen.getByRole("button", { name: "use-GH spare" }));
    expect(
      await screen.findByText("Expert is out of credentials seats"),
    ).toBeDefined();
  });
});

describe("ConnectorRow after a sign-in in an expert chat", () => {
  const existing = { ...savedGithub, id: "old-cred", title: "Old GitHub" };
  const added = { ...savedGithub, id: "new-cred", title: "New GitHub" };

  function reported(credential: typeof savedGithub) {
    return { ...credential, username: null } as CredentialsMetaResponse;
  }

  it("grants the credential the sign-in reports without waiting for a refresh", async () => {
    mockGrant.mockResolvedValue([]);
    mockReported.current = reported(added);
    const current = row({
      expertGrant: { expertId: "expert-a", credentials: [] },
    });
    render(
      <CredentialsProvidersContext.Provider
        value={providersWithGithub([existing])}
      >
        <ConnectorRow row={current} />
      </CredentialsProvidersContext.Provider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));
    fireEvent.click(screen.getByRole("button", { name: "finish sign-in" }));
    await waitFor(() =>
      expect(mockGrant).toHaveBeenCalledWith({
        expertId: "expert-a",
        data: { credential_ids: ["new-cred"] },
      }),
    );
    expect(mockGrant).toHaveBeenCalledTimes(1);
    expect(current.onConnected).toHaveBeenCalledTimes(1);
  });

  it("grants a reported credential when the provider has no accounts yet", async () => {
    mockGrant.mockResolvedValue([]);
    mockReported.current = reported(added);
    const current = row({
      expertGrant: { expertId: "expert-a", credentials: [] },
    });
    render(
      <CredentialsProvidersContext.Provider value={{}}>
        <ConnectorRow row={current} />
      </CredentialsProvidersContext.Provider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));
    fireEvent.click(screen.getByRole("button", { name: "finish sign-in" }));
    await waitFor(() =>
      expect(mockGrant).toHaveBeenCalledWith({
        expertId: "expert-a",
        data: { credential_ids: ["new-cred"] },
      }),
    );
    expect(current.onConnected).toHaveBeenCalledTimes(1);
  });

  it("grants a re-authenticated account that kept its id", async () => {
    mockGrant.mockResolvedValue([]);
    mockReported.current = reported(existing);
    const current = row({
      expertGrant: { expertId: "expert-a", credentials: [] },
    });
    render(
      <CredentialsProvidersContext.Provider
        value={providersWithGithub([existing])}
      >
        <ConnectorRow row={current} />
      </CredentialsProvidersContext.Provider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));
    fireEvent.click(screen.getByRole("button", { name: "finish sign-in" }));
    await waitFor(() =>
      expect(mockGrant).toHaveBeenCalledWith({
        expertId: "expert-a",
        data: { credential_ids: ["old-cred"] },
      }),
    );
  });

  it("reports an account that lacks the required scopes instead of granting it", async () => {
    mockReported.current = reported(added);
    const current = row({
      expertGrant: { expertId: "expert-a", credentials: [] },
      schema: {
        credentials_provider: ["github"],
        credentials_types: ["oauth2"],
        credentials_scopes: ["repo"],
      },
    });
    render(
      <CredentialsProvidersContext.Provider
        value={providersWithGithub([existing])}
      >
        <ConnectorRow row={current} />
      </CredentialsProvidersContext.Provider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));
    fireEvent.click(screen.getByRole("button", { name: "finish sign-in" }));
    expect(
      await screen.findByText(
        "That account is missing the access this needs. Try connecting again.",
      ),
    ).toBeDefined();
    expect(mockGrant).not.toHaveBeenCalled();
    expect(current.select).not.toHaveBeenCalled();
  });

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

  it("keeps Connect disabled until the accounts have loaded", async () => {
    const current = row({
      expertGrant: { expertId: "expert-a", credentials: [] },
    });
    const { rerender } = render(
      <CredentialsProvidersContext.Provider value={null}>
        <ConnectorRow row={current} />
      </CredentialsProvidersContext.Provider>,
    );
    const connect = screen.getByRole("button", { name: "Connect" });
    expect(connect.hasAttribute("disabled")).toBe(true);
    fireEvent.click(connect);
    expect(screen.queryByRole("button", { name: "finish sign-in" })).toBeNull();
    rerender(
      <CredentialsProvidersContext.Provider
        value={providersWithGithub([existing])}
      >
        <ConnectorRow row={current} />
      </CredentialsProvidersContext.Provider>,
    );
    await waitFor(() =>
      expect(
        screen
          .getByRole("button", { name: "Connect" })
          .hasAttribute("disabled"),
      ).toBe(false),
    );
    expect(mockGrant).not.toHaveBeenCalled();
    expect(current.select).not.toHaveBeenCalled();
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
    fireEvent.click(screen.getByRole("button", { name: "Grant access" }));
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

  it("asks for no grants at all", async () => {
    const current = row();
    render(
      <CredentialsProvidersContext.Provider value={providersWithGithub()}>
        <StatefulRow current={current} />
      </CredentialsProvidersContext.Provider>,
    );
    expect(await screen.findByText("Connected")).toBeDefined();
    expect(mockGrant).not.toHaveBeenCalled();
    expect(mockGrants).toHaveBeenCalledWith(
      "",
      expect.objectContaining({
        query: expect.objectContaining({ enabled: false }),
      }),
    );
  });
});
