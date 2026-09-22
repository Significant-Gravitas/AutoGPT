import type { ExistingCredentialsOffer } from "@/components/contextual/CredentialsInput/components/ConnectCredentialDialog/helpers";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useState, type ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ConnectorRow } from "../ConnectorRow";
import type { ConnectorRow as Row } from "../helpers";

vi.mock("@/app/api/__generated__/endpoints/experts/experts", () => ({
  useListExpertCredentials: () => ({
    data: undefined,
    isPending: false,
    isError: false,
    isFetching: false,
    refetch: vi.fn(),
  }),
  useGrantExpertCredentials: () => ({ mutateAsync: vi.fn(), isPending: false }),
  getListExpertCredentialsQueryKey: (expertId?: string) => [
    `/api/experts/${expertId}/credentials`,
  ],
}));

vi.mock(
  "@/components/contextual/CredentialsInput/components/ConnectCredentialDialog/ConnectCredentialDialog",
  () => ({
    ConnectCredentialDialog: ({
      open,
      existing,
    }: {
      open: boolean;
      existing?: ExistingCredentialsOffer;
    }) =>
      open ? (
        <div data-testid="connect-dialog">
          {existing?.purpose && <span>{`purpose-${existing.purpose}`}</span>}
          {existing?.credentials.map((credential) => (
            <button
              key={credential.id}
              onClick={() => void existing.onUse(credential)}
            >
              {`use-${credential.title}`}
            </button>
          ))}
        </div>
      ) : null,
  }),
);

const work = {
  id: "cred-work",
  provider: "github",
  type: "api_key",
  title: "work",
};
const personal = { ...work, id: "cred-personal", title: "personal" };

function providers(saved: (typeof work)[]) {
  return {
    github: {
      provider: "github",
      providerName: "github",
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
    description: "Issues, pull requests, repositories",
    schema: {
      credentials_provider: ["github"],
      credentials_types: ["api_key"],
    },
    selected: undefined,
    hasUnansweredTarget: false,
    select: vi.fn(),
    onConnected: vi.fn(),
    ...overrides,
  };
}

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

function renderRow(current: Row, saved: (typeof work)[]) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>
        <CredentialsProvidersContext.Provider value={providers(saved)}>
          {children}
        </CredentialsProvidersContext.Provider>
      </QueryClientProvider>
    );
  }
  return render(<StatefulRow current={current} />, { wrapper: Wrapper });
}

afterEach(cleanup);

describe("ConnectorRow with several saved accounts for one provider", () => {
  it("offers the accounts instead of picking one or asking for a new sign-in", () => {
    const current = row();
    renderRow(current, [work, personal]);

    expect(current.select).not.toHaveBeenCalled();
    fireEvent.click(screen.getByText("Choose account"));

    expect(screen.getByText("use-work")).toBeDefined();
    expect(screen.getByText("use-personal")).toBeDefined();
  });

  it("selects the account the user chose and keeps it", () => {
    const current = row();
    renderRow(current, [work, personal]);

    fireEvent.click(screen.getByText("Choose account"));
    fireEvent.click(screen.getByText("use-personal"));

    expect(current.select).toHaveBeenLastCalledWith(
      expect.objectContaining({ id: "cred-personal", provider: "github" }),
    );
    expect(current.onConnected).toHaveBeenCalled();
    // Two accounts still match, which used to clear the selection again.
    expect(screen.getByText("Connected")).toBeDefined();
  });

  it("still connects a single saved account on its own", () => {
    const current = row();
    renderRow(current, [work]);

    expect(current.select).toHaveBeenCalledWith(
      expect.objectContaining({ id: "cred-work" }),
    );
    expect(screen.queryByText("Choose account")).toBeNull();
  });
});

describe("ConnectorRow when no saved account has the access a card needs", () => {
  const oauthRow = () =>
    row({
      schema: {
        credentials_provider: ["github"],
        credentials_types: ["oauth2"],
        credentials_scopes: ["repo", "read:org"],
      },
    });
  const oauth = (id: string, title: string) => ({
    id,
    provider: "github",
    type: "oauth2",
    title,
    scopes: ["repo"],
  });

  it("asks which account to update when there are several", () => {
    renderRow(oauthRow(), [
      oauth("cred-a", "work"),
      oauth("cred-b", "personal"),
    ] as never);

    fireEvent.click(screen.getByText("Connect"));

    expect(screen.getByText("purpose-update")).toBeDefined();
    expect(screen.getByText("use-work")).toBeDefined();
    expect(screen.getByText("use-personal")).toBeDefined();
  });

  it("does not ask with a single account, which is upgraded in place already", () => {
    renderRow(oauthRow(), [oauth("cred-a", "work")] as never);

    fireEvent.click(screen.getByText("Connect"));

    expect(screen.queryByText("purpose-update")).toBeNull();
  });

  it("does not ask when the card takes an API key, which no sign-in widens", () => {
    // Several OAuth accounts exist, but this row only accepts an API key:
    // naming one of them changes nothing about the key form that follows.
    renderRow(row(), [
      oauth("cred-a", "work"),
      oauth("cred-b", "personal"),
    ] as never);

    fireEvent.click(screen.getByText("Connect"));

    expect(screen.queryByText("purpose-update")).toBeNull();
    expect(screen.queryByText("use-work")).toBeNull();
  });
});
