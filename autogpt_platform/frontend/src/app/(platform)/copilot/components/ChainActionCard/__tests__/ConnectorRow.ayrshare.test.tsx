import { getGetV1GetAyrshareSsoUrlMockHandler200 } from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import { server } from "@/mocks/mock-server";
import {
  CredentialsActionsContext,
  CredentialsProvidersContext,
} from "@/providers/agent-credentials/credentials-provider";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState, type ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ConnectorRow } from "../ConnectorRow";
import type { ConnectorRow as Row } from "../helpers";

const SSO_URL = "https://profile.ayrshare.com/social-accounts?jwt=fake";

const managed = {
  id: "cred-managed",
  provider: "ayrshare",
  type: "api_key",
  title: "Ayrshare (managed by AutoGPT)",
  is_managed: true,
};

function providers(saved: (typeof managed)[]) {
  return {
    ayrshare: {
      provider: "ayrshare",
      providerName: "Ayrshare",
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
    provider: "ayrshare",
    displayName: "Ayrshare",
    description: "Post to every social network",
    schema: {
      credentials_provider: ["ayrshare"],
      credentials_types: ["api_key"],
    },
    selected: undefined,
    hasUnansweredTarget: false,
    rejectedCredentialIds: [],
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

function renderRow(current: Row, saved: (typeof managed)[], reload = vi.fn()) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>
        <CredentialsActionsContext.Provider value={{ reload, upsert: vi.fn() }}>
          <CredentialsProvidersContext.Provider value={providers(saved)}>
            {children}
          </CredentialsProvidersContext.Provider>
        </CredentialsActionsContext.Provider>
      </QueryClientProvider>
    );
  }
  return render(<StatefulRow current={current} />, { wrapper: Wrapper });
}

let open: ReturnType<typeof vi.fn>;

beforeEach(() => {
  open = vi.fn(() => ({}) as Window);
  vi.stubGlobal("open", open);
  server.use(
    getGetV1GetAyrshareSsoUrlMockHandler200({
      sso_url: SSO_URL,
      expires_at: new Date("2026-10-14T00:00:00Z"),
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("ConnectorRow for the managed Ayrshare profile", () => {
  it("opens Ayrshare's linking page instead of the API key dialog", async () => {
    const reload = vi.fn();
    renderRow(row(), [], reload);

    fireEvent.click(screen.getByRole("button", { name: "Connect" }));

    await waitFor(() =>
      expect(open).toHaveBeenCalledWith(SSO_URL, "_blank", "popup=true"),
    );
    expect(reload).toHaveBeenCalled();
    expect(screen.queryByText("API Key")).toBeNull();
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("keeps Connect once the managed profile exists, since it says nothing about linked networks", async () => {
    const current = row();
    renderRow(current, [managed]);

    await waitFor(() =>
      expect(current.select).toHaveBeenCalledWith(
        expect.objectContaining({ id: managed.id }),
      ),
    );
    expect(screen.queryByText("Connected")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Connect" }));

    await waitFor(() =>
      expect(open).toHaveBeenCalledWith(SSO_URL, "_blank", "popup=true"),
    );
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
