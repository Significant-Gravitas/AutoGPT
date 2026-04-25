import { describe, expect, test, vi } from "vitest";

import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getDeleteV1DeleteCredentialsMockHandler,
  getGetV1ListCredentialsMockHandler,
  getGetV1ListCredentialsMockHandler401,
  getGetV1ListProvidersMockHandler,
} from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import type { ProviderMetadata } from "@/app/api/__generated__/models/providerMetadata";

import SettingsIntegrationsPage from "../page";

function makeCred(
  overrides: Partial<CredentialsMetaResponse> = {},
): CredentialsMetaResponse {
  return {
    id: "cred-1",
    provider: "github",
    type: "api_key",
    title: "Personal Token",
    scopes: null,
    username: null,
    host: null,
    is_managed: false,
    ...overrides,
  };
}

function makeProvider(
  overrides: Partial<ProviderMetadata> = {},
): ProviderMetadata {
  return {
    name: "github",
    description: "Issues and PRs",
    supported_auth_types: ["oauth2", "api_key"],
    ...overrides,
  };
}

describe("SettingsIntegrationsPage — list", () => {
  test("renders the header and Connect Service control", async () => {
    server.use(getGetV1ListCredentialsMockHandler([]));

    render(<SettingsIntegrationsPage />);

    expect(
      await screen.findByRole("heading", { name: /integrations/i }),
    ).toBeDefined();
    const connectButtons = screen.getAllByRole("button", {
      name: /connect.*service/i,
    });
    expect(connectButtons.length).toBeGreaterThan(0);
  });

  test("renders the empty state when the user has no credentials", async () => {
    server.use(getGetV1ListCredentialsMockHandler([]));

    render(<SettingsIntegrationsPage />);

    const buttons = await screen.findAllByRole("button", {
      name: /connect.*service/i,
    });
    expect(buttons.length).toBeGreaterThan(0);
    expect(screen.queryByRole("button", { name: /select.*token/i })).toBeNull();
  });

  test("groups credentials by provider and renders one row per credential", async () => {
    server.use(
      getGetV1ListCredentialsMockHandler([
        makeCred({
          id: "g1",
          provider: "github",
          title: "Personal",
        }),
        makeCred({
          id: "g2",
          provider: "github",
          title: "Work",
        }),
        makeCred({
          id: "o1",
          provider: "openai",
          title: "OpenAI key",
        }),
      ]),
    );

    render(<SettingsIntegrationsPage />);

    expect(await screen.findByText("Personal")).toBeDefined();
    expect(screen.getByText("Work")).toBeDefined();
    expect(screen.getByText("OpenAI key")).toBeDefined();

    // Provider headers (formatted via formatProviderName).
    expect(screen.getByText("GitHub")).toBeDefined();
    expect(screen.getByText("OpenAI")).toBeDefined();
  });

  test("renders an error card on 401 instead of the empty state", async () => {
    server.use(getGetV1ListCredentialsMockHandler401());

    render(<SettingsIntegrationsPage />);

    expect(await screen.findByText(/something went wrong/i)).toBeDefined();
  });
});

describe("SettingsIntegrationsPage — delete", () => {
  test("clicking the trash icon opens a confirm dialog naming the credential", async () => {
    server.use(
      getGetV1ListCredentialsMockHandler([
        makeCred({ id: "g1", provider: "github", title: "Personal" }),
      ]),
    );

    render(<SettingsIntegrationsPage />);

    fireEvent.click(
      await screen.findByRole("button", { name: /delete personal/i }),
    );

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText(/remove personal/i)).toBeDefined();
    expect(
      within(dialog).getByRole("button", { name: /^remove$/i }),
    ).toBeDefined();
  });

  test("cancel in confirm dialog leaves the credential row in place", async () => {
    server.use(
      getGetV1ListCredentialsMockHandler([
        makeCred({ id: "g1", provider: "github", title: "Personal" }),
      ]),
    );

    render(<SettingsIntegrationsPage />);

    fireEvent.click(
      await screen.findByRole("button", { name: /delete personal/i }),
    );
    const dialog = await screen.findByRole("dialog");
    fireEvent.click(within(dialog).getByRole("button", { name: /cancel/i }));

    await waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });
    expect(screen.getByText("Personal")).toBeDefined();
  });

  test("confirming a delete fires the API and the row is gone after refetch", async () => {
    server.use(
      getGetV1ListCredentialsMockHandler([
        makeCred({ id: "g1", provider: "github", title: "Personal" }),
      ]),
      getDeleteV1DeleteCredentialsMockHandler({
        deleted: true,
        revoked: null,
      }),
    );

    render(<SettingsIntegrationsPage />);

    fireEvent.click(
      await screen.findByRole("button", { name: /delete personal/i }),
    );
    const dialog = await screen.findByRole("dialog");

    // After the delete response, the list refetch returns no rows.
    server.use(getGetV1ListCredentialsMockHandler([]));
    fireEvent.click(within(dialog).getByRole("button", { name: /^remove$/i }));

    await waitFor(() => {
      expect(screen.queryByText("Personal")).toBeNull();
    });
  });
});

describe("SettingsIntegrationsPage — selection bar", () => {
  test("selecting a credential exposes a bulk delete bar", async () => {
    server.use(
      getGetV1ListCredentialsMockHandler([
        makeCred({ id: "g1", provider: "github", title: "Personal" }),
        makeCred({ id: "g2", provider: "github", title: "Work" }),
      ]),
    );

    render(<SettingsIntegrationsPage />);

    fireEvent.click(
      await screen.findByRole("checkbox", { name: /select personal/i }),
    );

    expect(await screen.findByText(/1 selected/i)).toBeDefined();
    expect(
      screen.getByRole("button", { name: /delete selected/i }),
    ).toBeDefined();
  });

  test("Deselect clears the selection bar", async () => {
    server.use(
      getGetV1ListCredentialsMockHandler([
        makeCred({ id: "g1", provider: "github", title: "Personal" }),
      ]),
    );

    render(<SettingsIntegrationsPage />);

    fireEvent.click(
      await screen.findByRole("checkbox", { name: /select personal/i }),
    );
    await screen.findByText(/1 selected/i);
    fireEvent.click(screen.getByRole("button", { name: /deselect/i }));

    await waitFor(() => {
      expect(screen.queryByText(/1 selected/i)).toBeNull();
    });
  });
});

describe("SettingsIntegrationsPage — search", () => {
  test("debounced search filters the list to matches", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      server.use(
        getGetV1ListCredentialsMockHandler([
          makeCred({ id: "g1", provider: "github", title: "Personal" }),
          makeCred({ id: "o1", provider: "openai", title: "OpenAI key" }),
        ]),
      );

      render(<SettingsIntegrationsPage />);

      await screen.findByText("Personal");

      const input = screen.getByLabelText(/search integrations/i);
      fireEvent.change(input, { target: { value: "openai" } });

      // Push past the 250ms debounce.
      vi.advanceTimersByTime(300);

      await waitFor(() => {
        expect(screen.queryByText("Personal")).toBeNull();
      });
      expect(screen.getByText("OpenAI key")).toBeDefined();
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("SettingsIntegrationsPage — connect dialog", () => {
  test("Connect Service opens a dialog with the provider list", async () => {
    server.use(
      getGetV1ListCredentialsMockHandler([]),
      getGetV1ListProvidersMockHandler([
        makeProvider({ name: "github", description: "Issues and PRs" }),
      ]),
    );

    render(<SettingsIntegrationsPage />);

    const connectButtons = await screen.findAllByRole("button", {
      name: /connect.*service/i,
    });
    fireEvent.click(connectButtons[0]);

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText(/connect a service/i)).toBeDefined();
    expect(await within(dialog).findByText(/issues and prs/i)).toBeDefined();
  });
});
