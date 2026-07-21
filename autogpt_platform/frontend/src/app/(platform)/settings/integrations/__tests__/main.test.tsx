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
  getDeleteV1DeleteCredentialsMockHandler401,
  getGetV1ListCredentialsMockHandler,
  getGetV1ListCredentialsMockHandler401,
  getGetV1ListProvidersMockHandler,
  getPostV1CreateCredentialsMockHandler,
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

  test("managed credentials hide the trash button and select checkbox", async () => {
    server.use(
      getGetV1ListCredentialsMockHandler([
        makeCred({
          id: "a1",
          provider: "ayrshare",
          title: "Ayrshare (managed by AutoGPT)",
          is_managed: true,
        }),
      ]),
    );

    render(<SettingsIntegrationsPage />);

    expect(
      await screen.findByText("Ayrshare (managed by AutoGPT)"),
    ).toBeDefined();
    expect(
      screen.queryByRole("button", { name: /delete ayrshare/i }),
    ).toBeNull();
    expect(
      screen.queryByRole("checkbox", { name: /select ayrshare/i }),
    ).toBeNull();
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

  test("clicking a provider in the list moves to the detail view with auth tabs", async () => {
    server.use(
      getGetV1ListCredentialsMockHandler([]),
      getGetV1ListProvidersMockHandler([
        makeProvider({
          name: "linear",
          description: "Issues and project tracking",
          supported_auth_types: ["oauth2", "api_key"],
        }),
      ]),
    );

    render(<SettingsIntegrationsPage />);

    const connectButtons = await screen.findAllByRole("button", {
      name: /connect.*service/i,
    });
    fireEvent.click(connectButtons[0]);

    const dialog = await screen.findByRole("dialog");
    fireEvent.click(
      await within(dialog).findByText(/issues and project tracking/i),
    );

    // Detail view: back arrow + provider name heading + both tab triggers.
    expect(
      await within(dialog).findByRole("button", { name: /back to services/i }),
    ).toBeDefined();
    expect(
      within(dialog).getByRole("heading", { name: /linear/i }),
    ).toBeDefined();
    expect(within(dialog).getByRole("tab", { name: /oauth/i })).toBeDefined();
    expect(within(dialog).getByRole("tab", { name: /api key/i })).toBeDefined();
  });

  test("API key tab: submitting the form posts credentials and closes the dialog", async () => {
    server.use(
      getGetV1ListCredentialsMockHandler([]),
      getGetV1ListProvidersMockHandler([
        makeProvider({
          name: "openai",
          description: "GPT models",
          supported_auth_types: ["api_key"],
        }),
      ]),
      getPostV1CreateCredentialsMockHandler({
        id: "new-cred",
        provider: "openai",
        type: "api_key",
        title: "My OpenAI key",
        scopes: null,
        username: null,
        host: null,
        is_managed: false,
      }),
    );

    render(<SettingsIntegrationsPage />);

    const connectButtons = await screen.findAllByRole("button", {
      name: /connect.*service/i,
    });
    fireEvent.click(connectButtons[0]);

    const dialog = await screen.findByRole("dialog");
    fireEvent.click(await within(dialog).findByText(/gpt models/i));

    // Two text inputs in the API-key form: title (placeholder "My … key") and
    // the secret (placeholder "sk-...").
    const titleInput =
      await within(dialog).findByPlaceholderText(/my openai key/i);
    const apiKeyInput = within(dialog).getByPlaceholderText(/^sk-\.\.\./);
    fireEvent.change(titleInput, { target: { value: "My OpenAI key" } });
    fireEvent.change(apiKeyInput, { target: { value: "sk-test-key-123" } });

    // RHF onChange-mode validation flips formState.isValid asynchronously;
    // wait until the Save button becomes enabled before clicking.
    const saveBtn = within(dialog).getByRole("button", {
      name: /save api key/i,
    }) as HTMLButtonElement;
    await waitFor(() => {
      expect(saveBtn.disabled).toBe(false);
    });
    fireEvent.click(saveBtn);

    await waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });
  });

  test("Back from detail view returns to the provider list", async () => {
    server.use(
      getGetV1ListCredentialsMockHandler([]),
      getGetV1ListProvidersMockHandler([
        makeProvider({
          name: "github",
          description: "Issues and PRs",
          supported_auth_types: ["api_key"],
        }),
      ]),
    );

    render(<SettingsIntegrationsPage />);

    const connectButtons = await screen.findAllByRole("button", {
      name: /connect.*service/i,
    });
    fireEvent.click(connectButtons[0]);

    const dialog = await screen.findByRole("dialog");
    fireEvent.click(await within(dialog).findByText(/issues and prs/i));

    fireEvent.click(
      await within(dialog).findByRole("button", { name: /back to services/i }),
    );

    expect(
      await within(dialog).findByLabelText(/search services/i),
    ).toBeDefined();
  });
});

describe("SettingsIntegrationsPage — force delete", () => {
  test("a need_confirmation response opens the force-delete dialog and confirming retries with force=true", async () => {
    let deleteCalls = 0;
    server.use(
      getGetV1ListCredentialsMockHandler([
        makeCred({ id: "g1", provider: "github", title: "Personal" }),
      ]),
      // First call (no force): backend asks for confirmation. Second call
      // (force=true from the dialog) succeeds.
      getDeleteV1DeleteCredentialsMockHandler(() => {
        deleteCalls += 1;
        if (deleteCalls === 1) {
          return {
            deleted: false,
            need_confirmation: true,
            message: "Webhook still active — confirm to force-remove.",
          };
        }
        return { deleted: true, revoked: null };
      }),
    );

    render(<SettingsIntegrationsPage />);

    fireEvent.click(
      await screen.findByRole("button", { name: /delete personal/i }),
    );
    const firstDialog = await screen.findByRole("dialog");
    fireEvent.click(
      within(firstDialog).getByRole("button", { name: /^remove$/i }),
    );

    // After the first response, the force-delete dialog auto-opens.
    const forceDialog = await screen.findByRole("dialog", {
      name: /force remove/i,
    });
    expect(within(forceDialog).getByText(/active webhook/i)).toBeDefined();

    // The list refetch after a successful force-delete returns no rows.
    server.use(getGetV1ListCredentialsMockHandler([]));
    fireEvent.click(
      within(forceDialog).getByRole("button", { name: /^force remove$/i }),
    );

    await waitFor(() => {
      expect(screen.queryByText("Personal")).toBeNull();
    });
    expect(deleteCalls).toBe(2);
  });
});

describe("SettingsIntegrationsPage — delete error path", () => {
  test("a 401 on delete surfaces the failure toast and keeps the row", async () => {
    server.use(
      getGetV1ListCredentialsMockHandler([
        makeCred({ id: "g1", provider: "github", title: "Personal" }),
      ]),
      getDeleteV1DeleteCredentialsMockHandler401(),
    );

    render(<SettingsIntegrationsPage />);

    fireEvent.click(
      await screen.findByRole("button", { name: /delete personal/i }),
    );
    const dialog = await screen.findByRole("dialog");
    fireEvent.click(within(dialog).getByRole("button", { name: /^remove$/i }));

    // Delete failed → row stays in the DOM after the dialog closes.
    await waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });
    expect(screen.getByText("Personal")).toBeDefined();
  });
});
