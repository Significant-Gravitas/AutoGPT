import { describe, expect, test } from "vitest";

import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getDeleteV1RevokeApiKeyMockHandler200,
  getDeleteV1RevokeApiKeyMockHandler422,
  getGetV1ListUserApiKeysMockHandler,
} from "@/app/api/__generated__/endpoints/api-keys/api-keys.msw";
import type { APIKeyInfo } from "@/app/api/__generated__/models/aPIKeyInfo";
import { APIKeyPermission } from "@/app/api/__generated__/models/aPIKeyPermission";
import { APIKeyStatus } from "@/app/api/__generated__/models/aPIKeyStatus";

import SettingsApiKeysPage from "../page";

function makeKey(overrides: Partial<APIKeyInfo> = {}): APIKeyInfo {
  return {
    id: "key-base",
    user_id: "user-1",
    name: "Base Key",
    head: "sk-abcd1234",
    tail: "wxyz5678",
    status: APIKeyStatus.ACTIVE,
    scopes: [APIKeyPermission.EXECUTE_GRAPH],
    created_at: new Date("2025-01-01T00:00:00Z"),
    last_used_at: null,
    revoked_at: null,
    expires_at: null,
    description: null,
    ...overrides,
  };
}

describe("SettingsApiKeysPage - revoke flow", () => {
  test("clicking the row trash opens a single-key confirm dialog", async () => {
    server.use(
      getGetV1ListUserApiKeysMockHandler([
        makeKey({ id: "k1", name: "Alpha" }),
        makeKey({ id: "k2", name: "Beta" }),
      ]),
      getDeleteV1RevokeApiKeyMockHandler200(),
    );

    render(<SettingsApiKeysPage />);
    await screen.findByText("Alpha");

    fireEvent.click(screen.getByRole("button", { name: /delete alpha/i }));

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText(/revoke api key/i)).toBeDefined();
    expect(
      within(dialog).getByRole("button", { name: /^revoke key$/i }),
    ).toBeDefined();
    expect(
      within(dialog).getByRole("button", { name: /cancel/i }),
    ).toBeDefined();
  });

  test("cancel closes the confirm dialog without deleting", async () => {
    server.use(
      getGetV1ListUserApiKeysMockHandler([
        makeKey({ id: "k1", name: "Alpha" }),
      ]),
    );

    render(<SettingsApiKeysPage />);
    await screen.findByText("Alpha");

    fireEvent.click(screen.getByRole("button", { name: /delete alpha/i }));
    const dialog = await screen.findByRole("dialog");

    fireEvent.click(within(dialog).getByRole("button", { name: /cancel/i }));

    await waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });

    // Row is still rendered; no crash, no redirect.
    expect(screen.getByText("Alpha")).toBeDefined();
  });

  test("confirming revoke closes the dialog", async () => {
    server.use(
      getGetV1ListUserApiKeysMockHandler([
        makeKey({ id: "k1", name: "Alpha" }),
      ]),
      getDeleteV1RevokeApiKeyMockHandler200(),
    );

    render(<SettingsApiKeysPage />);
    await screen.findByText("Alpha");

    fireEvent.click(screen.getByRole("button", { name: /delete alpha/i }));
    const dialog = await screen.findByRole("dialog");

    fireEvent.click(
      within(dialog).getByRole("button", { name: /^revoke key$/i }),
    );

    await waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });
  });

  test("selecting multiple rows surfaces the selection bar with count", async () => {
    server.use(
      getGetV1ListUserApiKeysMockHandler([
        makeKey({ id: "k1", name: "Alpha" }),
        makeKey({ id: "k2", name: "Beta" }),
        makeKey({ id: "k3", name: "Gamma" }),
      ]),
    );

    render(<SettingsApiKeysPage />);
    await screen.findByText("Alpha");

    fireEvent.click(screen.getByRole("checkbox", { name: /select alpha/i }));
    fireEvent.click(screen.getByRole("checkbox", { name: /select beta/i }));

    expect(await screen.findByText(/2 selected/i)).toBeDefined();
    expect(
      screen.getByRole("button", { name: /delete selected/i }),
    ).toBeDefined();
  });

  test("keeps the dialog open and does not clear selection when revoke fails", async () => {
    server.use(
      getGetV1ListUserApiKeysMockHandler([
        makeKey({ id: "k1", name: "Alpha" }),
        makeKey({ id: "k2", name: "Beta" }),
      ]),
      getDeleteV1RevokeApiKeyMockHandler422(),
    );

    render(<SettingsApiKeysPage />);
    await screen.findByText("Alpha");

    fireEvent.click(screen.getByRole("checkbox", { name: /select alpha/i }));
    fireEvent.click(screen.getByRole("checkbox", { name: /select beta/i }));
    await screen.findByText(/2 selected/i);

    fireEvent.click(screen.getByRole("button", { name: /delete selected/i }));
    const dialog = await screen.findByRole("dialog");

    fireEvent.click(
      within(dialog).getByRole("button", { name: /^revoke keys$/i }),
    );

    // Dialog stays open, selection bar still present, nothing cleared.
    await waitFor(() => {
      expect(
        within(dialog).getByRole("button", { name: /^revoke keys$/i }),
      ).toBeDefined();
    });
    expect(screen.getByText(/2 selected/i)).toBeDefined();
  });

  test("batch delete opens a multi-key confirm dialog and closes on confirm", async () => {
    server.use(
      getGetV1ListUserApiKeysMockHandler([
        makeKey({ id: "k1", name: "Alpha" }),
        makeKey({ id: "k2", name: "Beta" }),
      ]),
      getDeleteV1RevokeApiKeyMockHandler200(),
    );

    render(<SettingsApiKeysPage />);
    await screen.findByText("Alpha");

    fireEvent.click(screen.getByRole("checkbox", { name: /select alpha/i }));
    fireEvent.click(screen.getByRole("checkbox", { name: /select beta/i }));

    await screen.findByText(/2 selected/i);

    fireEvent.click(screen.getByRole("button", { name: /delete selected/i }));

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText(/revoke 2 api keys/i)).toBeDefined();

    fireEvent.click(
      within(dialog).getByRole("button", { name: /^revoke keys$/i }),
    );

    await waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });
  });
});
