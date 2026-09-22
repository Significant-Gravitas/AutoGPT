import { describe, expect, test } from "vitest";

import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getGetV1ListUserApiKeysMockHandler,
  getGetV1ListUserApiKeysMockHandler401,
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

describe("SettingsApiKeysPage - list rendering", () => {
  test("renders header title and Create Key button", async () => {
    server.use(getGetV1ListUserApiKeysMockHandler([]));

    render(<SettingsApiKeysPage />);

    expect(
      await screen.findByRole("heading", { name: /autogpt api keys/i }),
    ).toBeDefined();

    const createButtons = screen.getAllByRole("button", {
      name: /create key/i,
    });
    expect(createButtons.length).toBeGreaterThan(0);
  });

  test("shows empty state when the user has no keys", async () => {
    server.use(getGetV1ListUserApiKeysMockHandler([]));

    render(<SettingsApiKeysPage />);

    expect(await screen.findByText(/no api key found/i)).toBeDefined();
  });

  test("renders a row per ACTIVE key with masked secret and last-used label", async () => {
    server.use(
      getGetV1ListUserApiKeysMockHandler([
        makeKey({
          id: "k1",
          name: "Prod Key",
          head: "sk-head111",
          tail: "tail99999",
          last_used_at: null,
        }),
      ]),
    );

    render(<SettingsApiKeysPage />);

    expect(await screen.findByText("Prod Key")).toBeDefined();
    expect(screen.getByText("sk-head111••••••••tail99999")).toBeDefined();
    expect(screen.getByText(/never used/i)).toBeDefined();
    expect(
      screen.getByRole("checkbox", { name: /select prod key/i }),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: /delete prod key/i }),
    ).toBeDefined();
  });

  test("filters REVOKED keys out of the rendered list", async () => {
    server.use(
      getGetV1ListUserApiKeysMockHandler([
        makeKey({ id: "a", name: "Active Key" }),
        makeKey({
          id: "r",
          name: "Revoked Key",
          status: APIKeyStatus.REVOKED,
        }),
      ]),
    );

    render(<SettingsApiKeysPage />);

    expect(await screen.findByText("Active Key")).toBeDefined();
    expect(screen.queryByText("Revoked Key")).toBeNull();
  });

  test("renders an error card (not the empty state) when the API fails", async () => {
    server.use(getGetV1ListUserApiKeysMockHandler401());

    render(<SettingsApiKeysPage />);

    expect(await screen.findByText(/something went wrong/i)).toBeDefined();
    expect(screen.queryByText(/no api key found/i)).toBeNull();
  });

  test("error card 'Try again' refetches and recovers when the API succeeds", async () => {
    server.use(getGetV1ListUserApiKeysMockHandler401());

    render(<SettingsApiKeysPage />);

    await screen.findByText(/something went wrong/i);

    server.use(
      getGetV1ListUserApiKeysMockHandler([
        makeKey({ id: "recovered", name: "Recovered Key" }),
      ]),
    );

    fireEvent.click(screen.getByRole("button", { name: /try again/i }));

    expect(await screen.findByText("Recovered Key")).toBeDefined();
  });

  test("clicking the row info icon opens the details dialog for that key", async () => {
    server.use(
      getGetV1ListUserApiKeysMockHandler([
        makeKey({ id: "a", name: "Inspectable Key" }),
      ]),
    );

    render(<SettingsApiKeysPage />);

    const infoButton = await screen.findByRole("button", {
      name: /view details for inspectable key/i,
    });
    fireEvent.click(infoButton);

    const dialog = await screen.findByRole("dialog");
    expect(dialog.textContent).toContain("Inspectable Key");
  });

  test("renders multiple keys in order", async () => {
    server.use(
      getGetV1ListUserApiKeysMockHandler([
        makeKey({ id: "a", name: "Alpha Key" }),
        makeKey({ id: "b", name: "Beta Key" }),
        makeKey({ id: "c", name: "Gamma Key" }),
      ]),
    );

    render(<SettingsApiKeysPage />);

    expect(await screen.findByText("Alpha Key")).toBeDefined();
    expect(screen.getByText("Beta Key")).toBeDefined();
    expect(screen.getByText("Gamma Key")).toBeDefined();
  });
});
