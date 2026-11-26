import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, test, vi } from "vitest";

import { getGetOauthListMyOauthGrantsMockHandler } from "@/app/api/__generated__/endpoints/oauth/oauth.msw";
import { APIKeyPermission } from "@/app/api/__generated__/models/aPIKeyPermission";
import type { OAuthGrantInfo } from "@/app/api/__generated__/models/oAuthGrantInfo";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";

import SettingsOAuthAppsPage from "../page";

const toastSpy = vi.fn();

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return {
    ...actual,
    toast: (...args: Parameters<typeof actual.toast>) => toastSpy(...args),
  };
});

const GRANT_URL = "/api/proxy/api/oauth/grants/:appId";

function makeGrant(overrides: Partial<OAuthGrantInfo> = {}): OAuthGrantInfo {
  return {
    application_id: "app-notes",
    name: "Notes app",
    logo_url: null,
    scopes: [APIKeyPermission.IDENTITY, APIKeyPermission.READ_GRAPH],
    last_active_at: new Date(),
    ...overrides,
  };
}

async function openRevokeDialog(name: string) {
  fireEvent.click(
    await screen.findByRole("button", { name: `Revoke access for ${name}` }),
  );
  return screen.findByRole("dialog");
}

describe("Settings › OAuth Apps", () => {
  beforeEach(() => {
    toastSpy.mockClear();
  });

  test("lists each app with what it was allowed to do", async () => {
    server.use(
      getGetOauthListMyOauthGrantsMockHandler([
        makeGrant(),
        makeGrant({
          application_id: "app-cal",
          name: "Calendar app",
          scopes: [APIKeyPermission.RUN_AGENT],
        }),
      ]),
    );

    render(<SettingsOAuthAppsPage />);

    const notes = await screen.findByRole("list", {
      name: "What Notes app can do",
    });
    expect(
      within(notes).getByText("View your user ID, e-mail, and timezone"),
    ).toBeDefined();
    expect(
      within(notes).getByText("View your workflows and their configurations"),
    ).toBeDefined();
    const calendar = screen.getByRole("list", {
      name: "What Calendar app can do",
    });
    expect(
      within(calendar).getByText("Run workflows from your library"),
    ).toBeDefined();
    expect(screen.queryByText(/coming soon/i)).toBeNull();
  });

  test("says no app has access when there is none", async () => {
    server.use(getGetOauthListMyOauthGrantsMockHandler([]));

    render(<SettingsOAuthAppsPage />);

    expect(await screen.findByText("No apps have access")).toBeDefined();
  });

  test("revokes the app's access only after the user confirms", async () => {
    let grants = [makeGrant()];
    const revoked: string[] = [];
    server.use(
      getGetOauthListMyOauthGrantsMockHandler(() => grants),
      http.delete(GRANT_URL, ({ params }) => {
        revoked.push(String(params.appId));
        grants = [];
        return new HttpResponse(null, { status: 204 });
      }),
    );

    render(<SettingsOAuthAppsPage />);
    const dialog = await openRevokeDialog("Notes app");
    expect(revoked).toEqual([]);

    fireEvent.click(
      within(dialog).getByRole("button", { name: "Revoke access" }),
    );

    expect(await screen.findByText("No apps have access")).toBeDefined();
    expect(revoked).toEqual(["app-notes"]);
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(toastSpy).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Access revoked" }),
    );
  });

  test("an app that already lost its access closes the dialog and leaves the list", async () => {
    const calendar = makeGrant({
      application_id: "app-cal",
      name: "Calendar app",
    });
    let grants = [makeGrant(), calendar];
    server.use(
      getGetOauthListMyOauthGrantsMockHandler(() => grants),
      http.delete(GRANT_URL, () => {
        grants = [calendar];
        return HttpResponse.json(
          { detail: "This app has no access to your account" },
          { status: 404 },
        );
      }),
    );

    render(<SettingsOAuthAppsPage />);
    const dialog = await openRevokeDialog("Notes app");
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Revoke access" }),
    );

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Revoke access for Notes app" }),
      ).toBeNull(),
    );
    expect(
      screen.getByRole("button", { name: "Revoke access for Calendar app" }),
    ).toBeDefined();
    expect(toastSpy).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Notes app no longer has access" }),
    );
  });

  test("a revoke the server refuses keeps the dialog open and says so", async () => {
    server.use(
      getGetOauthListMyOauthGrantsMockHandler([makeGrant()]),
      http.delete(GRANT_URL, () =>
        HttpResponse.json({ detail: "Database unavailable" }, { status: 500 }),
      ),
    );

    render(<SettingsOAuthAppsPage />);
    const dialog = await openRevokeDialog("Notes app");
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Revoke access" }),
    );

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Couldn't revoke access for Notes app",
          description: "Database unavailable",
        }),
      ),
    );
    expect(screen.getByRole("dialog")).toBeDefined();
    expect(toastSpy).not.toHaveBeenCalledWith(
      expect.objectContaining({ title: "Access revoked" }),
    );
  });
});
