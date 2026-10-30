import {
  getGetAccountGetAccountDeletionStatusMockHandler200,
  getPostAccountKeepMyAccountMockHandler200,
} from "@/app/api/__generated__/endpoints/account/account.msw";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import AccountClosedPage from "../page";

const mocks = vi.hoisted(() => ({ replace: vi.fn() }));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: mocks.replace, push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/account-closed",
}));

const RESTORE_PATH = "/api/proxy/api/account/deletion/restore";
const ERASE_AFTER = new Date("2026-11-13T12:00:00Z");

let locationReplace: ReturnType<typeof vi.spyOn>;

function scheduled() {
  server.use(
    getGetAccountGetAccountDeletionStatusMockHandler200({
      status: "scheduled",
      erase_after: ERASE_AFTER,
    }),
  );
}

function renderPage() {
  render(
    <>
      <AccountClosedPage />
      <Toaster />
    </>,
  );
}

describe("AccountClosedPage", () => {
  beforeEach(() => {
    locationReplace = vi
      .spyOn(window.location, "replace")
      .mockImplementation(() => undefined);
  });

  afterEach(() => {
    locationReplace.mockRestore();
    mocks.replace.mockReset();
  });

  test("says when the account will be erased and offers to keep it", async () => {
    scheduled();
    renderPage();

    expect(
      await screen.findByRole("heading", { name: "Your account is closed" }),
    ).toBeDefined();
    expect(screen.getByText("Nov 13, 2026")).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Keep my account" }),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Sign out" })).toBeDefined();
  });

  test("Keep my account reopens it and loads the app again", async () => {
    scheduled();
    let restored = 0;
    server.use(
      getPostAccountKeepMyAccountMockHandler200(() => {
        restored += 1;
        return { status: "open", erase_after: null };
      }),
    );
    renderPage();

    fireEvent.click(
      await screen.findByRole("button", { name: "Keep my account" }),
    );

    await waitFor(() => expect(locationReplace).toHaveBeenCalledWith("/"));
    expect(restored).toBe(1);
    expect(
      await screen.findByText("Your account is open again."),
    ).toBeDefined();
  });

  test("too late to keep it: says so and stays", async () => {
    scheduled();
    const detail =
      "It's too late to keep this account: its data is being erased. Contact contact@agpt.co if you need help.";
    server.use(
      http.post(RESTORE_PATH, () =>
        HttpResponse.json({ detail }, { status: 409 }),
      ),
    );
    renderPage();

    fireEvent.click(
      await screen.findByRole("button", { name: "Keep my account" }),
    );

    expect(await screen.findByText(detail)).toBeDefined();
    expect(locationReplace).not.toHaveBeenCalled();
    expect(
      screen.getByRole("button", { name: "Keep my account" }),
    ).toBeDefined();
  });

  test("Sign out goes through /logout", async () => {
    scheduled();
    renderPage();

    fireEvent.click(await screen.findByRole("button", { name: "Sign out" }));

    expect(mocks.replace).toHaveBeenCalledWith("/logout");
  });

  test("an open account has nothing to keep and goes back to the app", async () => {
    server.use(
      getGetAccountGetAccountDeletionStatusMockHandler200({
        status: "open",
        erase_after: null,
      }),
    );
    renderPage();

    await waitFor(() => expect(locationReplace).toHaveBeenCalledWith("/"));
    expect(
      screen.queryByRole("button", { name: "Keep my account" }),
    ).toBeNull();
  });
});
