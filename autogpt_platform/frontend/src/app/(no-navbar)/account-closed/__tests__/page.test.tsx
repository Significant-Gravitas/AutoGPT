import {
  getGetAccountGetAccountDeletionStatusMockHandler200,
  getPostAccountKeepMyAccountMockHandler200,
} from "@/app/api/__generated__/endpoints/account/account.msw";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { FEEDBACK_FORM_URL } from "@/components/molecules/TallyPoup/helpers";
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

// The real page waits seconds between probes; these tests don't.
vi.mock("../helpers", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../helpers")>()),
  OPEN_PROBE: { intervalMS: 5, streak: 2, maxProbes: 20 },
}));

const RESTORE_PATH = "/api/proxy/api/account/deletion/restore";
const PROBE_PATH = "/api/proxy/api/auth/user/timezone";
const ERASE_AFTER = new Date("2099-11-13T12:00:00Z");
const CLOSED = { detail: "This account is closed." };

let locationReplace: ReturnType<typeof vi.spyOn>;

function status(
  body: Parameters<
    typeof getGetAccountGetAccountDeletionStatusMockHandler200
  >[0],
) {
  server.use(getGetAccountGetAccountDeletionStatusMockHandler200(body));
}

function scheduled(eraseAfter: Date = ERASE_AFTER) {
  status({ status: "scheduled", erase_after: eraseAfter });
}

// The ordinary route the page probes answers closed this many times, as a
// server still caching the closed answer would, then answers normally.
function probeAnswersClosed(times: number) {
  const calls = { count: 0 };
  server.use(
    http.get(PROBE_PATH, () => {
      calls.count += 1;
      return calls.count <= times
        ? HttpResponse.json(CLOSED, { status: 401 })
        : HttpResponse.json({ timezone: "UTC" });
    }),
  );
  return calls;
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
    // Where the page lives: a closed answer to its own probe sends nowhere.
    window.history.replaceState({}, "", "/account-closed");
    locationReplace = vi
      .spyOn(window.location, "replace")
      .mockImplementation(() => undefined);
  });

  afterEach(() => {
    locationReplace.mockRestore();
    mocks.replace.mockReset();
    window.history.replaceState({}, "", "/");
  });

  test("says when the account will be erased and offers to keep it", async () => {
    scheduled();
    renderPage();

    expect(
      await screen.findByRole("heading", { name: "Your account is closed" }),
    ).toBeDefined();
    expect(screen.getByText("Nov 13, 2099")).toBeDefined();
    expect(
      screen.getByText(/API keys, schedules, share links, and Marketplace/),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Keep my account" }),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Sign out" })).toBeDefined();
  });

  test("Keep my account waits until the account answers as open everywhere, then offers to continue", async () => {
    scheduled();
    const probes = probeAnswersClosed(2);
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

    expect(
      await screen.findByRole("heading", {
        name: "Your account is open again",
      }),
    ).toBeDefined();
    expect(restored).toBe(1);
    expect(screen.getByText(/Wait a minute and sign in again/)).toBeDefined();
    expect(
      screen.getByRole("link", { name: "Contact us" }).getAttribute("href"),
    ).toBe(FEEDBACK_FORM_URL);

    const proceed = await screen.findByRole("button", {
      name: "Continue to AutoGPT",
    });
    // Two closed answers, then two open ones in a row.
    expect(probes.count).toBe(4);
    // Never on its own: only when the person continues.
    expect(locationReplace).not.toHaveBeenCalled();

    fireEvent.click(proceed);

    expect(locationReplace).toHaveBeenCalledWith("/");
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

  test("past the erasure date it offers only Sign out", async () => {
    scheduled(new Date("2020-01-01T00:00:00Z"));
    renderPage();

    expect(
      await screen.findByText(/The 14 days to keep your account are over/),
    ).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Keep my account" }),
    ).toBeNull();
    expect(screen.getByRole("button", { name: "Sign out" })).toBeDefined();
    expect(screen.queryByText(/We'll erase your account/)).toBeNull();
  });

  test("Sign out goes through /logout", async () => {
    scheduled();
    renderPage();

    fireEvent.click(await screen.findByRole("button", { name: "Sign out" }));

    expect(mocks.replace).toHaveBeenCalledWith("/logout");
  });

  test("an open account sees the same open state and is not sent anywhere on its own", async () => {
    status({ status: "open", erase_after: null });
    probeAnswersClosed(0);
    renderPage();

    expect(
      await screen.findByRole("heading", {
        name: "Your account is open again",
      }),
    ).toBeDefined();
    expect(screen.getByText(/Wait a minute and sign in again/)).toBeDefined();
    expect(
      await screen.findByRole("button", { name: "Continue to AutoGPT" }),
    ).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Keep my account" }),
    ).toBeNull();
    await waitFor(() => expect(locationReplace).not.toHaveBeenCalled());
  });
});
