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
  OPEN_PROBE: { intervalMS: 5, streak: 2, minWaitMS: 0, maxProbes: 20 },
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

// The ordinary route the page probes refuses this many times (as a server
// still caching the closed answer would, by default), then answers normally.
function probeRefuses(times: number, status = 401, body: object = CLOSED) {
  const calls = { count: 0 };
  server.use(
    http.get(PROBE_PATH, () => {
      calls.count += 1;
      return calls.count <= times
        ? HttpResponse.json(body, { status })
        : HttpResponse.json({ timezone: "UTC" });
    }),
  );
  return calls;
}

function keepSucceeds() {
  server.use(
    getPostAccountKeepMyAccountMockHandler200({
      status: "open",
      erase_after: null,
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
      screen.getByText(/API keys, schedules and share links stopped/),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Keep my account" }),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Sign out" })).toBeDefined();
  });

  test("Keep my account waits until the account answers as open everywhere, then offers to continue", async () => {
    scheduled();
    const probes = probeRefuses(2);
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

    const heading = await screen.findByRole("heading", {
      name: "Your account is open again",
    });
    expect(restored).toBe(1);
    // Focus follows the swapped card instead of falling back to the page.
    await waitFor(() => expect(document.activeElement).toBe(heading));
    const progress = screen.getByText(/Opening your account everywhere/);
    expect(progress.getAttribute("aria-live")).toBe("polite");
    expect(screen.getByText(/Wait a minute and sign in again/)).toBeDefined();
    expect(
      screen.getByRole("link", { name: "Contact us" }).getAttribute("href"),
    ).toBe(FEEDBACK_FORM_URL);

    const proceed = await screen.findByRole("button", {
      name: "Continue to AutoGPT",
    });
    // Two closed answers, then two open ones in a row.
    expect(probes.count).toBe(4);
    // The same live region, its text changed.
    expect(progress.textContent).toMatch(/open everywhere/);
    await waitFor(() => expect(document.activeElement).toBe(proceed));
    // Never on its own: only when the person continues.
    expect(locationReplace).not.toHaveBeenCalled();

    fireEvent.click(proceed);

    expect(locationReplace).toHaveBeenCalledWith("/");
  });

  test("a server error or a lost session is not counted as open", async () => {
    scheduled();
    keepSucceeds();
    const probes = probeRefuses(2, 503, { detail: "Service unavailable" });
    renderPage();

    fireEvent.click(
      await screen.findByRole("button", { name: "Keep my account" }),
    );

    await screen.findByRole("button", { name: "Continue to AutoGPT" });
    expect(probes.count).toBe(4);
  });

  test("too late to keep it: the page switches to the too-late state", async () => {
    scheduled();
    server.use(
      http.post(RESTORE_PATH, () =>
        HttpResponse.json(
          { detail: "It's too late to keep this account." },
          { status: 409 },
        ),
      ),
    );
    renderPage();

    fireEvent.click(
      await screen.findByRole("button", { name: "Keep my account" }),
    );

    expect(
      await screen.findByText(/The 14 days to keep your account are over/),
    ).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Keep my account" }),
    ).toBeNull();
    expect(screen.getByRole("button", { name: "Sign out" })).toBeDefined();
    expect(locationReplace).not.toHaveBeenCalled();
  });

  test("once the erasure has started it offers only Sign out", async () => {
    status({ status: "too_late", erase_after: ERASE_AFTER });
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

  test("an account that wasn't kept on this page goes back to the app once it answers as open", async () => {
    // Never closed (say /login?next=/account-closed), or kept earlier: a
    // reload or a second tab while other servers still cache "closed".
    status({ status: "open", erase_after: null });
    const probes = probeRefuses(2);
    renderPage();

    await waitFor(() => expect(locationReplace).toHaveBeenCalledWith("/"));
    // Two closed answers, then two open ones in a row, before leaving.
    expect(probes.count).toBe(4);
    expect(locationReplace).toHaveBeenCalledTimes(1);
    expect(
      screen.queryByRole("heading", { name: "Your account is open again" }),
    ).toBeNull();
    expect(
      screen.queryByRole("button", { name: "Keep my account" }),
    ).toBeNull();
  });

  test("it still goes back to the app once the probes give up", async () => {
    status({ status: "open", erase_after: null });
    const probes = probeRefuses(1000);
    renderPage();

    await waitFor(() => expect(locationReplace).toHaveBeenCalledWith("/"));
    expect(probes.count).toBe(20);
  });
});
