import { ImpersonationState } from "@/lib/impersonation";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AdminUsersView } from "../components/AdminUsersView/AdminUsersView";

const mockToast = vi.hoisted(() => vi.fn());
const auth = vi.hoisted(() => ({
  user: { id: "admin-self" } as { id: string } | null,
}));
const reloadMock = vi.hoisted(() => vi.fn());

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: mockToast }),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: auth.user }),
}));

const ALEX = {
  id: "5d0a1c2e-0000-4000-8000-000000000001",
  email: "alex_rivera@example.com",
  name: "Alex Rivera",
  image: null,
  role: "user",
  emailVerified: true,
  createdAt: "2026-03-12T10:00:00.000Z",
  updatedAt: "2026-03-12T10:00:00.000Z",
  banned: false,
  banReason: null,
  banExpires: null,
};

const ORGANISATION = {
  id: "org-1",
  name: "Acme Inc",
  slug: "acme",
  avatar_url: null,
  description: null,
  is_personal: false,
  member_count: 0,
  created_at: "2026-03-12T10:00:00.000Z",
};

beforeEach(() => {
  mockToast.mockReset();
  reloadMock.mockReset();
  auth.user = { id: "admin-self" };
  // happy-dom can't navigate; a started impersonation reloads the page.
  Object.defineProperty(window.location, "reload", {
    configurable: true,
    value: reloadMock,
  });
  mockUserRoutes(ALEX);
});

afterEach(() => ImpersonationState.clear());

describe("/admin/users", () => {
  it("shows a user found by exact email with their plan, organisations and filtered money links", async () => {
    renderPage(`?user=${encodeURIComponent(ALEX.email)}`);

    expect(
      await screen.findByRole("heading", { name: "Alex Rivera" }),
    ).toBeDefined();
    const details = screen.getByRole("region", { name: "Details" });
    expect(await within(details).findByText("Pro")).toBeDefined();
    expect(within(details).getByText(ALEX.id)).toBeDefined();
    expect(await screen.findByText("Acme Inc")).toBeDefined();

    expect(
      screen
        .getByRole("link", { name: /spending history/i })
        .getAttribute("href"),
    ).toBe(`/admin/spending?search=${ALEX.id}`);
    expect(
      screen.getByRole("link", { name: /platform cost/i }).getAttribute("href"),
    ).toBe(`/admin/platform-costs?user_id=${ALEX.id}`);
  });

  it("offers the spending search when nothing matches exactly", async () => {
    renderPage("?user=alex");

    expect(
      await screen.findByRole("heading", { name: /no user matches alex/i }),
    ).toBeDefined();
    expect(
      screen
        .getByRole("link", { name: /search spending history/i })
        .getAttribute("href"),
    ).toBe("/admin/spending?search=alex");
  });

  it("bans only once a reason is given, then shows the user as banned", async () => {
    const bans = mockBan();
    renderPage(`?user=${ALEX.id}`);

    fireEvent.click(await screen.findByRole("button", { name: /ban user/i }));
    const dialog = await screen.findByRole("dialog");
    const confirm = within(dialog).getByRole("button", { name: "Ban user" });
    expect((confirm as HTMLButtonElement).disabled).toBe(true);

    fireEvent.change(within(dialog).getByLabelText(/reason/i), {
      target: { value: "Spam signups" },
    });
    expect((confirm as HTMLButtonElement).disabled).toBe(false);
    fireEvent.click(confirm);

    await waitFor(() =>
      expect(bans).toEqual([{ userId: ALEX.id, banReason: "Spam signups" }]),
    );
    expect(await screen.findByRole("alert")).toBeDefined();
    expect(screen.getByText(/“Spam signups”/)).toBeDefined();
    expect(screen.getByRole("button", { name: "Unban" })).toBeDefined();
  });

  it("treats a temporary ban that has run out as no ban", async () => {
    mockUserRoutes({
      ...ALEX,
      banned: true,
      banReason: "Spam",
      banExpires: "2026-01-01T00:00:00.000Z",
    } as unknown as typeof ALEX);
    renderPage(`?user=${ALEX.id}`);

    expect(
      await screen.findByRole("button", { name: /ban user/i }),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Unban" })).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByText("Active")).toBeDefined();
  });

  it("warns before banning another admin", async () => {
    mockUserRoutes({ ...ALEX, role: "admin" });
    renderPage(`?user=${ALEX.id}`);

    fireEvent.click(await screen.findByRole("button", { name: /ban user/i }));

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText("This account is an admin.")).toBeDefined();
  });

  it("offers no ban or impersonation on the signed-in admin's own account", async () => {
    auth.user = { id: ALEX.id };
    renderPage(`?user=${ALEX.id}`);

    expect(
      await screen.findByRole("heading", { name: "Alex Rivera" }),
    ).toBeDefined();
    expect(screen.getByText("You")).toBeDefined();
    expect(screen.queryByRole("button", { name: /ban user/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /impersonate/i })).toBeNull();
  });

  it("asks before changing the plan, and sends nothing on cancel", async () => {
    const tiers = mockSetTier();
    renderPage(`?user=${ALEX.id}`);

    await chooseOption(/plan/i, /^Max/);
    let dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText("Change plan to Max?")).toBeDefined();
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(tiers).toEqual([]);

    await chooseOption(/plan/i, /^Max/);
    dialog = await screen.findByRole("dialog");
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Change plan" }),
    );
    await waitFor(() =>
      expect(tiers).toEqual([{ user_id: ALEX.id, tier: "MAX" }]),
    );
  });

  it("says when the plan couldn't load rather than showing it blank", async () => {
    server.use(
      http.get("*/api/copilot/admin/rate_limit", () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );
    renderPage(`?user=${ALEX.id}`);

    const details = await screen.findByRole("region", { name: "Details" });
    expect(await within(details).findByText("Couldn't load")).toBeDefined();
  });

  it("shows a billing-set plan again after a plan change is cancelled", async () => {
    server.use(
      http.get("*/api/copilot/admin/rate_limit", () =>
        HttpResponse.json({
          user_id: ALEX.id,
          user_email: ALEX.email,
          daily_cost_limit_microdollars: 5_000_000,
          weekly_cost_limit_microdollars: 25_000_000,
          daily_cost_used_microdollars: 0,
          weekly_cost_used_microdollars: 0,
          tier: "TRIAL",
          tier_multipliers: { MAX: 10 },
        }),
      ),
    );
    renderPage(`?user=${ALEX.id}`);

    await chooseOption(/plan/i, /^Max/);
    const dialog = await screen.findByRole("dialog");
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());

    const select = screen.getByRole("combobox", { name: /plan/i });
    expect(select.textContent).toContain("Trial (set by billing)");
    expect(select.textContent).not.toContain("Max");
  });

  it("asks before impersonating, and stays put when the audit notice fails", async () => {
    const notices = mockImpersonationNotice(502);
    renderPage(`?user=${ALEX.id}`);

    fireEvent.click(await screen.findByRole("button", { name: "Impersonate" }));
    const dialog = await screen.findByRole("dialog");
    expect(notices).toEqual([]);
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Impersonate" }),
    );

    await waitFor(() =>
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Couldn't start impersonation",
          variant: "destructive",
        }),
      ),
    );
    expect(notices).toEqual([{ target_user_id: ALEX.id }]);
    expect(ImpersonationState.get()).toBeNull();
    expect(reloadMock).not.toHaveBeenCalled();
    expect(screen.getByRole("dialog")).toBeDefined();
  });

  it("starts impersonating once the audit notice is delivered", async () => {
    mockImpersonationNotice(200);
    renderPage(`?user=${ALEX.id}`);

    fireEvent.click(await screen.findByRole("button", { name: "Impersonate" }));
    const dialog = await screen.findByRole("dialog");
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Impersonate" }),
    );

    await waitFor(() => expect(reloadMock).toHaveBeenCalledTimes(1));
    expect(ImpersonationState.get()).toBe(ALEX.id);
  });

  it("offers a retry when the organisations fail to load", async () => {
    let calls = 0;
    server.use(
      http.get(`*/api/admin/users/${ALEX.id}/organizations`, () => {
        calls += 1;
        return calls === 1
          ? HttpResponse.json({ detail: "boom" }, { status: 500 })
          : HttpResponse.json([ORGANISATION]);
      }),
    );
    renderPage(`?user=${ALEX.id}`);

    const section = await screen.findByRole("region", {
      name: "Organisations",
    });
    fireEvent.click(
      await within(section).findByRole("button", { name: "Retry" }),
    );

    expect(await within(section).findByText("Acme Inc")).toBeDefined();
  });

  it("offers no ban or impersonation until the signed-in admin is known", async () => {
    auth.user = null;
    renderPage(`?user=${ALEX.id}`);

    expect(
      await screen.findByRole("heading", { name: "Alex Rivera" }),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: /ban user/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /impersonate/i })).toBeNull();
  });
});

function renderPage(search: string) {
  return render(
    <NuqsTestingAdapter hasMemory searchParams={search}>
      <AdminUsersView />
    </NuqsTestingAdapter>,
  );
}

async function chooseOption(combobox: RegExp, option: RegExp) {
  const select = await screen.findByRole("combobox", { name: combobox });
  fireEvent.click(select);
  fireEvent.click(await screen.findByRole("option", { name: option }));
}

function mockUserRoutes(user: typeof ALEX) {
  server.use(
    http.get("*/api/auth/admin/get-user", ({ request }) =>
      new URL(request.url).searchParams.get("id") === user.id
        ? HttpResponse.json(user)
        : HttpResponse.json({ code: "USER_NOT_FOUND" }, { status: 404 }),
    ),
    http.get("*/api/auth/admin/list-users", ({ request }) => {
      const value = new URL(request.url).searchParams.get("filterValue");
      const users = value === user.email ? [user] : [];
      return HttpResponse.json({ users, total: users.length });
    }),
    http.get("*/api/copilot/admin/rate_limit", () =>
      HttpResponse.json({
        user_id: user.id,
        user_email: user.email,
        daily_cost_limit_microdollars: 5_000_000,
        weekly_cost_limit_microdollars: 25_000_000,
        daily_cost_used_microdollars: 1_200_000,
        weekly_cost_used_microdollars: 4_800_000,
        tier: "PRO",
        tier_multipliers: { BASIC: 1, PRO: 3, MAX: 10 },
      }),
    ),
    http.get(`*/api/admin/users/${user.id}/organizations`, () =>
      HttpResponse.json([ORGANISATION]),
    ),
    // The admin's own session, for the Sign-ins card's "that's you" warning.
    http.get("*/api/auth/get-session", () => HttpResponse.json(null)),
  );
}

function mockBan() {
  const bodies: unknown[] = [];
  server.use(
    http.post("*/api/auth/admin/ban-user", async ({ request }) => {
      const body = (await request.json()) as { banReason: string };
      bodies.push(body);
      return HttpResponse.json({
        user: { ...ALEX, banned: true, banReason: body.banReason },
      });
    }),
  );
  return bodies;
}

function mockSetTier() {
  const bodies: unknown[] = [];
  server.use(
    http.post("*/api/copilot/admin/rate_limit/tier", async ({ request }) => {
      bodies.push(await request.json());
      return HttpResponse.json({ user_id: ALEX.id, tier: "MAX" });
    }),
  );
  return bodies;
}

function mockImpersonationNotice(status: 200 | 502) {
  const bodies: unknown[] = [];
  server.use(
    http.post("*/api/admin/impersonation/notify", async ({ request }) => {
      bodies.push(await request.json());
      return status === 200
        ? HttpResponse.json({ alerted: true })
        : HttpResponse.json(
            { detail: "audit channel unavailable" },
            { status },
          );
    }),
  );
  return bodies;
}
