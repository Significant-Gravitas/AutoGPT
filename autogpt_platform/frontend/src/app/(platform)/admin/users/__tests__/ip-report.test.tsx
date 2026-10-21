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
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AdminUsersView } from "../components/AdminUsersView/AdminUsersView";

const mockToast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: mockToast }),
}));
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: { id: "admin-self" } }),
}));

const ALEX = {
  id: "5d0a1c2e-0000-4000-8000-000000000001",
  email: "alex@example.com",
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

function related(
  id: string,
  email: string,
  fields: Partial<{
    role: string | null;
    banned: boolean;
    via_ips: string[];
    via_domain: boolean;
    plan: string | null;
    payment_status: string;
  }> = {},
) {
  return {
    id,
    email,
    name: "",
    role: "user",
    created_at: "2026-10-01T10:00:00.000Z",
    banned: false,
    ban_expires: null,
    has_profile: true,
    via_ips: [],
    via_domain: false,
    plan: null,
    payment_status: "no_subscription",
    ...fields,
  };
}

const SAM = related("sam", "sam@example.com", {
  via_ips: ["203.0.113.7"],
  via_domain: true,
});
const JO = related("jo", "jo@other.test", {
  via_ips: ["203.0.113.7"],
  banned: true,
});
const RILEY = related("riley", "riley@example.com", {
  via_domain: true,
  role: "admin",
});
const ME = related("admin-self", "me@agpt.test", { via_ips: ["203.0.113.7"] });

const BLOCK = {
  id: "block-1",
  kind: "IP",
  value: "198.51.100.22",
  reason: "Signup farm",
  created_by_id: "admin-2",
  created_by_email: "nick@agpt.test",
  created_at: "2026-10-03T10:00:00.000Z",
  expires_at: null,
};

function ip(address: string, fields: Record<string, unknown> = {}) {
  return {
    ip_address: address,
    display_address: address,
    is_ipv6_prefix: false,
    has_sign_up: false,
    sign_in_count: 1,
    first_seen: "2026-10-18T09:00:00.000Z",
    last_seen: "2026-10-20T22:14:00.000Z",
    last_user_agent:
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    other_account_count: 0,
    block: null,
    events: [],
    event_count: 1,
    ...fields,
  };
}

function report(overrides: Record<string, unknown> = {}) {
  return {
    retention_days: 90,
    ips: [
      ip("203.0.113.7", { has_sign_up: true, other_account_count: 3 }),
      ip("198.51.100.22", { block: BLOCK }),
    ],
    domain: {
      value: "example.com",
      is_public_provider: false,
      other_account_count: 2,
      block: null,
    },
    related: [SAM, JO, RILEY, ME],
    related_total: 4,
    ...overrides,
  };
}

beforeEach(() => {
  mockToast.mockReset();
  server.use(
    http.get("*/api/auth/admin/get-user", () => HttpResponse.json(ALEX)),
    http.get("*/api/auth/get-session", () =>
      HttpResponse.json({
        session: { ipAddress: "192.0.2.99" },
        user: { id: "admin-self" },
      }),
    ),
    http.get(`*/api/admin/users/${ALEX.id}/organizations`, () =>
      HttpResponse.json([]),
    ),
  );
  mockReport(report());
});

describe("Ban all", () => {
  it("ticks the accounts on this account's addresses but not those on its domain alone, and bans exactly the ticked ones", async () => {
    const bans = mockBanAccounts();
    renderPage(`?user=${ALEX.id}`);

    const dialog = await openBanAll();
    expect(within(dialog).getByText("Ban 2 accounts?")).toBeDefined();
    expect(
      within(dialog).getByRole("checkbox", {
        name: "Ban riley@example.com",
        checked: false,
      }),
    ).toBeDefined();
    expect(within(dialog).queryByText(/is an admin/)).toBeNull();
    expect(
      within(dialog).getByText("Your own account isn't included."),
    ).toBeDefined();

    fireEvent.click(
      within(dialog).getByRole("checkbox", { name: "Ban riley@example.com" }),
    );
    expect(within(dialog).getByText("Ban 3 accounts?")).toBeDefined();
    expect(
      within(dialog).getByText("1 of these accounts is an admin."),
    ).toBeDefined();
    fireEvent.click(
      within(dialog).getByRole("checkbox", { name: "Ban sam@example.com" }),
    );
    fireEvent.change(within(dialog).getByLabelText(/reason/i), {
      target: { value: "Signup farm" },
    });
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Ban 2 accounts" }),
    );

    await waitFor(() =>
      expect(bans).toEqual([
        { userIds: [ALEX.id, RILEY.id], reason: "Signup farm" },
      ]),
    );
    await waitFor(() =>
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Banned 2 accounts" }),
      ),
    );
  });

  it("names what it blocks, and bans nobody, once every account is unticked", async () => {
    const bans = mockBanAccounts();
    const blocks = mockCreateBlock();
    renderPage(`?user=${ALEX.id}`);

    const dialog = await openBanAll();
    for (const email of [ALEX.email, "sam@example.com"]) {
      fireEvent.click(
        within(dialog).getByRole("checkbox", { name: `Ban ${email}` }),
      );
    }
    fireEvent.click(
      within(dialog).getByRole("checkbox", { name: /block 203\.0\.113\.7/i }),
    );
    expect(within(dialog).getByText("Block 1 address?")).toBeDefined();
    expect(within(dialog).queryByText(/Ban 0/)).toBeNull();
    fireEvent.change(within(dialog).getByLabelText(/reason/i), {
      target: { value: "Signup farm" },
    });
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Block 1 address" }),
    );

    await waitFor(() =>
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Blocked 1 address" }),
      ),
    );
    expect(blocks).toHaveLength(1);
    expect(bans).toEqual([]);
  });

  it("after a partial failure, says which block failed too and retries only the failed accounts still ticked", async () => {
    const bans = mockBanAccounts((userIds) =>
      userIds.map((userId) =>
        bans.length === 1
          ? { userId, ok: false, error: "Ban refused" }
          : { userId, ok: true },
      ),
    );
    mockCreateBlock(409);
    renderPage(`?user=${ALEX.id}`);

    const dialog = await openBanAll();
    fireEvent.click(
      within(dialog).getByRole("checkbox", { name: /block 203\.0\.113\.7/i }),
    );
    fireEvent.change(within(dialog).getByLabelText(/reason/i), {
      target: { value: "Signup farm" },
    });
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Ban 2 accounts, block 1" }),
    );

    expect((await within(dialog).findAllByText("Ban refused")).length).toBe(2);
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Banned 0 of 2 accounts",
        description: expect.stringContaining("is already blocked"),
        variant: "destructive",
      }),
    );
    fireEvent.click(
      within(dialog).getByRole("checkbox", { name: "Ban sam@example.com" }),
    );
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Retry 1 failed" }),
    );
    await waitFor(() => expect(bans.at(-1)?.userIds).toEqual([ALEX.id]));
  });
});

describe("Block address", () => {
  it("blocks the address with the reason and offers the Blocks tab", async () => {
    const blocks = mockCreateBlock();
    renderPage(`?user=${ALEX.id}`);

    const dialog = await openDialog(/^block…$/i);
    expect(within(dialog).getByText("Block 203.0.113.7?")).toBeDefined();
    fireEvent.change(within(dialog).getByLabelText(/reason/i), {
      target: { value: "Signup farm" },
    });
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Block address" }),
    );

    await waitFor(() =>
      expect(blocks).toEqual([
        {
          kind: "IP",
          value: "203.0.113.7",
          reason: "Signup farm",
          expires_at: null,
        },
      ]),
    );
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Blocked 203.0.113.7" }),
    );
  });

  it("shows a block someone else made first in the dialog, and stops", async () => {
    mockCreateBlock(409);
    renderPage(`?user=${ALEX.id}`);

    const dialog = await openDialog(/^block…$/i);
    fireEvent.change(within(dialog).getByLabelText(/reason/i), {
      target: { value: "Signup farm" },
    });
    const confirm = within(dialog).getByRole("button", {
      name: "Block address",
    });
    fireEvent.click(confirm);

    expect(
      await within(dialog).findByText(/203\.0\.113\.7 is already blocked/),
    ).toBeDefined();
    expect((confirm as HTMLButtonElement).disabled).toBe(true);
    expect(mockToast).not.toHaveBeenCalled();
  });
});

describe("Block domain", () => {
  it("turns into a ban once its accounts are included, and bans exactly them", async () => {
    const blocks = mockCreateBlock();
    const bans = mockBanAccounts();
    renderPage(`?user=${ALEX.id}`);

    const dialog = await openDialog(/block domain…/i);
    expect(
      within(dialog).getByRole("button", { name: "Block domain" }),
    ).toBeDefined();
    fireEvent.click(
      within(dialog).getByRole("checkbox", { name: /also ban/i }),
    );
    fireEvent.change(within(dialog).getByLabelText(/reason/i), {
      target: { value: "Throwaway domain" },
    });
    fireEvent.click(
      within(dialog).getByRole("button", {
        name: "Block domain and ban 2 accounts",
      }),
    );

    await waitFor(() =>
      expect(bans).toEqual([
        { userIds: [SAM.id, RILEY.id], reason: "Throwaway domain" },
      ]),
    );
    expect(blocks).toEqual([
      {
        kind: "EMAIL_DOMAIN",
        value: "example.com",
        reason: "Throwaway domain",
        expires_at: null,
      },
    ]);
  });

  it("can't block a public mail provider", async () => {
    mockReport(
      report({
        domain: {
          value: "gmail.com",
          is_public_provider: true,
          other_account_count: 0,
          block: null,
        },
        related: [JO],
        related_total: 1,
      }),
    );
    renderPage(`?user=${ALEX.id}`);

    expect(await screen.findByText(/is a public mail provider/)).toBeDefined();
    expect(screen.queryByRole("button", { name: /block domain…/i })).toBeNull();
  });
});

describe("Block domain, while the page refreshes", () => {
  it("stays open with its failed bans when the refreshed report shows the domain blocked", async () => {
    mockCreateBlock();
    mockBanAccounts((userIds) =>
      userIds.map((userId) =>
        userId === RILEY.id
          ? { userId, ok: false, error: "Ban refused" }
          : { userId, ok: true },
      ),
    );
    let blocked = false;
    server.use(
      http.post("*/api/admin/access-blocks", () => {
        blocked = true;
        return HttpResponse.json({ ...BLOCK, id: "new" }, { status: 201 });
      }),
      http.get(`*/api/admin/users/${ALEX.id}/ip-report`, () =>
        HttpResponse.json(
          blocked
            ? report({
                domain: {
                  value: "example.com",
                  is_public_provider: false,
                  other_account_count: 2,
                  block: {
                    ...BLOCK,
                    kind: "EMAIL_DOMAIN",
                    value: "example.com",
                  },
                },
              })
            : report(),
        ),
      ),
    );
    renderPage(`?user=${ALEX.id}`);

    const dialog = await openDialog(/block domain…/i);
    fireEvent.click(
      within(dialog).getByRole("checkbox", { name: /also ban/i }),
    );
    fireEvent.change(within(dialog).getByLabelText(/reason/i), {
      target: { value: "Throwaway domain" },
    });
    fireEvent.click(
      within(dialog).getByRole("button", {
        name: "Block domain and ban 2 accounts",
      }),
    );

    // The address row's Unblock was there already; the domain's joins it.
    await waitFor(() =>
      expect(
        screen.getAllByRole("button", { name: "Unblock", hidden: true }),
      ).toHaveLength(2),
    );
    const open = screen.getByRole("dialog");
    expect(within(open).getByText("Ban refused")).toBeDefined();
    expect(
      within(open).getByRole("button", { name: "Retry 1 failed" }),
    ).toBeDefined();
  });

  it("offers to block a private domain before any related account exists", async () => {
    mockReport(
      report({
        domain: {
          value: "example.com",
          is_public_provider: false,
          other_account_count: 0,
          block: null,
        },
        related: [],
        related_total: 0,
      }),
    );
    renderPage(`?user=${ALEX.id}`);

    expect(
      await screen.findByRole("button", { name: /block domain…/i }),
    ).toBeDefined();
  });
});

describe("Related accounts", () => {
  it("shows each account's plan and payment status", async () => {
    const statuses = [
      ["paying@example.com", "PRO", "active", "Pro · Active"],
      ["owing@example.com", "MAX", "past_due", "Max · Past due"],
      ["granted@example.com", "PRO", "granted", "Pro · Granted, not billed"],
      ["trial@example.com", "TRIAL", "trialing", "Trial"],
      ["lapsed@example.com", "NO_TIER", "canceled", "No plan · Canceled"],
      ["bare@example.com", null, "no_subscription", "No plan"],
    ] as const;
    mockReport(
      report({
        related: statuses.map(([email, plan, payment_status]) =>
          related(email, email, { via_domain: true, plan, payment_status }),
        ),
        related_total: statuses.length,
      }),
    );
    renderPage(`?user=${ALEX.id}`);

    for (const [email, , , label] of statuses) {
      const row = (await screen.findByText(email)).closest("li");
      expect(within(row!).getByText(label)).toBeDefined();
    }
  });
});

describe("Sign-ins", () => {
  it("shows a lost admin role instead of offering a retry", async () => {
    server.use(
      http.get(`*/api/admin/users/${ALEX.id}/ip-report`, () =>
        HttpResponse.json({ detail: "Admin access required" }, { status: 403 }),
      ),
    );
    renderPage(`?user=${ALEX.id}`);

    expect(
      (
        await screen.findAllByText(
          "You no longer have admin access. Sign in again.",
        )
      ).length,
    ).toBe(2);
  });
});

describe("Blocks tab", () => {
  it("lists the blocks in force and lifts one", async () => {
    const deleted: string[] = [];
    server.use(
      http.get("*/api/admin/access-blocks", () =>
        HttpResponse.json({ blocks: [BLOCK], total: 1 }),
      ),
      http.delete("*/api/admin/access-blocks/:blockId", ({ params }) => {
        deleted.push(String(params.blockId));
        return new HttpResponse(null, { status: 204 });
      }),
    );
    renderPage("?tab=blocks");

    expect(await screen.findByText("Signup farm")).toBeDefined();
    expect(screen.getByText("nick@agpt.test")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Unblock" }));
    const dialog = await screen.findByRole("dialog");
    fireEvent.click(within(dialog).getByRole("button", { name: "Unblock" }));

    await waitFor(() => expect(deleted).toEqual([BLOCK.id]));
  });
});

describe("Blocks tab, paging", () => {
  it("loads further blocks by offset until it has them all", async () => {
    const offsets: number[] = [];
    const TOTAL = 120;
    server.use(
      http.get("*/api/admin/access-blocks", ({ request }) => {
        const params = new URL(request.url).searchParams;
        const limit = Number(params.get("limit"));
        const offset = Number(params.get("offset") ?? 0);
        if (limit !== 1) offsets.push(offset);
        const count = Math.max(0, Math.min(limit, TOTAL - offset));
        const blocks = Array.from({ length: count }, (_, i) => ({
          ...BLOCK,
          id: `block-${offset + i}`,
          value: `198.51.100.${offset + i}`,
          reason: `Reason ${offset + i}`,
        }));
        return HttpResponse.json({ blocks, total: TOTAL });
      }),
    );
    renderPage("?tab=blocks");

    expect(await screen.findByText("Reason 0")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Show 50 more" }));
    expect(await screen.findByText("Reason 50")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Show 50 more" }));
    expect(await screen.findByText("Reason 119")).toBeDefined();

    expect(screen.queryByRole("button", { name: "Show 50 more" })).toBeNull();
    expect(offsets).toEqual([0, 50, 100]);
  });
});

function renderPage(search: string) {
  return render(
    <NuqsTestingAdapter hasMemory searchParams={search}>
      <AdminUsersView />
    </NuqsTestingAdapter>,
  );
}

async function openDialog(trigger: RegExp) {
  fireEvent.click(await screen.findByRole("button", { name: trigger }));
  return screen.findByRole("dialog");
}

function openBanAll() {
  return openDialog(/ban all…/i);
}

function mockReport(body: ReturnType<typeof report>) {
  server.use(
    http.get(`*/api/admin/users/${ALEX.id}/ip-report`, () =>
      HttpResponse.json(body),
    ),
    http.get("*/api/admin/access-blocks", () =>
      HttpResponse.json({ blocks: [], total: 0 }),
    ),
  );
}

type BanResult = { userId: string; ok: boolean; error?: string };

function mockBanAccounts(
  answer: (userIds: string[]) => BanResult[] = (userIds) =>
    userIds.map((userId) => ({ userId, ok: true })),
) {
  const bodies: Array<{ userIds: string[]; reason: string }> = [];
  server.use(
    http.post("*/api/admin/ban-accounts", async ({ request }) => {
      const body = (await request.json()) as {
        userIds: string[];
        reason: string;
      };
      bodies.push({ userIds: body.userIds, reason: body.reason });
      return HttpResponse.json({ results: answer(body.userIds) });
    }),
  );
  return bodies;
}

function mockCreateBlock(status: 201 | 409 = 201) {
  const bodies: unknown[] = [];
  server.use(
    http.post("*/api/admin/access-blocks", async ({ request }) => {
      const body = await request.json();
      bodies.push(body);
      return status === 201
        ? HttpResponse.json({ ...BLOCK, id: "new" }, { status: 201 })
        : HttpResponse.json(
            {
              detail:
                "203.0.113.7 is already blocked, by nick@agpt.test on 3 Oct 2026.",
            },
            { status: 409 },
          );
    }),
  );
  return bodies;
}
