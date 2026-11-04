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
import { OrgAccessView } from "../components/OrgAccessView/OrgAccessView";

const mockToast = vi.hoisted(() => vi.fn());

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: mockToast }),
}));

const EMAIL = "maya@acme.test";

const PLAN = {
  tier: "BUSINESS",
  seats: 10,
  status: "active",
  ends_at: "2027-01-31T00:00:00Z" as string | null,
  note: "Invoice INV-1042" as string | null,
  updated_at: "2026-11-01T10:00:00Z",
  in_effect: true,
};

type Plan = typeof PLAN;

function org(plan: Partial<Plan> | null, balance = 12_500) {
  return {
    org_id: "org-acme",
    name: "Acme",
    slug: "acme",
    owner_user_id: "user-1",
    owner_email: EMAIL,
    plan: plan ? { ...PLAN, ...plan } : null,
    members: 2,
    pending_invitations: 1,
    seats_in_use: 3,
    balance,
  };
}

// Answers the lookup for EMAIL with what orgs() returns now; any other email
// has no account. Returns how many lookups were made.
function mockLookup(orgs: () => ReturnType<typeof org>[]) {
  let lookups = 0;
  server.use(
    http.get("*/api/admin/org-access", () =>
      HttpResponse.json({ grants: [], total: 0, page: 1, page_size: 25 }),
    ),
    http.get("*/api/admin/org-plans/orgs", ({ request }) => {
      lookups += 1;
      return new URL(request.url).searchParams.get("email") === EMAIL
        ? HttpResponse.json(orgs())
        : HttpResponse.json(
            { detail: "No account uses that email" },
            { status: 404 },
          );
    }),
  );
  return () => lookups;
}

function renderPage(email: string | null = EMAIL) {
  const search = email ? `?account=${encodeURIComponent(email)}` : "";
  return render(
    <NuqsTestingAdapter hasMemory searchParams={search}>
      <OrgAccessView />
    </NuqsTestingAdapter>,
  );
}

// The value the card shows for one of its labelled facts.
function fact(card: HTMLElement, label: string) {
  const term = within(card).getByText(label, { selector: "dt *" });
  return term.closest("dt")?.nextElementSibling?.textContent;
}

async function openDialog(button: string, title: string) {
  const card = await screen.findByTestId("team-org");
  fireEvent.click(within(card).getByRole("button", { name: button }));
  return screen.findByRole("dialog", { name: title });
}

async function chooseOption(dialog: HTMLElement, field: RegExp, name: string) {
  fireEvent.click(within(dialog).getByRole("combobox", { name: field }));
  fireEvent.click(await screen.findByRole("option", { name }));
}

beforeEach(() => {
  mockToast.mockReset();
});

describe("Team plans", () => {
  it("says so when the account is in no team organization", async () => {
    mockLookup(() => []);
    renderPage(null);

    fireEvent.change(await screen.findByLabelText("Owner or member email"), {
      target: { value: ` ${EMAIL} ` },
    });
    fireEvent.click(screen.getByRole("button", { name: "Look up" }));

    expect(
      await screen.findByText(`${EMAIL} isn't in any team organization.`),
    ).toBeDefined();
  });

  it("says so when no account signs in with the email", async () => {
    mockLookup(() => []);
    renderPage("nobody@acme.test");

    expect(
      await screen.findByText("No account signs in with nobody@acme.test."),
    ).toBeDefined();
  });

  it("says plainly when a team organization has no plan", async () => {
    mockLookup(() => [org(null)]);
    renderPage();

    const card = await screen.findByTestId("team-org");
    expect(within(card).getByText("Acme")).toBeDefined();
    expect(within(card).getByText("Not in effect")).toBeDefined();
    expect(
      within(card).getByText(/^There's no plan, so members/),
    ).toBeDefined();
    expect(fact(card, "Plan")).toBe("No plan");
    expect(fact(card, "Seats")).toBe(
      "3 in use (2 members, 1 pending invitation)",
    );
    expect(fact(card, "Balance")).toBe("12,500 credits ($125.00)");
  });

  it("shows a plan that is in effect", async () => {
    mockLookup(() => [org(PLAN)]);
    renderPage();

    const card = await screen.findByTestId("team-org");
    expect(within(card).getByText("In effect")).toBeDefined();
    expect(within(card).queryByText(/members see the paywall/)).toBeNull();
    expect(fact(card, "Plan")).toBe("Business");
    expect(fact(card, "Status")).toBe("Active");
    expect(fact(card, "End date")).toBe("Ends 31 Jan 2027");
    expect(fact(card, "Seats")).toBe(
      "3 of 10 in use (2 members, 1 pending invitation)",
    );
    expect(fact(card, "Balance")).toBe("12,500 credits ($125.00)");
    expect(fact(card, "Note")).toBe("Invoice INV-1042");
  });

  it("says why an ended plan is not in effect", async () => {
    mockLookup(() => [
      org({ ends_at: "2026-10-01T00:00:00Z", in_effect: false }),
    ]);
    renderPage();

    const card = await screen.findByTestId("team-org");
    expect(within(card).getByText("Not in effect")).toBeDefined();
    expect(
      within(card).getByText(/^The plan ended on 1 Oct 2026, so members/),
    ).toBeDefined();
    expect(fact(card, "End date")).toBe("Ended 1 Oct 2026");
  });

  it("sets a plan and shows it on the organization", async () => {
    const saved: unknown[] = [];
    const savedPlan = {
      seats: 5,
      ends_at: "2027-03-31T00:00:00Z",
      note: "Invoice INV-2001",
    };
    mockLookup(() => [org(saved.length ? savedPlan : null)]);
    server.use(
      http.put("*/api/admin/org-plans/orgs/:orgId", async ({ request }) => {
        saved.push(await request.json());
        return HttpResponse.json(org(savedPlan));
      }),
    );
    renderPage();

    const dialog = await openDialog("Set plan", "Plan for Acme");
    expect(
      within(dialog).getByRole("button", { name: "Save plan" }),
    ).toHaveProperty("disabled", true);
    await chooseOption(dialog, /tier/i, "Business");
    fireEvent.change(within(dialog).getByLabelText("Seats"), {
      target: { value: "5" },
    });
    fireEvent.change(within(dialog).getByLabelText("End date"), {
      target: { value: "2027-03-31" },
    });
    fireEvent.change(within(dialog).getByLabelText("Note"), {
      target: { value: " Invoice INV-2001 " },
    });
    fireEvent.click(within(dialog).getByRole("button", { name: "Save plan" }));

    await waitFor(() =>
      expect(saved).toEqual([
        {
          tier: "BUSINESS",
          seats: 5,
          ends_at: "2027-03-31T00:00:00.000Z",
          status: "active",
          note: "Invoice INV-2001",
        },
      ]),
    );
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    const card = await screen.findByTestId("team-org");
    expect(await within(card).findByText("In effect")).toBeDefined();
    expect(fact(card, "Seats")).toBe(
      "3 of 5 in use (2 members, 1 pending invitation)",
    );
    expect(fact(card, "End date")).toBe("Ends 31 Mar 2027");
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Saved the plan for Acme",
        variant: "success",
      }),
    );
  });

  it("shows the server's error and keeps the plan form open", async () => {
    mockLookup(() => [org(PLAN)]);
    server.use(
      http.put("*/api/admin/org-plans/orgs/:orgId", () =>
        HttpResponse.json(
          { detail: "Organization #org-acme not found" },
          { status: 404 },
        ),
      ),
    );
    renderPage();

    const dialog = await openDialog("Set plan", "Plan for Acme");
    fireEvent.click(within(dialog).getByRole("button", { name: "Save plan" }));

    expect(
      await within(dialog).findByText("Organization #org-acme not found"),
    ).toBeDefined();
    expect(screen.getByRole("dialog", { name: "Plan for Acme" })).toBeDefined();
    expect(mockToast).not.toHaveBeenCalled();
  });

  it("warns before saving fewer seats than are in use, then saves them", async () => {
    const saved: unknown[] = [];
    mockLookup(() => [org(saved.length ? { seats: 2 } : PLAN)]);
    server.use(
      http.put("*/api/admin/org-plans/orgs/:orgId", async ({ request }) => {
        saved.push(await request.json());
        return HttpResponse.json(org({ seats: 2 }));
      }),
    );
    renderPage();

    const dialog = await openDialog("Set plan", "Plan for Acme");
    expect(within(dialog).queryByText(/already uses 3 seats/)).toBeNull();
    fireEvent.change(within(dialog).getByLabelText("Seats"), {
      target: { value: "2" },
    });

    expect(
      within(dialog).getByText(
        "Acme already uses 3 seats: 2 members and 1 pending invitation. " +
          "Nobody is removed, but with 2 seats, new invitations and " +
          "acceptances are refused until fewer seats are in use.",
      ),
    ).toBeDefined();
    fireEvent.click(within(dialog).getByRole("button", { name: "Save plan" }));

    await waitFor(() =>
      expect(saved).toEqual([expect.objectContaining({ seats: 2 })]),
    );
    const card = await screen.findByTestId("team-org");
    await waitFor(() =>
      expect(fact(card, "Seats")).toBe(
        "3 of 2 in use (2 members, 1 pending invitation)",
      ),
    );
  });

  it("adds credits and shows the new balance", async () => {
    const grants: unknown[] = [];
    mockLookup(() => [org(PLAN, grants.length ? 62_500 : 12_500)]);
    server.use(
      http.post(
        "*/api/admin/org-plans/orgs/:orgId/credits",
        async ({ request }) => {
          grants.push(await request.json());
          return HttpResponse.json({
            org_id: "org-acme",
            reference: "INV-1042",
            granted: true,
            balance: 62_500,
          });
        },
      ),
    );
    renderPage();

    const dialog = await openDialog("Add credits", "Add credits to Acme");
    fireEvent.change(within(dialog).getByLabelText("Credits"), {
      target: { value: "50000" },
    });
    fireEvent.change(
      within(dialog).getByLabelText("Invoice or payment reference"),
      { target: { value: " INV-1042 " } },
    );
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Add credits" }),
    );

    await waitFor(() =>
      expect(grants).toEqual([{ amount: 50_000, reference: "INV-1042" }]),
    );
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    await waitFor(() =>
      expect(fact(screen.getByTestId("team-org"), "Balance")).toBe(
        "62,500 credits ($625.00)",
      ),
    );
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Added 50,000 credits to Acme",
        description: "The balance is now 62,500 credits ($625.00).",
        variant: "success",
      }),
    );
  });

  it("says nothing changed when credits were already added for the reference", async () => {
    let posts = 0;
    const lookups = mockLookup(() => [org(PLAN)]);
    server.use(
      http.post("*/api/admin/org-plans/orgs/:orgId/credits", () => {
        posts += 1;
        return HttpResponse.json({
          org_id: "org-acme",
          reference: "INV-1042",
          granted: false,
          balance: 12_500,
        });
      }),
    );
    renderPage();

    const dialog = await openDialog("Add credits", "Add credits to Acme");
    fireEvent.change(within(dialog).getByLabelText("Credits"), {
      target: { value: "50000" },
    });
    fireEvent.change(
      within(dialog).getByLabelText("Invoice or payment reference"),
      { target: { value: "INV-1042" } },
    );
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Add credits" }),
    );

    expect(
      await within(dialog).findByText(
        "Credits for INV-1042 were already added, so nothing changed. " +
          "The balance is 12,500 credits ($125.00).",
      ),
    ).toBeDefined();
    expect(posts).toBe(1);
    expect(
      screen.getByRole("dialog", { name: "Add credits to Acme" }),
    ).toBeDefined();
    await waitFor(() => expect(lookups()).toBe(2));
    expect(mockToast).not.toHaveBeenCalled();
  });
});
