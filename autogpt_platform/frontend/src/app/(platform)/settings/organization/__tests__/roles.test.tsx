import { getPostV2CreateInvitationMockHandler } from "@/app/api/__generated__/endpoints/invitations/invitations.msw";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import OrganizationSettingsPage from "../page";
import {
  ADMIN_MEMBER,
  BILLING_MEMBER,
  PLAIN_MEMBER,
  TEAM_ORG,
  membersViewedAs,
  mockOrg,
  seedActiveOrg,
} from "./fixtures";

const mockSearchParams = vi.hoisted(() => ({
  current: new URLSearchParams(),
}));
vi.mock("next/navigation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("next/navigation")>();
  return {
    ...actual,
    useSearchParams: () => mockSearchParams.current,
    useRouter: () => ({
      push: vi.fn(),
      replace: vi.fn(),
      prefetch: vi.fn(),
      back: vi.fn(),
      forward: vi.fn(),
      refresh: vi.fn(),
    }),
    usePathname: () => "/settings/organization",
    useParams: () => ({}),
  };
});

const toastSpy = vi.hoisted(() => vi.fn());

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-owner" },
    isLoggedIn: true,
    isUserLoading: false,
  }),
}));

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

const NO_FREE_SEATS =
  "All 3 seats on this organization's plan are taken by members and pending invitations. Ask AutoGPT to add seats, or revoke a pending invitation.";

async function tabNames() {
  await screen.findByTestId("org-profile-section");
  return screen.getAllByRole("tab").map((tab) => tab.textContent);
}

afterEach(() => {
  delete process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS;
});

describe("Organization settings by role", () => {
  beforeEach(() => {
    window.localStorage.clear();
    toastSpy.mockClear();
    process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS = "true";
    mockSearchParams.current = new URLSearchParams();
    seedActiveOrg(TEAM_ORG);
  });

  it("gives a plain member only General and Members", async () => {
    mockOrg({ members: membersViewedAs(PLAIN_MEMBER) });
    render(<OrganizationSettingsPage />);

    expect(await tabNames()).toEqual(["General", "Members"]);
  });

  it("hides Billing from an admin, who has no billing rights", async () => {
    mockOrg({ members: membersViewedAs(ADMIN_MEMBER) });
    render(<OrganizationSettingsPage />);

    expect(await tabNames()).toEqual(["General", "Members", "Invitations"]);
  });

  it("gives a billing manager Billing but not Invitations", async () => {
    mockOrg({ members: membersViewedAs(BILLING_MEMBER) });
    render(<OrganizationSettingsPage />);

    expect(await tabNames()).toEqual(["General", "Members", "Billing"]);
    await userEvent.click(screen.getByRole("tab", { name: "Billing" }));
    expect(await screen.findByTestId("org-team-spend-section")).toBeDefined();
  });

  it("falls back to General when a ?tab=billing link reaches a member without billing rights", async () => {
    mockOrg({ members: membersViewedAs(ADMIN_MEMBER) });
    mockSearchParams.current = new URLSearchParams({ tab: "billing" });
    render(<OrganizationSettingsPage />);

    const generalTab = await screen.findByRole("tab", { name: "General" });
    expect(generalTab.getAttribute("data-state")).toBe("active");
    expect(screen.queryByTestId("org-team-spend-section")).toBeNull();
  });

  it("falls back to General when a ?tab=invitations link reaches a plain member", async () => {
    mockOrg({ members: membersViewedAs(PLAIN_MEMBER) });
    mockSearchParams.current = new URLSearchParams({ tab: "invitations" });
    render(<OrganizationSettingsPage />);

    const generalTab = await screen.findByRole("tab", { name: "General" });
    expect(generalTab.getAttribute("data-state")).toBe("active");
    expect(screen.queryByTestId("org-invitations-section")).toBeNull();
  });

  it("keeps role controls for the owner: an admin sees role badges and Remove", async () => {
    mockOrg({
      members: membersViewedAs(ADMIN_MEMBER, [PLAIN_MEMBER, BILLING_MEMBER]),
    });
    render(<OrganizationSettingsPage />);

    await userEvent.click(await screen.findByRole("tab", { name: "Members" }));
    const section = await screen.findByTestId("org-members-section");
    await within(section).findByText("Bob");

    expect(within(section).queryByRole("combobox")).toBeNull();
    const billyRow = within(section)
      .getByText("Billy")
      .closest("li") as HTMLElement;
    expect(within(billyRow).getByText("Billing manager")).toBeDefined();
    // Admins still remove members, as far as the backend allows.
    expect(
      within(section).getByRole("button", { name: "Remove Bob" }),
    ).toBeDefined();
    expect(
      within(section).getByRole("button", { name: "Remove Billy" }),
    ).toBeDefined();
    // Nobody removes the owner or themselves.
    expect(
      within(section).queryByRole("button", { name: "Remove Olivia" }),
    ).toBeNull();
    expect(
      within(section).queryByRole("button", { name: "Remove Adam" }),
    ).toBeNull();
  });

  it("lets an admin invite plain members only, with no role picker", async () => {
    let sentBody: Record<string, unknown> | undefined;
    mockOrg({ members: membersViewedAs(ADMIN_MEMBER) });
    server.use(
      getPostV2CreateInvitationMockHandler(async (info) => {
        sentBody = (await info.request.json()) as Record<string, unknown>;
        return {
          id: "inv-admin",
          email: "new@acme.test",
          is_admin: false,
          is_billing_manager: false,
          token: "tok-admin",
          expires_at: new Date("2026-08-01T00:00:00Z"),
          created_at: new Date("2026-07-01T00:00:00Z"),
          team_ids: [],
        };
      }),
    );
    render(<OrganizationSettingsPage />);

    await userEvent.click(
      await screen.findByRole("tab", { name: "Invitations" }),
    );
    const section = await screen.findByTestId("org-invitations-section");
    expect(
      within(section).queryByRole("combobox", { name: "Role" }),
    ).toBeNull();

    await userEvent.type(
      within(section).getByPlaceholderText("teammate@example.com"),
      "new@acme.test",
    );
    await userEvent.click(
      within(section).getByRole("button", { name: "Invite" }),
    );

    await waitFor(() => {
      expect(sentBody).toMatchObject({
        email: "new@acme.test",
        is_admin: false,
        is_billing_manager: false,
      });
    });
  });

  it("shows the server's message when no seat is free for an invitation", async () => {
    mockOrg();
    server.use(
      http.post("*/api/orgs/:orgId/invitations", () =>
        HttpResponse.json({ detail: NO_FREE_SEATS }, { status: 409 }),
      ),
    );
    render(<OrganizationSettingsPage />);

    await userEvent.click(
      await screen.findByRole("tab", { name: "Invitations" }),
    );
    const section = await screen.findByTestId("org-invitations-section");
    await userEvent.type(
      within(section).getByPlaceholderText("teammate@example.com"),
      "new@acme.test",
    );
    await userEvent.click(
      within(section).getByRole("button", { name: "Invite" }),
    );

    expect(await within(section).findByRole("alert")).toBeDefined();
    expect(within(section).getByText(NO_FREE_SEATS)).toBeDefined();
    // The refusal stays on the form instead of a toast that disappears.
    expect(toastSpy).not.toHaveBeenCalledWith(
      expect.objectContaining({ variant: "destructive" }),
    );
  });
});
