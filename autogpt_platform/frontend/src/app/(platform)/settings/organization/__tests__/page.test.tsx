import { useOrgTeamStore } from "@/services/org-team/store";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  getDeleteV2DeleteOrganizationMockHandler,
  getDeleteV2DeleteOrganizationMockHandler422,
  getDeleteV2RemoveMemberFromOrganizationMockHandler,
  getGetV2GetOrganizationDetailsMockHandler,
  getGetV2ListOrganizationMembersMockHandler,
  getGetV2ListOrganizationMembersMockHandler401,
  getGetV2ListUserOrganizationsMockHandler,
  getGetV2ListWorkspacesMockHandler,
  getPatchV2UpdateMemberRoleMockHandler,
  getPatchV2UpdateOrganizationMockHandler,
  getPostV2TransferOrganizationOwnershipMockHandler,
  getPostV2TransferOrganizationOwnershipMockHandler422,
} from "@/app/api/__generated__/endpoints/orgs/orgs.msw";
import {
  getDeleteV2RevokeInvitationMockHandler,
  getGetV2ListPendingInvitationsMockHandler,
  getPostV2AcceptInvitationMockHandler,
  getPostV2CreateInvitationMockHandler,
} from "@/app/api/__generated__/endpoints/invitations/invitations.msw";

import OrganizationSettingsPage from "../page";
import {
  ADMIN_BILLING_MEMBER,
  ADMIN_MEMBER,
  BILLING_MEMBER,
  DEFAULT_TEAM,
  ENGINEERING_TEAM,
  MARKETING_TEAM,
  OTHER_ORG,
  OWNER_MEMBER,
  PENDING_INVITATION_FROM_OTHER_ORG,
  PERSONAL_ORG,
  PLAIN_MEMBER,
  TEAM_ORG,
  membersViewedAs,
  mockOrg,
  seedActiveOrg,
  storeOrg,
} from "./fixtures";

// Allow per-test override of the search params Next.js exposes to the page so
// the ?tab= deep-link can be exercised.
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

async function goToTab(
  name: "General" | "Members" | "Invitations" | "Billing",
) {
  await userEvent.click(await screen.findByRole("tab", { name }));
}

afterEach(() => {
  delete process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS;
});

describe("OrganizationSettingsPage", () => {
  beforeEach(() => {
    window.localStorage.clear();
    toastSpy.mockClear();
    process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS = "true";
    mockSearchParams.current = new URLSearchParams();
    seedActiveOrg(TEAM_ORG);
  });

  it("returns not found for a direct link when org settings are disabled", () => {
    let requests = 0;
    process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS = "false";
    server.use(
      getGetV2GetOrganizationDetailsMockHandler(() => {
        requests += 1;
        return TEAM_ORG;
      }),
      getGetV2ListOrganizationMembersMockHandler(() => {
        requests += 1;
        return [OWNER_MEMBER];
      }),
    );

    expect(() => render(<OrganizationSettingsPage />)).toThrow();
    expect(requests).toBe(0);
  });

  it("gives a team org's owner every tab, with profile and danger zone on General", async () => {
    mockOrg();
    render(<OrganizationSettingsPage />);

    expect(await screen.findByTestId("org-profile-section")).toBeDefined();
    expect(screen.getByTestId("org-danger-zone")).toBeDefined();

    const tabs = screen.getAllByRole("tab").map((tab) => tab.textContent);
    expect(tabs).toEqual(["General", "Members", "Invitations", "Billing"]);
    // Team switching is off, so the Teams tab stays hidden.
    expect(screen.queryByRole("tab", { name: "Teams" })).toBeNull();
  });

  it("keeps the members roster on the Members tab, separate from invitations", async () => {
    mockOrg();
    render(<OrganizationSettingsPage />);

    await goToTab("Members");
    expect(await screen.findByText("Members (2)")).toBeDefined();
    expect(screen.getAllByTestId("org-member-row")).toHaveLength(2);
    expect(screen.getByText("Bob")).toBeDefined();
    expect(screen.queryByTestId("org-invitations-section")).toBeNull();

    await goToTab("Invitations");
    expect(await screen.findByTestId("org-invitations-section")).toBeDefined();
    expect(screen.getByPlaceholderText("teammate@example.com")).toBeDefined();
  });

  it("hides admin-only sections from a plain member", async () => {
    mockOrg({ members: membersViewedAs(PLAIN_MEMBER) });
    render(<OrganizationSettingsPage />);

    await screen.findByTestId("org-profile-section");
    expect(screen.queryByTestId("org-danger-zone")).toBeNull();
    expect(screen.queryByRole("combobox", { name: "New owner" })).toBeNull();
    expect(screen.queryByText("Save changes")).toBeNull();

    expect(screen.queryByRole("tab", { name: "Invitations" })).toBeNull();
    expect(screen.queryByRole("tab", { name: "Billing" })).toBeNull();

    await goToTab("Members");
    await screen.findByTestId("org-members-section");
    expect(screen.queryByTestId("org-invitations-section")).toBeNull();
    expect(screen.queryByRole("button", { name: /^Remove/ })).toBeNull();
  });

  it("sends an invitation from the invite form", async () => {
    const createSpy = vi.fn();
    mockOrg();
    server.use(
      getPostV2CreateInvitationMockHandler(() => {
        createSpy();
        return {
          id: "inv-1",
          email: "new@acme.test",
          is_admin: false,
          is_billing_manager: false,
          token: "tok-1",
          expires_at: new Date("2026-08-01T00:00:00Z"),
          created_at: new Date("2026-07-01T00:00:00Z"),
          team_ids: [],
        };
      }),
    );
    render(<OrganizationSettingsPage />);

    await goToTab("Invitations");
    const emailInput = await screen.findByPlaceholderText(
      "teammate@example.com",
    );
    await userEvent.type(emailInput, "new@acme.test");
    await userEvent.click(screen.getByRole("button", { name: "Invite" }));

    await waitFor(() => {
      expect(createSpy).toHaveBeenCalledTimes(1);
    });
  });

  it("sends the chosen role's flags when the owner invites", async () => {
    let sentBody: Record<string, unknown> | undefined;
    mockOrg();
    server.use(
      getPostV2CreateInvitationMockHandler(async (info) => {
        sentBody = (await info.request.json()) as Record<string, unknown>;
        return {
          id: "inv-role",
          email: "new@acme.test",
          is_admin: true,
          is_billing_manager: false,
          token: "tok-role",
          expires_at: new Date("2026-08-01T00:00:00Z"),
          created_at: new Date("2026-07-01T00:00:00Z"),
          team_ids: [],
        };
      }),
    );
    render(<OrganizationSettingsPage />);

    await goToTab("Invitations");
    await userEvent.type(
      await screen.findByPlaceholderText("teammate@example.com"),
      "new@acme.test",
    );
    fireEvent.click(screen.getByRole("combobox", { name: "Role" }));
    fireEvent.click(await screen.findByRole("option", { name: "Admin" }));
    await userEvent.click(screen.getByRole("button", { name: "Invite" }));

    await waitFor(() => {
      expect(sentBody).toBeDefined();
    });
    expect(sentBody).toMatchObject({
      email: "new@acme.test",
      is_admin: true,
      is_billing_manager: false,
    });
  });

  it("pre-assigns the selected teams when sending an invitation", async () => {
    let sentTeamIds: string[] | undefined;
    mockOrg();
    server.use(
      getGetV2ListWorkspacesMockHandler([
        DEFAULT_TEAM,
        MARKETING_TEAM,
        ENGINEERING_TEAM,
      ]),
      getPostV2CreateInvitationMockHandler(async (info) => {
        const body = (await info.request.json()) as { team_ids?: string[] };
        sentTeamIds = body.team_ids;
        return {
          id: "inv-2",
          email: "new@acme.test",
          is_admin: false,
          is_billing_manager: false,
          token: "tok-2",
          expires_at: new Date("2026-08-01T00:00:00Z"),
          created_at: new Date("2026-07-01T00:00:00Z"),
          team_ids: [MARKETING_TEAM.id, ENGINEERING_TEAM.id],
        };
      }),
    );
    render(<OrganizationSettingsPage />);

    await goToTab("Invitations");
    await userEvent.click(
      await screen.findByRole("checkbox", { name: "Marketing" }),
    );
    await userEvent.click(
      screen.getByRole("checkbox", { name: "Engineering" }),
    );
    await userEvent.type(
      screen.getByPlaceholderText("teammate@example.com"),
      "new@acme.test",
    );
    await userEvent.click(screen.getByRole("button", { name: "Invite" }));

    await waitFor(() => {
      expect(sentTeamIds).toBeDefined();
    });
    expect(sentTeamIds).toEqual(
      expect.arrayContaining([MARKETING_TEAM.id, ENGINEERING_TEAM.id]),
    );
    expect(sentTeamIds).toHaveLength(2);
    expect(sentTeamIds).not.toContain(DEFAULT_TEAM.id);
  });

  it("hides the team selector when the org has only the default team", async () => {
    let teamsServed = () => {};
    const teamsAnswered = new Promise<void>((resolve) => {
      teamsServed = resolve;
    });
    mockOrg();
    server.use(
      getGetV2ListWorkspacesMockHandler(() => {
        teamsServed();
        return [DEFAULT_TEAM];
      }),
      // Answer the invitation list only after the team list, so the empty
      // list showing means the team list has already arrived.
      getGetV2ListPendingInvitationsMockHandler(async () => {
        await teamsAnswered;
        return [];
      }),
    );
    render(<OrganizationSettingsPage />);

    await goToTab("Invitations");
    await screen.findByText("No pending invitations.");
    // The default team is auto-joined, so with no other teams there is nothing
    // to pre-assign — the selector must not render.
    expect(screen.queryByText("Pre-assign to teams")).toBeNull();
    expect(
      screen.queryByRole("group", { name: "Pre-assign to teams" }),
    ).toBeNull();
  });

  it("spells out assigned team names on a pending invitation", async () => {
    mockOrg({
      orgInvitations: [
        {
          id: "inv-teams",
          email: "new@acme.test",
          is_admin: false,
          is_billing_manager: false,
          expires_at: new Date("2026-08-01T00:00:00Z"),
          created_at: new Date("2026-07-01T00:00:00Z"),
          team_ids: [MARKETING_TEAM.id, ENGINEERING_TEAM.id],
        },
      ],
    });
    server.use(
      getGetV2ListWorkspacesMockHandler([
        DEFAULT_TEAM,
        MARKETING_TEAM,
        ENGINEERING_TEAM,
      ]),
    );
    render(<OrganizationSettingsPage />);

    await goToTab("Invitations");
    const row = await screen.findByTestId("org-invitation-row");
    expect(await within(row).findByText("Marketing")).toBeDefined();
    expect(within(row).getByText("Engineering")).toBeDefined();
    expect(within(row).queryByText(/\+\d+ teams?/)).toBeNull();
    expect(within(row).getByText(/Expires/).textContent).not.toContain(
      "Marketing",
    );
  });

  it("accepts a pending invitation and switches to the inviting org", async () => {
    const acceptSpy = vi.fn();
    mockOrg({ myInvitations: [PENDING_INVITATION_FROM_OTHER_ORG] });
    server.use(
      getPostV2AcceptInvitationMockHandler(() => {
        acceptSpy();
        return { orgId: OTHER_ORG.id, message: "Invitation accepted" };
      }),
      getGetV2ListUserOrganizationsMockHandler([TEAM_ORG, OTHER_ORG]),
    );
    render(<OrganizationSettingsPage />);

    expect(await screen.findByText("Other Corp")).toBeDefined();
    await userEvent.click(screen.getByRole("button", { name: "Accept" }));

    await waitFor(() => {
      expect(acceptSpy).toHaveBeenCalledTimes(1);
      expect(useOrgTeamStore.getState().activeOrgID).toBe(OTHER_ORG.id);
    });
    // The joined org must land in the store too — an activeOrgID that isn't in
    // `orgs` leaves the switcher with a null active org until a page reload.
    await waitFor(() => {
      expect(useOrgTeamStore.getState().orgs.map((org) => org.id)).toContain(
        OTHER_ORG.id,
      );
    });
  });

  it("keeps the joined org in the store when the org list refresh fails", async () => {
    mockOrg({ myInvitations: [PENDING_INVITATION_FROM_OTHER_ORG] });
    server.use(
      getPostV2AcceptInvitationMockHandler(() => ({
        orgId: OTHER_ORG.id,
        message: "Invitation accepted",
      })),
      getGetV2ListUserOrganizationsMockHandler(() => {
        throw new Error("network down");
      }),
    );
    render(<OrganizationSettingsPage />);

    expect(await screen.findByText("Other Corp")).toBeDefined();
    await userEvent.click(screen.getByRole("button", { name: "Accept" }));

    await waitFor(() => {
      const state = useOrgTeamStore.getState();
      expect(state.activeOrgID).toBe(OTHER_ORG.id);
      expect(state.orgs.map((org) => org.id)).toContain(OTHER_ORG.id);
    });
  });

  it("keeps the joined org in the store when the refreshed list omits it", async () => {
    mockOrg({ myInvitations: [PENDING_INVITATION_FROM_OTHER_ORG] });
    server.use(
      getPostV2AcceptInvitationMockHandler(() => ({
        orgId: OTHER_ORG.id,
        message: "Invitation accepted",
      })),
      // The membership hasn't propagated to the list endpoint yet.
      getGetV2ListUserOrganizationsMockHandler([TEAM_ORG]),
    );
    render(<OrganizationSettingsPage />);

    expect(await screen.findByText("Other Corp")).toBeDefined();
    await userEvent.click(screen.getByRole("button", { name: "Accept" }));

    await waitFor(() => {
      const state = useOrgTeamStore.getState();
      expect(state.activeOrgID).toBe(OTHER_ORG.id);
      expect(state.orgs.map((org) => org.id)).toContain(OTHER_ORG.id);
    });
  });

  it("shows an error when the members query fails", async () => {
    mockOrg();
    server.use(getGetV2ListOrganizationMembersMockHandler401());
    render(<OrganizationSettingsPage />);

    expect(
      await screen.findByText(/Failed to load organization/i),
    ).toBeDefined();
    expect(screen.queryByTestId("org-members-section")).toBeNull();
  });

  it("only shows a loading state on the invitation being revoked", async () => {
    let releaseRevoke = () => {};
    const revokeInFlight = new Promise<void>((resolve) => {
      releaseRevoke = resolve;
    });
    mockOrg({
      orgInvitations: [
        {
          id: "inv-1",
          email: "first@acme.test",
          is_admin: false,
          is_billing_manager: false,
          expires_at: new Date("2026-08-01T00:00:00Z"),
          created_at: new Date("2026-07-01T00:00:00Z"),
          team_ids: [],
        },
        {
          id: "inv-2",
          email: "second@acme.test",
          is_admin: false,
          is_billing_manager: false,
          expires_at: new Date("2026-08-01T00:00:00Z"),
          created_at: new Date("2026-07-01T00:00:00Z"),
          team_ids: [],
        },
      ],
    });
    server.use(
      getDeleteV2RevokeInvitationMockHandler(async () => {
        await revokeInFlight;
        return undefined;
      }),
    );
    render(<OrganizationSettingsPage />);

    await goToTab("Invitations");
    const rows = await screen.findAllByTestId("org-invitation-row");
    expect(rows).toHaveLength(2);

    await userEvent.click(
      within(rows[0]).getByRole("button", { name: "Revoke" }),
    );

    await waitFor(() => {
      const revoking = within(rows[0]).getByRole("button", {
        name: "Revoke",
      }) as HTMLButtonElement;
      expect(revoking.disabled).toBe(true);
    });
    const untouched = within(rows[1]).getByRole("button", {
      name: "Revoke",
    }) as HTMLButtonElement;
    expect(untouched.disabled).toBe(false);

    releaseRevoke();
    await waitFor(() => {
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({ variant: "success" }),
      );
    });
  });

  it("removes a member after confirmation", async () => {
    const removeSpy = vi.fn();
    mockOrg();
    server.use(
      getDeleteV2RemoveMemberFromOrganizationMockHandler(() => {
        removeSpy();
        return undefined;
      }),
    );
    render(<OrganizationSettingsPage />);

    await goToTab("Members");
    await screen.findByText("Bob");
    await userEvent.click(screen.getByRole("button", { name: "Remove Bob" }));
    await userEvent.click(
      await screen.findByRole("button", { name: "Remove member" }),
    );

    await waitFor(() => {
      expect(removeSpy).toHaveBeenCalledTimes(1);
    });
  });

  it("transfers ownership to a selected member and drops the ex-owner's controls", async () => {
    let sentNewOwnerId: string | undefined;
    let ownershipTransferred = false;
    mockOrg();
    server.use(
      getGetV2ListOrganizationMembersMockHandler(() =>
        ownershipTransferred
          ? [
              { ...OWNER_MEMBER, is_owner: false },
              { ...PLAIN_MEMBER, is_owner: true, is_admin: true },
            ]
          : [OWNER_MEMBER, PLAIN_MEMBER],
      ),
      getPostV2TransferOrganizationOwnershipMockHandler(
        async (info: { request: Request }) => {
          const body = (await info.request.json()) as { new_owner_id?: string };
          sentNewOwnerId = body.new_owner_id;
          ownershipTransferred = true;
        },
      ),
    );
    render(<OrganizationSettingsPage />);

    const dangerZone = await screen.findByTestId("org-danger-zone");
    fireEvent.click(
      within(dangerZone).getByRole("combobox", { name: "New owner" }),
    );
    fireEvent.click(await screen.findByRole("option", { name: "Bob" }));
    await userEvent.click(
      within(dangerZone).getByRole("button", { name: "Transfer" }),
    );
    await userEvent.click(
      await screen.findByRole("button", { name: "Transfer ownership" }),
    );

    await waitFor(() => {
      expect(sentNewOwnerId).toBe(PLAIN_MEMBER.user_id);
    });
    // The refetch re-runs role gating: the ex-owner loses the danger zone.
    await waitFor(() => {
      expect(screen.queryByTestId("org-danger-zone")).toBeNull();
    });
  });

  it("deletes the organization and falls back to the personal org", async () => {
    let deleteRequested = false;
    useOrgTeamStore.setState({
      orgs: [storeOrg(TEAM_ORG), storeOrg(PERSONAL_ORG)],
    });
    mockOrg();
    server.use(
      getDeleteV2DeleteOrganizationMockHandler(() => {
        deleteRequested = true;
      }),
    );
    render(<OrganizationSettingsPage />);

    const dangerZone = await screen.findByTestId("org-danger-zone");
    await userEvent.click(
      within(dangerZone).getByRole("button", { name: "Delete organization" }),
    );
    const dialog = await screen.findByRole("dialog");
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Delete organization" }),
    );

    await waitFor(() => {
      expect(deleteRequested).toBe(true);
    });
    await waitFor(() => {
      expect(useOrgTeamStore.getState().activeOrgID).toBe(PERSONAL_ORG.id);
    });
    expect(useOrgTeamStore.getState().orgs.map((org) => org.id)).not.toContain(
      TEAM_ORG.id,
    );
    expect(toastSpy).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "success" }),
    );
  });

  it("shows an error toast when deleting the organization fails", async () => {
    mockOrg();
    server.use(getDeleteV2DeleteOrganizationMockHandler422());
    render(<OrganizationSettingsPage />);

    const dangerZone = await screen.findByTestId("org-danger-zone");
    await userEvent.click(
      within(dangerZone).getByRole("button", { name: "Delete organization" }),
    );
    const dialog = await screen.findByRole("dialog");
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Delete organization" }),
    );

    await waitFor(() => {
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Failed to delete organization",
          variant: "destructive",
        }),
      );
    });
  });

  it("shows an error toast when the ownership transfer fails", async () => {
    mockOrg();
    server.use(getPostV2TransferOrganizationOwnershipMockHandler422());
    render(<OrganizationSettingsPage />);

    const dangerZone = await screen.findByTestId("org-danger-zone");
    fireEvent.click(
      within(dangerZone).getByRole("combobox", { name: "New owner" }),
    );
    fireEvent.click(await screen.findByRole("option", { name: "Bob" }));
    await userEvent.click(
      within(dangerZone).getByRole("button", { name: "Transfer" }),
    );
    await userEvent.click(
      await screen.findByRole("button", { name: "Transfer ownership" }),
    );

    await waitFor(() => {
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Failed to transfer ownership",
          variant: "destructive",
        }),
      );
    });
  });

  it("hides the transfer control when the owner is the only member", async () => {
    mockOrg({ members: [OWNER_MEMBER] });
    render(<OrganizationSettingsPage />);

    await screen.findByTestId("org-danger-zone");
    expect(screen.queryByRole("combobox", { name: "New owner" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Transfer" })).toBeNull();
  });

  it("updates the org profile", async () => {
    const patchSpy = vi.fn();
    mockOrg();
    server.use(
      getPatchV2UpdateOrganizationMockHandler(() => {
        patchSpy();
        return { ...TEAM_ORG, name: "Acme Corp" };
      }),
    );
    render(<OrganizationSettingsPage />);

    const nameInput = await screen.findByLabelText("Name");
    await userEvent.clear(nameInput);
    await userEvent.type(nameInput, "Acme Corp");
    await userEvent.click(screen.getByRole("button", { name: "Save changes" }));

    await waitFor(() => {
      expect(patchSpy).toHaveBeenCalledTimes(1);
    });
  });

  it("shows the owner each member's current role in the dropdown for every flag pair", async () => {
    mockOrg({
      members: [
        OWNER_MEMBER,
        PLAIN_MEMBER,
        BILLING_MEMBER,
        ADMIN_MEMBER,
        ADMIN_BILLING_MEMBER,
      ],
    });
    render(<OrganizationSettingsPage />);

    await goToTab("Members");
    await screen.findByText("Bob");

    function roleFor(name: string) {
      return within(
        screen.getByText(name).closest("li") as HTMLElement,
      ).getByRole("combobox");
    }

    expect(roleFor("Bob").textContent?.trim()).toBe("Member");
    expect(roleFor("Billy").textContent?.trim()).toBe("Billing manager");
    expect(roleFor("Adam").textContent?.trim()).toBe("Admin");
    expect(roleFor("Abby").textContent?.trim()).toBe("Admin & billing");
  });

  it("PATCHes both flags (admin false, billing true) when the role becomes billing manager", async () => {
    let sentBody: Record<string, unknown> | undefined;
    mockOrg();
    server.use(
      getPatchV2UpdateMemberRoleMockHandler(async (info) => {
        sentBody = (await info.request.json()) as Record<string, unknown>;
        return { ...PLAIN_MEMBER, is_billing_manager: true };
      }),
    );
    render(<OrganizationSettingsPage />);

    await goToTab("Members");
    const bobRow = (await screen.findByText("Bob")).closest(
      "li",
    ) as HTMLElement;
    fireEvent.click(
      within(bobRow).getByRole("combobox", { name: "Role for Bob" }),
    );
    fireEvent.click(
      await screen.findByRole("option", { name: "Billing manager" }),
    );

    await waitFor(() => {
      expect(sentBody).toBeDefined();
    });
    // Admin and billing are disjoint: switching to billing manager must clear
    // admin and set billing, sending both flags in one PATCH.
    expect(sentBody).toEqual({ is_admin: false, is_billing_manager: true });
    await waitFor(() => {
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({ variant: "success" }),
      );
    });
  });

  it("PATCHes both flags true when the role becomes admin & billing", async () => {
    let sentBody: Record<string, unknown> | undefined;
    mockOrg();
    server.use(
      getPatchV2UpdateMemberRoleMockHandler(async (info) => {
        sentBody = (await info.request.json()) as Record<string, unknown>;
        return { ...PLAIN_MEMBER, is_admin: true, is_billing_manager: true };
      }),
    );
    render(<OrganizationSettingsPage />);

    await goToTab("Members");
    const bobRow = (await screen.findByText("Bob")).closest(
      "li",
    ) as HTMLElement;
    fireEvent.click(
      within(bobRow).getByRole("combobox", { name: "Role for Bob" }),
    );
    fireEvent.click(
      await screen.findByRole("option", { name: "Admin & billing" }),
    );

    await waitFor(() => {
      expect(sentBody).toEqual({ is_admin: true, is_billing_manager: true });
    });
  });

  it("shows a static Owner badge and no role dropdown for the org owner", async () => {
    mockOrg();
    render(<OrganizationSettingsPage />);

    await goToTab("Members");
    const section = await screen.findByTestId("org-members-section");
    const janeRow = (await within(section).findByText(/Jane/)).closest(
      "li",
    ) as HTMLElement;
    expect(within(janeRow).getByText("Owner")).toBeDefined();
    expect(within(janeRow).queryByRole("combobox")).toBeNull();
  });

  it("shows a read-only role badge (not a dropdown) to a plain member", async () => {
    mockOrg({ members: membersViewedAs(PLAIN_MEMBER, [BILLING_MEMBER]) });
    render(<OrganizationSettingsPage />);

    await goToTab("Members");
    const section = await screen.findByTestId("org-members-section");
    const billyRow = (await within(section).findByText("Billy")).closest(
      "li",
    ) as HTMLElement;
    expect(within(billyRow).getByText("Billing manager")).toBeDefined();
    expect(within(billyRow).queryByRole("combobox")).toBeNull();
  });

  it("defaults to the General tab", async () => {
    mockOrg();
    render(<OrganizationSettingsPage />);

    const generalTab = await screen.findByRole("tab", { name: "General" });
    expect(generalTab.getAttribute("data-state")).toBe("active");
    expect(screen.getByTestId("org-profile-section")).toBeDefined();
    // Inactive tabs' sections stay unmounted until selected.
    expect(screen.queryByTestId("org-members-section")).toBeNull();
    expect(screen.queryByTestId("org-invitations-section")).toBeNull();
    expect(screen.queryByTestId("org-team-spend-section")).toBeNull();
  });

  it("reveals the members roster when the Members tab is selected", async () => {
    mockOrg();
    render(<OrganizationSettingsPage />);

    await screen.findByTestId("org-profile-section");
    await goToTab("Members");

    expect(await screen.findByTestId("org-members-section")).toBeDefined();
    expect(screen.queryByTestId("org-invitations-section")).toBeNull();
  });

  it("reveals the invite form when the Invitations tab is selected", async () => {
    mockOrg();
    render(<OrganizationSettingsPage />);

    await screen.findByTestId("org-profile-section");
    await goToTab("Invitations");

    expect(await screen.findByTestId("org-invitations-section")).toBeDefined();
    expect(screen.getByPlaceholderText("teammate@example.com")).toBeDefined();
    expect(screen.queryByTestId("org-members-section")).toBeNull();
  });

  it("keeps the invitation banner visible from every tab", async () => {
    mockOrg({ myInvitations: [PENDING_INVITATION_FROM_OTHER_ORG] });
    render(<OrganizationSettingsPage />);

    expect(await screen.findByTestId("my-invitations-section")).toBeDefined();
    await goToTab("Members");
    expect(screen.getByTestId("my-invitations-section")).toBeDefined();
    await goToTab("Invitations");
    expect(screen.getByTestId("my-invitations-section")).toBeDefined();
    await goToTab("Billing");
    expect(screen.getByTestId("my-invitations-section")).toBeDefined();
  });

  it("opens the tab a ?tab= deep link names", async () => {
    mockOrg();
    mockSearchParams.current = new URLSearchParams({ tab: "members" });
    render(<OrganizationSettingsPage />);

    const membersTab = await screen.findByRole("tab", { name: "Members" });
    expect(membersTab.getAttribute("data-state")).toBe("active");
    expect(await screen.findByTestId("org-members-section")).toBeDefined();
  });

  it("falls back to General for a ?tab=teams link while teams are hidden", async () => {
    mockOrg();
    mockSearchParams.current = new URLSearchParams({ tab: "teams" });
    render(<OrganizationSettingsPage />);

    const generalTab = await screen.findByRole("tab", { name: "General" });
    expect(generalTab.getAttribute("data-state")).toBe("active");
    expect(screen.queryByTestId("org-teams-section")).toBeNull();
  });

  it("ignores an unknown ?tab= value and falls back to the General default", async () => {
    mockOrg();
    mockSearchParams.current = new URLSearchParams({ tab: "bogus" });
    render(<OrganizationSettingsPage />);

    const generalTab = await screen.findByRole("tab", { name: "General" });
    expect(generalTab.getAttribute("data-state")).toBe("active");
  });
});
