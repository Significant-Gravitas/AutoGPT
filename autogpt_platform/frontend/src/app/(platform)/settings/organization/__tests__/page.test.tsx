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
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  getDeleteV2DeleteOrganizationMockHandler,
  getDeleteV2DeleteOrganizationMockHandler422,
  getDeleteV2RemoveMemberFromOrganizationMockHandler,
  getGetV2GetOrganizationDetailsMockHandler,
  getGetV2ListOrganizationMembersMockHandler,
  getGetV2ListWorkspacesMockHandler,
  getPatchV2UpdateMemberRoleMockHandler,
  getPatchV2UpdateOrganizationMockHandler,
  getPostV2TransferOrganizationOwnershipMockHandler,
  getPostV2TransferOrganizationOwnershipMockHandler422,
} from "@/app/api/__generated__/endpoints/orgs/orgs.msw";
import {
  getGetV2ListPendingInvitationsForCurrentUserMockHandler,
  getGetV2ListPendingInvitationsMockHandler,
  getPostV2AcceptInvitationMockHandler,
  getPostV2CreateInvitationMockHandler,
} from "@/app/api/__generated__/endpoints/invitations/invitations.msw";
import type { InvitationResponse } from "@/app/api/__generated__/models/invitationResponse";
import type { OrgMemberResponse } from "@/app/api/__generated__/models/orgMemberResponse";
import type { TeamResponse } from "@/app/api/__generated__/models/teamResponse";
import type { UserInvitationResponse } from "@/app/api/__generated__/models/userInvitationResponse";

import OrganizationSettingsPage from "../page";

// Allow per-test override of the search params Next.js exposes to the page so
// the ?tab= deep-link can be exercised.
const mockSearchParams = { current: new URLSearchParams() };
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

const OWNER_USER_ID = "user-owner";

const toastSpy = vi.hoisted(() => vi.fn());

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({ user: { id: OWNER_USER_ID } }),
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

const TEAM_ORG = {
  id: "org-company",
  name: "Acme Inc",
  slug: "acme",
  avatar_url: null,
  description: "We make everything",
  is_personal: false,
  member_count: 2,
  created_at: new Date("2026-01-01T00:00:00Z"),
};

const PERSONAL_ORG = {
  ...TEAM_ORG,
  id: "org-personal",
  name: "Jane's Org",
  slug: "jane",
  is_personal: true,
  member_count: 1,
};

const OWNER_MEMBER = {
  id: "m-1",
  user_id: OWNER_USER_ID,
  email: "jane@acme.test",
  name: "Jane",
  is_owner: true,
  is_admin: true,
  is_billing_manager: false,
  joined_at: new Date("2026-01-01T00:00:00Z"),
};

const PLAIN_MEMBER = {
  ...OWNER_MEMBER,
  id: "m-2",
  user_id: "user-member",
  email: "bob@acme.test",
  name: "Bob",
  is_owner: false,
  is_admin: false,
};

const DEFAULT_TEAM: TeamResponse = {
  id: "team-default",
  name: "General",
  slug: "general",
  description: null,
  is_default: true,
  join_policy: "OPEN",
  org_id: TEAM_ORG.id,
  member_count: 2,
  created_at: new Date("2026-01-01T00:00:00Z"),
};

const MARKETING_TEAM: TeamResponse = {
  ...DEFAULT_TEAM,
  id: "team-marketing",
  name: "Marketing",
  slug: "marketing",
  is_default: false,
};

const ENGINEERING_TEAM: TeamResponse = {
  ...DEFAULT_TEAM,
  id: "team-engineering",
  name: "Engineering",
  slug: "engineering",
  is_default: false,
};

function seedActiveOrg(orgID: string) {
  useOrgTeamStore.setState({
    activeOrgID: orgID,
    activeTeamID: null,
    orgs: [
      {
        id: TEAM_ORG.id,
        name: TEAM_ORG.name,
        slug: TEAM_ORG.slug,
        avatarUrl: null,
        isPersonal: false,
        memberCount: 2,
      },
    ],
    teams: [],
    isLoaded: true,
  });
}

interface MockTeamOrgArgs {
  members?: OrgMemberResponse[];
  myInvitations?: UserInvitationResponse[];
  orgInvitations?: InvitationResponse[];
}

function mockTeamOrg({
  members = [OWNER_MEMBER, PLAIN_MEMBER],
  myInvitations = [],
  orgInvitations = [],
}: MockTeamOrgArgs = {}) {
  server.use(
    getGetV2GetOrganizationDetailsMockHandler(TEAM_ORG),
    getGetV2ListOrganizationMembersMockHandler(members),
    getGetV2ListWorkspacesMockHandler([]),
    getGetV2ListPendingInvitationsMockHandler(orgInvitations),
    getGetV2ListPendingInvitationsForCurrentUserMockHandler(myInvitations),
  );
}

// Sections live under tabs now: the members list sits on the Members tab, the
// invite flow on its own Invitations tab, and teams on the Teams tab. General
// (the renamed Profile tab) is the default.
async function goToTab(name: "General" | "Members" | "Invitations" | "Teams") {
  await userEvent.click(await screen.findByRole("tab", { name }));
}

describe("OrganizationSettingsPage", () => {
  beforeEach(() => {
    window.localStorage.clear();
    mockSearchParams.current = new URLSearchParams();
    seedActiveOrg(TEAM_ORG.id);
  });

  it("gives an owner all four tabs with profile and danger-zone on General", async () => {
    mockTeamOrg();
    render(<OrganizationSettingsPage />);

    // General tab (default) owns the profile + danger-zone sections.
    expect(await screen.findByTestId("org-profile-section")).toBeDefined();
    expect(screen.getByTestId("org-danger-zone")).toBeDefined();

    // Admins see the full four-tab strip: General / Members / Invitations / Teams.
    expect(screen.getByRole("tab", { name: "General" })).toBeDefined();
    expect(screen.getByRole("tab", { name: "Members" })).toBeDefined();
    expect(screen.getByRole("tab", { name: "Invitations" })).toBeDefined();
    expect(screen.getByRole("tab", { name: "Teams" })).toBeDefined();
  });

  it("keeps the members roster on the Members tab, separate from invitations", async () => {
    mockTeamOrg();
    render(<OrganizationSettingsPage />);

    await goToTab("Members");
    expect(await screen.findByText("Members (2)")).toBeDefined();
    expect(screen.getAllByTestId("org-member-row")).toHaveLength(2);
    expect(screen.getByText("Bob")).toBeDefined();
    // Invitations moved to their own tab — the Members tab no longer hosts them.
    expect(screen.queryByTestId("org-invitations-section")).toBeNull();

    await goToTab("Invitations");
    expect(await screen.findByTestId("org-invitations-section")).toBeDefined();
    expect(screen.getByPlaceholderText("teammate@example.com")).toBeDefined();
  });

  it("hides admin-only sections from a plain member", async () => {
    server.use(
      getGetV2GetOrganizationDetailsMockHandler(TEAM_ORG),
      getGetV2ListOrganizationMembersMockHandler([
        { ...OWNER_MEMBER, user_id: "someone-else" },
        { ...PLAIN_MEMBER, user_id: OWNER_USER_ID },
      ]),
      getGetV2ListPendingInvitationsForCurrentUserMockHandler([]),
    );
    render(<OrganizationSettingsPage />);

    // General tab: no danger zone or profile-edit controls for a plain member.
    await screen.findByTestId("org-profile-section");
    expect(screen.queryByTestId("org-danger-zone")).toBeNull();
    expect(screen.queryByRole("combobox", { name: "New owner" })).toBeNull();
    expect(screen.queryByText("Save changes")).toBeNull();

    // The Invitations tab is admin-only, so a plain member never sees it.
    expect(screen.queryByRole("tab", { name: "Invitations" })).toBeNull();

    // Members tab shows the roster with no invitations section anywhere.
    await goToTab("Members");
    await screen.findByTestId("org-members-section");
    expect(screen.queryByTestId("org-invitations-section")).toBeNull();
  });

  it("shows the solo note instead of members for a personal org", async () => {
    useOrgTeamStore.setState({ activeOrgID: PERSONAL_ORG.id });
    server.use(
      getGetV2GetOrganizationDetailsMockHandler(PERSONAL_ORG),
      getGetV2ListOrganizationMembersMockHandler([OWNER_MEMBER]),
      getGetV2ListPendingInvitationsForCurrentUserMockHandler([]),
    );
    render(<OrganizationSettingsPage />);

    await screen.findByTestId("org-profile-section");
    expect(screen.queryByTestId("org-members-section")).toBeNull();
    expect(screen.queryByTestId("org-danger-zone")).toBeNull();
    expect(
      screen.getByText(/Personal organizations have a single member/),
    ).toBeDefined();
  });

  it("sends an invitation from the invite form", async () => {
    const createSpy = vi.fn();
    mockTeamOrg();
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

  it("pre-assigns the selected teams when sending an invitation", async () => {
    let sentTeamIds: string[] | undefined;
    mockTeamOrg();
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
    mockTeamOrg();
    server.use(getGetV2ListWorkspacesMockHandler([DEFAULT_TEAM]));
    render(<OrganizationSettingsPage />);

    // Resolve + cache the workspaces query via the Teams tab so the invite
    // form on the Invitations tab reads a settled (default-only) team list.
    // Scope to the section: the renamed "General" tab shares the default
    // team's name, so a page-wide lookup would be ambiguous.
    await goToTab("Teams");
    await within(await screen.findByTestId("org-teams-section")).findByText(
      "General",
    );

    await goToTab("Invitations");
    await screen.findByRole("button", { name: "Invite" });
    // The default team is auto-joined, so with no other teams there is nothing
    // to pre-assign — the selector must not render.
    expect(screen.queryByText("Pre-assign to teams")).toBeNull();
    expect(
      screen.queryByRole("group", { name: "Pre-assign to teams" }),
    ).toBeNull();
  });

  it("accepts a pending invitation and switches to the inviting org", async () => {
    const acceptSpy = vi.fn();
    mockTeamOrg({
      myInvitations: [
        {
          id: "inv-9",
          token: "tok-9",
          org_id: "org-other",
          org_name: "Other Corp",
          org_slug: "other-corp",
          is_admin: false,
          is_billing_manager: false,
          expires_at: new Date("2026-08-01T00:00:00Z"),
          created_at: new Date("2026-07-01T00:00:00Z"),
        },
      ],
    });
    server.use(
      getPostV2AcceptInvitationMockHandler(() => {
        acceptSpy();
        return { orgId: "org-other", message: "Invitation accepted" };
      }),
    );
    render(<OrganizationSettingsPage />);

    expect(await screen.findByText("Other Corp")).toBeDefined();
    await userEvent.click(screen.getByRole("button", { name: "Accept" }));

    await waitFor(() => {
      expect(acceptSpy).toHaveBeenCalledTimes(1);
      expect(useOrgTeamStore.getState().activeOrgID).toBe("org-other");
    });
  });

  it("removes a member after confirmation", async () => {
    const removeSpy = vi.fn();
    mockTeamOrg();
    server.use(
      getDeleteV2RemoveMemberFromOrganizationMockHandler(() => {
        removeSpy();
        return undefined;
      }),
    );
    render(<OrganizationSettingsPage />);

    await goToTab("Members");
    await screen.findByText("Bob");
    await userEvent.click(screen.getByRole("button", { name: "Remove" }));
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
    mockTeamOrg();
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
      orgs: [
        {
          id: TEAM_ORG.id,
          name: TEAM_ORG.name,
          slug: TEAM_ORG.slug,
          avatarUrl: null,
          isPersonal: false,
          memberCount: 2,
        },
        {
          id: PERSONAL_ORG.id,
          name: PERSONAL_ORG.name,
          slug: PERSONAL_ORG.slug,
          avatarUrl: null,
          isPersonal: true,
          memberCount: 1,
        },
      ],
    });
    mockTeamOrg();
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
    mockTeamOrg();
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
    mockTeamOrg();
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
    mockTeamOrg({ members: [OWNER_MEMBER] });
    render(<OrganizationSettingsPage />);

    await screen.findByTestId("org-danger-zone");
    expect(screen.queryByRole("combobox", { name: "New owner" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Transfer" })).toBeNull();
  });

  it("updates the org profile", async () => {
    const patchSpy = vi.fn();
    mockTeamOrg();
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

  it("grants billing-manager to a member via an independent toggle (PATCHes is_billing_manager only)", async () => {
    let sentBody: Record<string, unknown> | undefined;
    mockTeamOrg();
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
    await userEvent.click(
      within(bobRow).getByRole("switch", { name: "Billing manager for Bob" }),
    );

    await waitFor(() => {
      expect(sentBody).toBeDefined();
    });
    // Independent of the role Select — only the billing flag is sent.
    expect(sentBody).toEqual({ is_billing_manager: true });
    expect(toastSpy).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "success" }),
    );
  });

  it("shows a read-only Billing badge (not a toggle) to a plain member", async () => {
    // Viewer is a plain member, so no row is manageable: the billing status
    // renders as a badge rather than an editable switch.
    server.use(
      getGetV2GetOrganizationDetailsMockHandler(TEAM_ORG),
      getGetV2ListOrganizationMembersMockHandler([
        { ...OWNER_MEMBER, user_id: "someone-else", is_billing_manager: true },
        { ...PLAIN_MEMBER, user_id: OWNER_USER_ID },
      ]),
      getGetV2ListPendingInvitationsForCurrentUserMockHandler([]),
    );
    render(<OrganizationSettingsPage />);

    await goToTab("Members");
    const section = await screen.findByTestId("org-members-section");
    const janeRow = (await within(section).findByText("Jane")).closest(
      "li",
    ) as HTMLElement;
    expect(within(janeRow).getByText("Billing")).toBeDefined();
    expect(within(janeRow).queryByRole("switch")).toBeNull();
  });

  it("defaults to the General tab", async () => {
    mockTeamOrg();
    render(<OrganizationSettingsPage />);

    const generalTab = await screen.findByRole("tab", { name: "General" });
    expect(generalTab.getAttribute("data-state")).toBe("active");
    expect(screen.getByTestId("org-profile-section")).toBeDefined();
    // Inactive tabs' sections stay unmounted until selected.
    expect(screen.queryByTestId("org-members-section")).toBeNull();
    expect(screen.queryByTestId("org-invitations-section")).toBeNull();
    expect(screen.queryByTestId("org-teams-section")).toBeNull();
  });

  it("reveals the members roster when the Members tab is selected", async () => {
    mockTeamOrg();
    render(<OrganizationSettingsPage />);

    await screen.findByTestId("org-profile-section");
    await goToTab("Members");

    expect(await screen.findByTestId("org-members-section")).toBeDefined();
    // The invite form lives on the Invitations tab, not here.
    expect(screen.queryByTestId("org-invitations-section")).toBeNull();
  });

  it("reveals the invite form when the Invitations tab is selected", async () => {
    mockTeamOrg();
    render(<OrganizationSettingsPage />);

    await screen.findByTestId("org-profile-section");
    await goToTab("Invitations");

    expect(await screen.findByTestId("org-invitations-section")).toBeDefined();
    expect(screen.getByPlaceholderText("teammate@example.com")).toBeDefined();
    expect(screen.queryByTestId("org-members-section")).toBeNull();
  });

  it("spells out assigned team names on a pending invitation", async () => {
    mockTeamOrg({
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
    // Team names are spelled out rather than shown as a "+N teams" count.
    expect(within(row).getByText(/Marketing, Engineering/)).toBeDefined();
    expect(within(row).queryByText(/\+\d+ teams?/)).toBeNull();
  });

  it("keeps the invitation banner visible from every tab", async () => {
    mockTeamOrg({
      myInvitations: [
        {
          id: "inv-banner",
          token: "tok-banner",
          org_id: "org-other",
          org_name: "Other Corp",
          org_slug: "other-corp",
          is_admin: false,
          is_billing_manager: false,
          expires_at: new Date("2026-08-01T00:00:00Z"),
          created_at: new Date("2026-07-01T00:00:00Z"),
        },
      ],
    });
    render(<OrganizationSettingsPage />);

    expect(await screen.findByTestId("my-invitations-section")).toBeDefined();
    await goToTab("Members");
    expect(screen.getByTestId("my-invitations-section")).toBeDefined();
    await goToTab("Invitations");
    expect(screen.getByTestId("my-invitations-section")).toBeDefined();
    await goToTab("Teams");
    expect(screen.getByTestId("my-invitations-section")).toBeDefined();
  });

  it("renders a personal org without tab chrome", async () => {
    useOrgTeamStore.setState({ activeOrgID: PERSONAL_ORG.id });
    server.use(
      getGetV2GetOrganizationDetailsMockHandler(PERSONAL_ORG),
      getGetV2ListOrganizationMembersMockHandler([OWNER_MEMBER]),
      getGetV2ListPendingInvitationsForCurrentUserMockHandler([]),
    );
    render(<OrganizationSettingsPage />);

    await screen.findByTestId("org-profile-section");
    expect(screen.queryByRole("tablist")).toBeNull();
    expect(screen.queryByRole("tab")).toBeNull();
  });

  it("opens the Teams tab from a ?tab=teams deep link", async () => {
    mockTeamOrg();
    mockSearchParams.current = new URLSearchParams({ tab: "teams" });
    render(<OrganizationSettingsPage />);

    const teamsTab = await screen.findByRole("tab", { name: "Teams" });
    expect(teamsTab.getAttribute("data-state")).toBe("active");
    expect(await screen.findByTestId("org-teams-section")).toBeDefined();
  });

  it("ignores an unknown ?tab= value and falls back to the General default", async () => {
    mockTeamOrg();
    mockSearchParams.current = new URLSearchParams({ tab: "bogus" });
    render(<OrganizationSettingsPage />);

    const generalTab = await screen.findByRole("tab", { name: "General" });
    expect(generalTab.getAttribute("data-state")).toBe("active");
  });
});
