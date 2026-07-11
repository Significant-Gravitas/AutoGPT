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
  getDeleteV2DeleteWorkspaceMockHandler,
  getDeleteV2RemoveMemberFromWorkspaceMockHandler,
  getGetV2GetOrganizationDetailsMockHandler,
  getGetV2ListOrganizationAliasesMockHandler,
  getGetV2ListOrganizationMembersMockHandler,
  getGetV2ListWorkspaceMembersMockHandler,
  getGetV2ListWorkspacesMockHandler,
  getPatchV2UpdateWorkspaceMockHandler,
  getPostV2AddMemberToWorkspaceMockHandler,
  getPostV2CreateWorkspaceMockHandler,
  getPostV2LeaveWorkspaceMockHandler,
  getPostV2SelfJoinOpenWorkspaceMockHandler,
} from "@/app/api/__generated__/endpoints/orgs/orgs.msw";
import {
  getGetV2ListPendingInvitationsForCurrentUserMockHandler,
  getGetV2ListPendingInvitationsMockHandler,
} from "@/app/api/__generated__/endpoints/invitations/invitations.msw";
import type { OrgMemberResponse } from "@/app/api/__generated__/models/orgMemberResponse";
import type { TeamMemberResponse } from "@/app/api/__generated__/models/teamMemberResponse";
import type { TeamResponse } from "@/app/api/__generated__/models/teamResponse";

import OrganizationSettingsPage from "../page";

const OWNER_USER_ID = "user-owner";

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({ user: { id: OWNER_USER_ID } }),
}));

const TEAM_ORG = {
  id: "org-company",
  name: "Acme Inc",
  slug: "acme",
  avatar_url: null,
  description: "We make everything",
  is_personal: false,
  member_count: 3,
  created_at: new Date("2026-01-01T00:00:00Z"),
};

const OWNER_MEMBER: OrgMemberResponse = {
  id: "m-1",
  user_id: OWNER_USER_ID,
  email: "jane@acme.test",
  name: "Jane",
  is_owner: true,
  is_admin: true,
  is_billing_manager: false,
  joined_at: new Date("2026-01-01T00:00:00Z"),
};

const BOB_MEMBER: OrgMemberResponse = {
  ...OWNER_MEMBER,
  id: "m-2",
  user_id: "user-bob",
  email: "bob@acme.test",
  name: "Bob",
  is_owner: false,
  is_admin: false,
};

const CARL_MEMBER: OrgMemberResponse = {
  ...BOB_MEMBER,
  id: "m-3",
  user_id: "user-carl",
  email: "carl@acme.test",
  name: "Carl",
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

const OPEN_TEAM: TeamResponse = {
  ...DEFAULT_TEAM,
  id: "team-open",
  name: "Marketing",
  slug: "marketing",
  is_default: false,
  join_policy: "OPEN",
  member_count: 4,
};

const PRIVATE_TEAM: TeamResponse = {
  ...DEFAULT_TEAM,
  id: "team-private",
  name: "Skunkworks",
  slug: null,
  is_default: false,
  join_policy: "PRIVATE",
  member_count: 1,
};

function teamMember(over: Partial<TeamMemberResponse>): TeamMemberResponse {
  return {
    id: `tm-${over.user_id}`,
    user_id: "user-x",
    email: "x@acme.test",
    name: "X",
    is_admin: false,
    is_billing_manager: false,
    joined_at: new Date("2026-01-01T00:00:00Z"),
    ...over,
  };
}

function seedActiveOrg() {
  useOrgTeamStore.setState({
    activeOrgID: TEAM_ORG.id,
    activeTeamID: null,
    orgs: [
      {
        id: TEAM_ORG.id,
        name: TEAM_ORG.name,
        slug: TEAM_ORG.slug,
        avatarUrl: null,
        isPersonal: false,
        memberCount: 3,
      },
    ],
    teams: [],
    isLoaded: true,
  });
}

interface MockOrgArgs {
  members?: OrgMemberResponse[];
  teams?: TeamResponse[];
}

function mockOrg({
  members = [OWNER_MEMBER, BOB_MEMBER, CARL_MEMBER],
  teams = [DEFAULT_TEAM, OPEN_TEAM, PRIVATE_TEAM],
}: MockOrgArgs = {}) {
  server.use(
    getGetV2GetOrganizationDetailsMockHandler(TEAM_ORG),
    getGetV2ListOrganizationMembersMockHandler(members),
    getGetV2ListWorkspacesMockHandler(teams),
    getGetV2ListOrganizationAliasesMockHandler([]),
    getGetV2ListPendingInvitationsMockHandler([]),
    getGetV2ListPendingInvitationsForCurrentUserMockHandler([]),
  );
}

// TeamsSection now lives on the "Teams" tab; activate it before reading the
// section. Clicking again once active is a no-op, so this is safe to call from
// every teams-section lookup.
async function showTeamsTab() {
  await userEvent.click(await screen.findByRole("tab", { name: "Teams" }));
  return screen.findByTestId("org-teams-section");
}

// The teams list resolves from its own query, a tick after the page's
// org/members queries settle the section. Await the row before using it.
// Scope to the teams section — the invitations section also lists team
// names (its pre-assign selector), so a page-wide lookup is ambiguous.
async function teamRow(teamName: string) {
  const section = await showTeamsTab();
  const label = await within(section).findByText(teamName);
  return label.closest("li") as HTMLElement;
}

async function openManagePanel(teamName: string) {
  const row = await teamRow(teamName);
  await userEvent.click(within(row).getByTestId("manage-team-button"));
  return screen.findByTestId("team-manage-panel");
}

describe("TeamsSection", () => {
  beforeEach(() => {
    window.localStorage.clear();
    seedActiveOrg();
  });

  it("lists teams with default, open and private badges", async () => {
    mockOrg();
    render(<OrganizationSettingsPage />);

    const section = await showTeamsTab();
    await within(section).findByText("Marketing");
    expect(within(section).getByText("Teams (3)")).toBeDefined();
    expect(within(section).getAllByTestId("org-team-row")).toHaveLength(3);
    expect(within(section).getByText("General")).toBeDefined();
    expect(within(section).getByText("Marketing")).toBeDefined();
    expect(within(section).getByText("Skunkworks")).toBeDefined();
    expect(within(section).getByText("Default")).toBeDefined();
    expect(within(section).getAllByText("Open")).toHaveLength(2);
    expect(within(section).getByText("Private")).toBeDefined();
    expect(within(section).getByText("4 members")).toBeDefined();
  });

  it("creates a team from the dialog (posts to the API)", async () => {
    const createSpy = vi.fn();
    mockOrg();
    server.use(
      getPostV2CreateWorkspaceMockHandler(() => {
        createSpy();
        return { ...OPEN_TEAM, id: "team-new", name: "Engineering" };
      }),
    );
    render(<OrganizationSettingsPage />);

    await showTeamsTab();
    await userEvent.click(screen.getByTestId("create-team-button"));

    const dialog = await screen.findByRole("dialog");
    await userEvent.type(
      within(dialog).getByPlaceholderText("Engineering"),
      "Engineering",
    );
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Create team" }),
    );

    await waitFor(() => {
      expect(createSpy).toHaveBeenCalledTimes(1);
    });
  });

  it("joins an open team from its row", async () => {
    const joinSpy = vi.fn();
    mockOrg();
    server.use(
      getPostV2SelfJoinOpenWorkspaceMockHandler(() => {
        joinSpy();
        return OPEN_TEAM;
      }),
    );
    render(<OrganizationSettingsPage />);

    await showTeamsTab();
    const row = await teamRow("Marketing");
    await userEvent.click(within(row).getByRole("button", { name: "Join" }));

    await waitFor(() => {
      expect(joinSpy).toHaveBeenCalledTimes(1);
    });
  });

  it("deletes a non-default team after confirmation", async () => {
    const deleteSpy = vi.fn();
    mockOrg();
    server.use(
      getDeleteV2DeleteWorkspaceMockHandler(() => {
        deleteSpy();
      }),
    );
    render(<OrganizationSettingsPage />);

    await showTeamsTab();
    const row = await teamRow("Skunkworks");
    await userEvent.click(within(row).getByRole("button", { name: "Delete" }));

    const dialog = await screen.findByRole("dialog");
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Delete team" }),
    );

    await waitFor(() => {
      expect(deleteSpy).toHaveBeenCalledTimes(1);
    });
  });

  it("hides create and delete controls from a plain org member", async () => {
    mockOrg({
      members: [
        { ...OWNER_MEMBER, user_id: "someone-else" },
        { ...BOB_MEMBER, user_id: OWNER_USER_ID },
      ],
    });
    render(<OrganizationSettingsPage />);

    const section = await showTeamsTab();
    expect(screen.queryByTestId("create-team-button")).toBeNull();
    expect(
      within(section).queryByRole("button", { name: "Delete" }),
    ).toBeNull();
  });

  it("shows admin management affordances in the manage panel for a team admin", async () => {
    mockOrg();
    server.use(
      getGetV2ListWorkspaceMembersMockHandler([
        teamMember({ user_id: OWNER_USER_ID, name: "Jane", is_admin: true }),
      ]),
    );
    render(<OrganizationSettingsPage />);

    await showTeamsTab();
    const panel = await openManagePanel("Marketing");

    expect(
      await within(panel).findByRole("button", { name: "Save changes" }),
    ).toBeDefined();
    expect(
      within(panel).getByRole("combobox", { name: "Add a member" }),
    ).toBeDefined();
    expect(within(panel).queryByTestId("team-leave-button")).toBeNull();
  });

  it("shows a read-only view with a Leave action for a non-admin member", async () => {
    mockOrg();
    server.use(
      getGetV2ListWorkspaceMembersMockHandler([
        teamMember({ user_id: OWNER_USER_ID, name: "Jane", is_admin: false }),
        teamMember({ user_id: "user-carl", name: "Carl", is_admin: true }),
      ]),
    );
    render(<OrganizationSettingsPage />);

    await showTeamsTab();
    const panel = await openManagePanel("Marketing");

    expect(await within(panel).findByTestId("team-leave-button")).toBeDefined();
    expect(
      within(panel).queryByRole("button", { name: "Save changes" }),
    ).toBeNull();
    expect(
      within(panel).queryByRole("combobox", { name: "Add a member" }),
    ).toBeNull();
  });

  it("renames a team and sends the X-Team-Id header for the target team", async () => {
    const patchSpy = vi.fn();
    let sentTeamHeader: string | null = null;
    mockOrg();
    server.use(
      getGetV2ListWorkspaceMembersMockHandler([
        teamMember({ user_id: OWNER_USER_ID, name: "Jane", is_admin: true }),
      ]),
      getPatchV2UpdateWorkspaceMockHandler((info) => {
        patchSpy();
        sentTeamHeader = info.request.headers.get("X-Team-Id");
        return { ...OPEN_TEAM, name: "Growth" };
      }),
    );
    render(<OrganizationSettingsPage />);

    await showTeamsTab();
    const panel = await openManagePanel("Marketing");

    const nameInput = await within(panel).findByLabelText("Name");
    await userEvent.clear(nameInput);
    await userEvent.type(nameInput, "Growth");
    await userEvent.click(
      within(panel).getByRole("button", { name: "Save changes" }),
    );

    await waitFor(() => {
      expect(patchSpy).toHaveBeenCalledTimes(1);
    });
    expect(sentTeamHeader).toBe(OPEN_TEAM.id);
  });

  it("adds an org member to the team from the manage panel", async () => {
    const addSpy = vi.fn();
    mockOrg();
    server.use(
      getGetV2ListWorkspaceMembersMockHandler([
        teamMember({ user_id: OWNER_USER_ID, name: "Jane", is_admin: true }),
      ]),
      getPostV2AddMemberToWorkspaceMockHandler(() => {
        addSpy();
        return teamMember({ user_id: "user-bob", name: "Bob" });
      }),
    );
    render(<OrganizationSettingsPage />);

    await showTeamsTab();
    const panel = await openManagePanel("Marketing");

    const combo = await within(panel).findByRole("combobox", {
      name: "Add a member",
    });
    fireEvent.click(combo);
    fireEvent.click(await screen.findByRole("option", { name: "Bob" }));
    await userEvent.click(within(panel).getByRole("button", { name: "Add" }));

    await waitFor(() => {
      expect(addSpy).toHaveBeenCalledTimes(1);
    });
  });

  it("removes a team member after confirmation", async () => {
    const removeSpy = vi.fn();
    mockOrg();
    server.use(
      getGetV2ListWorkspaceMembersMockHandler([
        teamMember({ user_id: OWNER_USER_ID, name: "Jane", is_admin: true }),
        teamMember({ user_id: "user-carl", name: "Carl" }),
      ]),
      getDeleteV2RemoveMemberFromWorkspaceMockHandler(() => {
        removeSpy();
      }),
    );
    render(<OrganizationSettingsPage />);

    await showTeamsTab();
    const panel = await openManagePanel("Marketing");

    const carlRow = (await within(panel).findByText("Carl")).closest(
      "li",
    ) as HTMLElement;
    await userEvent.click(
      within(carlRow).getByRole("button", { name: "Remove" }),
    );

    const dialog = await screen.findByRole("dialog");
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Remove member" }),
    );

    await waitFor(() => {
      expect(removeSpy).toHaveBeenCalledTimes(1);
    });
  });

  it("lets a non-admin leave a team without touching global active-team state", async () => {
    const leaveSpy = vi.fn();
    mockOrg();
    server.use(
      getGetV2ListWorkspaceMembersMockHandler([
        teamMember({ user_id: OWNER_USER_ID, name: "Jane", is_admin: false }),
      ]),
      getPostV2LeaveWorkspaceMockHandler(() => {
        leaveSpy();
      }),
    );
    render(<OrganizationSettingsPage />);

    await showTeamsTab();
    const panel = await openManagePanel("Marketing");

    await userEvent.click(
      await within(panel).findByTestId("team-leave-button"),
    );
    const dialog = await screen.findByRole("dialog");
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Leave team" }),
    );

    await waitFor(() => {
      expect(leaveSpy).toHaveBeenCalledTimes(1);
    });
    expect(useOrgTeamStore.getState().activeTeamID).toBeNull();
  });
});
