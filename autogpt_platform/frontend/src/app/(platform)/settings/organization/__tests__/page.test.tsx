import { useOrgTeamStore } from "@/services/org-team/store";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  getDeleteV2RemoveMemberFromOrganizationMockHandler,
  getGetV2GetOrganizationDetailsMockHandler,
  getGetV2ListOrganizationMembersMockHandler,
  getGetV2ListWorkspacesMockHandler,
  getPatchV2UpdateOrganizationMockHandler,
} from "@/app/api/__generated__/endpoints/orgs/orgs.msw";
import {
  getGetV2ListPendingInvitationsForCurrentUserMockHandler,
  getGetV2ListPendingInvitationsMockHandler,
  getPostV2AcceptInvitationMockHandler,
  getPostV2CreateInvitationMockHandler,
} from "@/app/api/__generated__/endpoints/invitations/invitations.msw";
import type { InvitationResponse } from "@/app/api/__generated__/models/invitationResponse";
import type { OrgMemberResponse } from "@/app/api/__generated__/models/orgMemberResponse";
import type { UserInvitationResponse } from "@/app/api/__generated__/models/userInvitationResponse";

import OrganizationSettingsPage from "../page";

const OWNER_USER_ID = "user-owner";

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: OWNER_USER_ID },
    isLoggedIn: true,
    isUserLoading: false,
  }),
}));

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

describe("OrganizationSettingsPage", () => {
  beforeEach(() => {
    window.localStorage.clear();
    seedActiveOrg(TEAM_ORG.id);
  });

  it("renders profile, members and invitations for an owner", async () => {
    mockTeamOrg();
    render(<OrganizationSettingsPage />);

    expect(await screen.findByTestId("org-profile-section")).toBeDefined();
    expect(await screen.findByText("Members (2)")).toBeDefined();
    expect(screen.getByTestId("org-invitations-section")).toBeDefined();
    expect(screen.getByTestId("org-danger-zone")).toBeDefined();
    expect(screen.getAllByTestId("org-member-row")).toHaveLength(2);
    expect(screen.getByText("Bob")).toBeDefined();
  });

  it("hides admin-only sections from a plain member", async () => {
    server.use(
      getGetV2GetOrganizationDetailsMockHandler(TEAM_ORG),
      getGetV2ListOrganizationMembersMockHandler([
        { ...OWNER_MEMBER, user_id: "someone-else" },
        { ...PLAIN_MEMBER, user_id: OWNER_USER_ID },
      ]),
      getGetV2ListWorkspacesMockHandler([]),
      getGetV2ListPendingInvitationsForCurrentUserMockHandler([]),
    );
    render(<OrganizationSettingsPage />);

    await screen.findByTestId("org-profile-section");
    expect(screen.queryByTestId("org-invitations-section")).toBeNull();
    expect(screen.queryByTestId("org-danger-zone")).toBeNull();
    expect(screen.queryByText("Save changes")).toBeNull();
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

    const emailInput = await screen.findByPlaceholderText(
      "teammate@example.com",
    );
    await userEvent.type(emailInput, "new@acme.test");
    await userEvent.click(screen.getByRole("button", { name: "Invite" }));

    await waitFor(() => {
      expect(createSpy).toHaveBeenCalledTimes(1);
    });
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

    await screen.findByText("Bob");
    await userEvent.click(screen.getByRole("button", { name: "Remove" }));
    await userEvent.click(
      await screen.findByRole("button", { name: "Remove member" }),
    );

    await waitFor(() => {
      expect(removeSpy).toHaveBeenCalledTimes(1);
    });
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
});
