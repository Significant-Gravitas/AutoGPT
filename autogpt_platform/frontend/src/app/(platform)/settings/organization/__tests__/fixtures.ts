import {
  getGetV2GetOrganizationDetailsMockHandler,
  getGetV2GetPersonalOrganizationContextMockHandler,
  getGetV2ListOrganizationMembersMockHandler,
  getGetV2ListWorkspacesMockHandler,
} from "@/app/api/__generated__/endpoints/orgs/orgs.msw";
import {
  getGetV2ListPendingInvitationsForCurrentUserMockHandler,
  getGetV2ListPendingInvitationsMockHandler,
} from "@/app/api/__generated__/endpoints/invitations/invitations.msw";
import type { InvitationResponse } from "@/app/api/__generated__/models/invitationResponse";
import type { OrgMemberResponse } from "@/app/api/__generated__/models/orgMemberResponse";
import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import type { TeamResponse } from "@/app/api/__generated__/models/teamResponse";
import type { UserInvitationResponse } from "@/app/api/__generated__/models/userInvitationResponse";
import { server } from "@/mocks/mock-server";
import { useOrgTeamStore } from "@/services/org-team/store";

const OWNER_USER_ID = "user-owner";

export const TEAM_ORG: OrgResponse = {
  id: "org-company",
  name: "Acme Inc",
  slug: "acme",
  avatar_url: null,
  description: "We make everything",
  is_personal: false,
  member_count: 2,
  created_at: new Date("2026-01-01T00:00:00Z"),
};

export const PERSONAL_ORG: OrgResponse = {
  ...TEAM_ORG,
  id: "org-personal",
  name: "Jane's Org",
  slug: "jane",
  is_personal: true,
  member_count: 1,
};

export const OTHER_ORG: OrgResponse = {
  ...TEAM_ORG,
  id: "org-other",
  name: "Other Corp",
  slug: "other-corp",
};

export const OWNER_MEMBER: OrgMemberResponse = {
  id: "m-1",
  user_id: OWNER_USER_ID,
  email: "jane@acme.test",
  name: "Jane",
  is_owner: true,
  is_admin: true,
  is_billing_manager: false,
  joined_at: new Date("2026-01-01T00:00:00Z"),
};

export const PLAIN_MEMBER: OrgMemberResponse = {
  ...OWNER_MEMBER,
  id: "m-2",
  user_id: "user-member",
  email: "bob@acme.test",
  name: "Bob",
  is_owner: false,
  is_admin: false,
};

export const BILLING_MEMBER: OrgMemberResponse = {
  ...PLAIN_MEMBER,
  id: "m-3",
  user_id: "user-billing",
  email: "billy@acme.test",
  name: "Billy",
  is_billing_manager: true,
};

export const ADMIN_MEMBER: OrgMemberResponse = {
  ...PLAIN_MEMBER,
  id: "m-4",
  user_id: "user-admin",
  email: "adam@acme.test",
  name: "Adam",
  is_admin: true,
};

export const ADMIN_BILLING_MEMBER: OrgMemberResponse = {
  ...PLAIN_MEMBER,
  id: "m-5",
  user_id: "user-admin-billing",
  email: "abby@acme.test",
  name: "Abby",
  is_admin: true,
  is_billing_manager: true,
};

export const DEFAULT_TEAM: TeamResponse = {
  id: "team-default",
  name: "General",
  slug: "general",
  description: null,
  is_default: true,
  join_policy: "OPEN",
  org_id: TEAM_ORG.id,
  member_count: 2,
  is_member: true,
  created_at: new Date("2026-01-01T00:00:00Z"),
};

export const MARKETING_TEAM: TeamResponse = {
  ...DEFAULT_TEAM,
  id: "team-marketing",
  name: "Marketing",
  slug: "marketing",
  is_default: false,
};

export const ENGINEERING_TEAM: TeamResponse = {
  ...DEFAULT_TEAM,
  id: "team-engineering",
  name: "Engineering",
  slug: "engineering",
  is_default: false,
};

export const PENDING_INVITATION_FROM_OTHER_ORG: UserInvitationResponse = {
  id: "inv-9",
  token: "tok-9",
  org_id: OTHER_ORG.id,
  org_name: OTHER_ORG.name,
  org_slug: OTHER_ORG.slug,
  is_admin: false,
  is_billing_manager: false,
  expires_at: new Date("2026-08-01T00:00:00Z"),
  created_at: new Date("2026-07-01T00:00:00Z"),
};

export function storeOrg(org: OrgResponse) {
  return {
    id: org.id,
    name: org.name,
    slug: org.slug,
    avatarUrl: null,
    isPersonal: org.is_personal,
    memberCount: org.member_count,
  };
}

export function seedActiveOrg(org: OrgResponse = TEAM_ORG) {
  useOrgTeamStore.setState({
    activeOrgID: org.id,
    activeTeamID: null,
    orgs: [storeOrg(org)],
    teams: [],
    isLoaded: true,
  });
}

interface MockOrgArgs {
  org?: OrgResponse;
  members?: OrgMemberResponse[];
  myInvitations?: UserInvitationResponse[];
  orgInvitations?: InvitationResponse[];
  canCreateOrganization?: boolean;
}

export function mockOrg({
  org = TEAM_ORG,
  members = [OWNER_MEMBER, PLAIN_MEMBER],
  myInvitations = [],
  orgInvitations = [],
  canCreateOrganization = true,
}: MockOrgArgs = {}) {
  mockOrgDetails(org, canCreateOrganization);
  server.use(
    getGetV2ListOrganizationMembersMockHandler(members),
    getGetV2ListWorkspacesMockHandler([]),
    getGetV2ListPendingInvitationsMockHandler(orgInvitations),
    getGetV2ListPendingInvitationsForCurrentUserMockHandler(myInvitations),
  );
}

// GET /orgs/:orgId also matches GET /orgs/default, so the personal-context
// handler has to come first whenever the details handler is replaced.
export function mockOrgDetails(
  org: OrgResponse | (() => OrgResponse),
  canCreateOrganization = true,
) {
  server.use(
    getGetV2GetPersonalOrganizationContextMockHandler({
      ...PERSONAL_ORG,
      can_create_organization: canCreateOrganization,
    }),
    getGetV2GetOrganizationDetailsMockHandler(
      typeof org === "function" ? () => org() : org,
    ),
  );
}

// The viewer is always OWNER_USER_ID (see each file's useAuth mock), so a
// test changes the viewer's role by giving that id to another member row.
export function membersViewedAs(
  viewer: OrgMemberResponse,
  others: OrgMemberResponse[] = [],
): OrgMemberResponse[] {
  return [
    { ...OWNER_MEMBER, user_id: "someone-else", name: "Olivia" },
    { ...viewer, user_id: OWNER_USER_ID },
    ...others,
  ];
}
