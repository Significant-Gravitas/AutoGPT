import { useOrgTeamStore } from "@/services/org-team/store";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  getGetV2GetOrganizationDetailsMockHandler,
  getGetV2ListOrganizationAliasesMockHandler,
  getGetV2ListOrganizationMembersMockHandler,
  getGetV2ListWorkspacesMockHandler,
  getPostV2CreateOrganizationAliasMockHandler,
} from "@/app/api/__generated__/endpoints/orgs/orgs.msw";
import {
  getGetV2ListPendingInvitationsForCurrentUserMockHandler,
  getGetV2ListPendingInvitationsMockHandler,
} from "@/app/api/__generated__/endpoints/invitations/invitations.msw";
import type { OrgAliasResponse } from "@/app/api/__generated__/models/orgAliasResponse";
import type { OrgMemberResponse } from "@/app/api/__generated__/models/orgMemberResponse";

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
  member_count: 2,
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

const PLAIN_MEMBER: OrgMemberResponse = {
  ...OWNER_MEMBER,
  id: "m-2",
  user_id: "user-bob",
  email: "bob@acme.test",
  name: "Bob",
  is_owner: false,
  is_admin: false,
};

const MANUAL_ALIAS: OrgAliasResponse = {
  id: "alias-1",
  alias_slug: "acme-old",
  alias_type: "MANUAL",
  created_at: new Date("2026-02-01T00:00:00Z"),
};

const RENAME_ALIAS: OrgAliasResponse = {
  id: "alias-2",
  alias_slug: "acme-legacy",
  alias_type: "RENAME",
  created_at: new Date("2026-03-01T00:00:00Z"),
};

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
        memberCount: 2,
      },
    ],
    teams: [],
    isLoaded: true,
  });
}

interface MockOrgArgs {
  members?: OrgMemberResponse[];
  aliases?: OrgAliasResponse[];
}

function mockOrg({
  members = [OWNER_MEMBER, PLAIN_MEMBER],
  aliases = [MANUAL_ALIAS, RENAME_ALIAS],
}: MockOrgArgs = {}) {
  server.use(
    getGetV2GetOrganizationDetailsMockHandler(TEAM_ORG),
    getGetV2ListOrganizationMembersMockHandler(members),
    getGetV2ListWorkspacesMockHandler([]),
    getGetV2ListPendingInvitationsMockHandler([]),
    getGetV2ListPendingInvitationsForCurrentUserMockHandler([]),
    getGetV2ListOrganizationAliasesMockHandler(aliases),
  );
}

async function aliasesSection() {
  const section = await screen.findByTestId("org-aliases-section");
  // Aliases resolve from their own query, a tick after the org/members
  // queries settle the section shell. Await a row before asserting on it.
  await within(section).findByText(MANUAL_ALIAS.alias_slug);
  return section;
}

describe("AliasesSection", () => {
  beforeEach(() => {
    window.localStorage.clear();
    seedActiveOrg();
  });

  it("lists existing aliases with an add form for an admin", async () => {
    mockOrg();
    render(<OrganizationSettingsPage />);

    const section = await aliasesSection();
    expect(within(section).getAllByTestId("org-alias-row")).toHaveLength(2);
    expect(within(section).getByText("acme-old")).toBeDefined();
    expect(within(section).getByText("acme-legacy")).toBeDefined();
    // The RENAME alias is tagged so admins know it was auto-created.
    expect(within(section).getByText("From rename")).toBeDefined();
    expect(
      within(section).getByRole("button", { name: "Add alias" }),
    ).toBeDefined();
  });

  it("adds an alias, posting the slug, then refetches the list", async () => {
    let sentSlug: string | undefined;
    let created = false;
    mockOrg();
    server.use(
      getGetV2ListOrganizationAliasesMockHandler(() =>
        created
          ? [
              MANUAL_ALIAS,
              RENAME_ALIAS,
              { ...MANUAL_ALIAS, id: "alias-3", alias_slug: "acme-original" },
            ]
          : [MANUAL_ALIAS, RENAME_ALIAS],
      ),
      getPostV2CreateOrganizationAliasMockHandler(async (info) => {
        const body = (await info.request.json()) as { alias_slug?: string };
        sentSlug = body.alias_slug;
        created = true;
        return {
          id: "alias-3",
          alias_slug: body.alias_slug ?? "",
          alias_type: "MANUAL",
          created_at: new Date("2026-04-01T00:00:00Z"),
        };
      }),
    );
    render(<OrganizationSettingsPage />);

    const section = await aliasesSection();
    await userEvent.type(
      within(section).getByPlaceholderText("old-slug"),
      "acme-original",
    );
    await userEvent.click(
      within(section).getByRole("button", { name: "Add alias" }),
    );

    await waitFor(() => {
      expect(sentSlug).toBe("acme-original");
    });
    // Refetch surfaces the newly created alias in the list.
    expect(await within(section).findByText("acme-original")).toBeDefined();
  });

  it("shows the alias list read-only without an add form for a non-admin member", async () => {
    mockOrg({
      members: [
        { ...OWNER_MEMBER, user_id: "someone-else" },
        { ...PLAIN_MEMBER, user_id: OWNER_USER_ID },
      ],
    });
    render(<OrganizationSettingsPage />);

    const section = await aliasesSection();
    expect(within(section).getAllByTestId("org-alias-row")).toHaveLength(2);
    expect(
      within(section).queryByRole("button", { name: "Add alias" }),
    ).toBeNull();
    expect(within(section).queryByPlaceholderText("old-slug")).toBeNull();
  });
});
