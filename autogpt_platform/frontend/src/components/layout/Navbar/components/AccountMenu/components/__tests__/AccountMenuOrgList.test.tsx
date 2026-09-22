import { useOrgTeamStore } from "@/services/org-team/store";
import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { AccountMenuOrgList } from "../AccountMenuOrgList";

const PERSONAL_ORG = {
  id: "org-personal",
  name: "Jane's Org",
  slug: "jane",
  avatarUrl: null,
  isPersonal: true,
  memberCount: 1,
};

const COMPANY_ORG = {
  id: "org-company",
  name: "Acme Inc",
  slug: "acme",
  avatarUrl: null,
  isPersonal: false,
  memberCount: 12,
};

const DEFAULT_TEAM = {
  id: "team-default",
  name: "General",
  slug: "general",
  isDefault: true,
  joinPolicy: "OPEN",
  orgId: COMPANY_ORG.id,
};

const PRIVATE_TEAM = {
  id: "team-private",
  name: "Skunkworks",
  slug: "skunkworks",
  isDefault: false,
  joinPolicy: "PRIVATE",
  orgId: COMPANY_ORG.id,
};

function seedStore(overrides = {}) {
  useOrgTeamStore.setState({
    activeOrgID: COMPANY_ORG.id,
    activeTeamID: DEFAULT_TEAM.id,
    orgs: [PERSONAL_ORG, COMPANY_ORG],
    teams: [DEFAULT_TEAM, PRIVATE_TEAM],
    isLoaded: true,
    ...overrides,
  });
}

describe("AccountMenuOrgList", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("shows an empty state with a create button when there are no orgs", () => {
    seedStore({ orgs: [], teams: [] });

    render(<AccountMenuOrgList />);

    expect(screen.getByText("No organizations yet")).toBeDefined();
    expect(screen.getByTestId("create-organization-button")).toBeDefined();
  });

  it("lists every org with a Personal badge and marks the active org", () => {
    seedStore();

    render(<AccountMenuOrgList />);

    expect(screen.getByText(PERSONAL_ORG.name)).toBeDefined();
    expect(screen.getByText(COMPANY_ORG.name)).toBeDefined();
    expect(screen.getByText("Personal")).toBeDefined();
  });

  it("lists teams with a Private badge and a manage-teams link", () => {
    seedStore();

    render(<AccountMenuOrgList />);

    expect(screen.getByText(DEFAULT_TEAM.name)).toBeDefined();
    expect(screen.getByText(PRIVATE_TEAM.name)).toBeDefined();
    expect(screen.getByText("Private")).toBeDefined();
    expect(screen.getByText("Manage teams")).toBeDefined();
  });

  it("hides the team section when the org has no teams", () => {
    seedStore({ teams: [] });

    render(<AccountMenuOrgList />);

    expect(screen.queryByText("Teams")).toBeNull();
    expect(screen.queryByText("Manage teams")).toBeNull();
  });

  it("switching org updates the active org in the store", async () => {
    seedStore();
    render(<AccountMenuOrgList />);

    await userEvent.click(screen.getByText(PERSONAL_ORG.name));

    expect(useOrgTeamStore.getState().activeOrgID).toBe(PERSONAL_ORG.id);
  });

  it("switching team updates the active team in the store", async () => {
    seedStore();
    render(<AccountMenuOrgList />);

    await userEvent.click(screen.getByText(PRIVATE_TEAM.name));

    expect(useOrgTeamStore.getState().activeTeamID).toBe(PRIVATE_TEAM.id);
  });
});
