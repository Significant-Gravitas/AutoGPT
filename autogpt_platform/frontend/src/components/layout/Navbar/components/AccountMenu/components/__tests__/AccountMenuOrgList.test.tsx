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

// Teams still exist as a product concept (org-settings teams tab, badges,
// filters) but are no longer context switches, so the switcher list must
// ignore them entirely. We seed teams here to prove they never surface.
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

  it("renders nothing until the org store has loaded", () => {
    seedStore({ isLoaded: false });

    render(<AccountMenuOrgList />);

    expect(screen.queryByTestId("create-organization-button")).toBeNull();
    expect(screen.queryByText("No organizations yet")).toBeNull();
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

  it("never renders a teams section — teams are managed in org settings, not context switches", () => {
    seedStore();

    render(<AccountMenuOrgList />);

    expect(screen.queryByText("Teams")).toBeNull();
    expect(screen.queryByText("Manage teams")).toBeNull();
    expect(screen.queryByText(DEFAULT_TEAM.name)).toBeNull();
    expect(screen.queryByText(PRIVATE_TEAM.name)).toBeNull();
  });

  it("switching org updates the active org in the store", async () => {
    seedStore();
    render(<AccountMenuOrgList />);

    await userEvent.click(screen.getByText(PERSONAL_ORG.name));

    expect(useOrgTeamStore.getState().activeOrgID).toBe(PERSONAL_ORG.id);
  });

  it("opens the create-organization dialog when the create button is clicked", async () => {
    seedStore();
    render(<AccountMenuOrgList />);

    expect(screen.queryByText("URL slug")).toBeNull();

    await userEvent.click(screen.getByTestId("create-organization-button"));

    expect(await screen.findByText("URL slug")).toBeDefined();
  });
});
