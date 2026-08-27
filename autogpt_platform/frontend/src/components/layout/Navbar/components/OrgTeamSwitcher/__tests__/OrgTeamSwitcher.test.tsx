import { useOrgTeamStore } from "@/services/org-team/store";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { OrgTeamSwitcher } from "../OrgTeamSwitcher";

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

function seedStore(overrides: Partial<ReturnType<typeof baseState>> = {}) {
  useOrgTeamStore.setState({ ...baseState(), ...overrides });
}

function baseState() {
  return {
    activeOrgID: COMPANY_ORG.id,
    activeTeamID: DEFAULT_TEAM.id,
    orgs: [PERSONAL_ORG, COMPANY_ORG],
    teams: [DEFAULT_TEAM, PRIVATE_TEAM],
    isLoaded: true,
  };
}

async function openSwitcher() {
  await userEvent.click(screen.getByTestId("org-switcher-trigger"));
  await waitFor(() => {
    expect(screen.getByTestId("org-switcher-popover")).toBeDefined();
  });
}

describe("OrgTeamSwitcher", () => {
  beforeEach(() => {
    process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS = "true";
    window.localStorage.clear();
  });

  afterEach(() => {
    delete process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS;
  });

  it("renders nothing before the org context has loaded", () => {
    seedStore({ isLoaded: false });

    const { container } = render(<OrgTeamSwitcher />);

    expect(container.innerHTML).toBe("");
  });

  it("renders nothing when the user belongs to no orgs", () => {
    seedStore({ orgs: [], teams: [] });

    const { container } = render(<OrgTeamSwitcher />);

    expect(container.innerHTML).toBe("");
  });

  it("shows the active org name on the trigger", () => {
    seedStore();

    render(<OrgTeamSwitcher />);

    expect(screen.getByTestId("org-switcher-trigger").textContent).toContain(
      COMPANY_ORG.name,
    );
  });

  it("lists every org with a Personal badge on the personal org", async () => {
    seedStore();
    render(<OrgTeamSwitcher />);

    await openSwitcher();

    expect(screen.getAllByText(COMPANY_ORG.name).length).toBeGreaterThan(0);
    expect(screen.getByText(PERSONAL_ORG.name)).toBeDefined();
    expect(screen.getByText("Personal")).toBeDefined();
    expect(screen.getByTestId("org-switcher-create")).toBeDefined();
    expect(screen.getByTestId("org-switcher-manage")).toBeDefined();
  });

  it("hides organization management when the feature flag is off", async () => {
    process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS = "false";
    seedStore();
    render(<OrgTeamSwitcher />);

    await openSwitcher();

    expect(screen.queryByTestId("org-switcher-create")).toBeNull();
    expect(screen.queryByTestId("org-switcher-manage")).toBeNull();
  });

  it("does not render a team-switching section (teams are badges, not context)", async () => {
    seedStore();
    render(<OrgTeamSwitcher />);

    await openSwitcher();

    expect(screen.queryByText("Teams")).toBeNull();
    expect(screen.queryByText(DEFAULT_TEAM.name)).toBeNull();
    expect(screen.queryByText(PRIVATE_TEAM.name)).toBeNull();
    expect(screen.getByText("Manage organization")).toBeDefined();
  });

  it("switching org updates the store and resets the active team", async () => {
    seedStore();
    render(<OrgTeamSwitcher />);

    await openSwitcher();
    await userEvent.click(screen.getByText(PERSONAL_ORG.name));

    expect(useOrgTeamStore.getState().activeOrgID).toBe(PERSONAL_ORG.id);
    expect(useOrgTeamStore.getState().activeTeamID).toBeNull();
    await waitFor(() =>
      expect(screen.queryByTestId("org-switcher-popover")).toBeNull(),
    );
  });

  it("re-selecting the already-active org leaves state untouched", async () => {
    seedStore();
    render(<OrgTeamSwitcher />);

    await openSwitcher();
    const popover = screen.getByTestId("org-switcher-popover");
    const activeOrgButton = Array.from(popover.querySelectorAll("button")).find(
      (b) => b.textContent?.includes(COMPANY_ORG.name),
    );
    await userEvent.click(activeOrgButton!);

    expect(useOrgTeamStore.getState().activeOrgID).toBe(COMPANY_ORG.id);
    expect(useOrgTeamStore.getState().activeTeamID).toBe(DEFAULT_TEAM.id);
  });
});
