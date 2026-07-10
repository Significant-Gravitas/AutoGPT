import { SidebarProvider } from "@/components/ui/sidebar";
import { useOrgTeamStore } from "@/services/org-team/store";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { SidebarOrgSwitcher } from "../SidebarOrgSwitcher";

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

function baseState() {
  return {
    activeOrgID: COMPANY_ORG.id,
    activeTeamID: DEFAULT_TEAM.id,
    orgs: [PERSONAL_ORG, COMPANY_ORG],
    teams: [DEFAULT_TEAM, PRIVATE_TEAM],
    isLoaded: true,
  };
}

function seedStore(overrides: Partial<ReturnType<typeof baseState>> = {}) {
  useOrgTeamStore.setState({ ...baseState(), ...overrides });
}

function renderSwitcher() {
  return render(
    <SidebarProvider>
      <SidebarOrgSwitcher />
    </SidebarProvider>,
  );
}

async function openSwitcher() {
  await userEvent.click(screen.getByTestId("sidebar-org-switcher-trigger"));
  await waitFor(() => {
    expect(screen.getByTestId("sidebar-org-switcher-content")).toBeDefined();
  });
}

describe("SidebarOrgSwitcher", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("renders nothing before the org context has loaded", () => {
    seedStore({ isLoaded: false });
    renderSwitcher();
    expect(screen.queryByTestId("sidebar-org-switcher-trigger")).toBeNull();
  });

  it("renders nothing when the user belongs to no orgs", () => {
    seedStore({ orgs: [], teams: [] });
    renderSwitcher();
    expect(screen.queryByTestId("sidebar-org-switcher-trigger")).toBeNull();
  });

  it("shows the active org and team on the trigger", () => {
    seedStore();
    renderSwitcher();
    const trigger = screen.getByTestId("sidebar-org-switcher-trigger");
    expect(trigger.textContent).toContain(COMPANY_ORG.name);
    expect(trigger.textContent).toContain(DEFAULT_TEAM.name);
  });

  it("lists every org and team with badges when opened", async () => {
    seedStore();
    renderSwitcher();
    await openSwitcher();

    expect(screen.getByText(PERSONAL_ORG.name)).toBeDefined();
    expect(screen.getAllByText(COMPANY_ORG.name).length).toBeGreaterThan(0);
    expect(screen.getByText("Personal")).toBeDefined();
    expect(screen.getByText(PRIVATE_TEAM.name)).toBeDefined();
    expect(screen.getByText("Private")).toBeDefined();
  });

  it("does not surface Create organization (hidden until org mgmt ships)", async () => {
    seedStore();
    renderSwitcher();
    await openSwitcher();
    expect(screen.queryByText("Create organization")).toBeNull();
  });

  it("switches the active org when another org is picked", async () => {
    seedStore();
    renderSwitcher();
    await openSwitcher();

    await userEvent.click(screen.getByText(PERSONAL_ORG.name));

    await waitFor(() => {
      expect(useOrgTeamStore.getState().activeOrgID).toBe(PERSONAL_ORG.id);
    });
  });
});
