import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { AgentActionsDropdown } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/selected-views/AgentActionsDropdown";
import { useOrgTeamStore } from "@/services/org-team/store";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AgentCardMenu } from "./AgentCardMenu";

vi.mock("../../MoveToFolderDialog/MoveToFolderDialog", () => ({
  MoveToFolderDialog: () => null,
}));

const agent = {
  id: "library-agent-1",
  graph_id: "graph-1",
  graph_version: 1,
  name: "Personal agent",
  can_access_graph: true,
  organization_id: "org-1",
  team_id: "default-team",
} as LibraryAgent;

beforeEach(() => {
  useOrgTeamStore.setState({
    activeOrgID: "org-1",
    activeTeamID: "default-team",
    isLoaded: true,
    teams: [
      {
        id: "team-a",
        orgId: "org-1",
        name: "Growth",
        slug: "growth",
        isDefault: false,
        joinPolicy: "closed",
      },
    ],
  });
});

afterEach(() => vi.unstubAllEnvs());

describe.each([
  ["library card", AgentCardMenu],
  ["agent details", AgentActionsDropdown],
] as const)("%s sharing entry", (_name, Menu) => {
  it("keeps personal actions and hides sharing for a member with teams when disabled", async () => {
    vi.stubEnv("NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS", "false");
    render(<Menu agent={agent} />);
    fireEvent.pointerDown(
      screen.getByRole("button", { name: "More actions" }),
      {
        button: 0,
      },
    );

    expect(
      await screen.findByRole("menuitem", { name: "Edit agent" }),
    ).toBeDefined();
    expect(
      screen.getByRole("menuitem", { name: "Delete agent" }),
    ).toBeDefined();
    expect(
      screen.queryByRole("menuitem", { name: "Share with a team" }),
    ).toBeNull();
    expect(screen.queryByTestId("share-agent-dialog")).toBeNull();
  });

  it("retains the team sharing entry when enabled", async () => {
    vi.stubEnv("NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS", "true");
    render(<Menu agent={agent} />);
    fireEvent.pointerDown(
      screen.getByRole("button", { name: "More actions" }),
      {
        button: 0,
      },
    );

    expect(
      await screen.findByRole("menuitem", { name: "Share with a team" }),
    ).toBeDefined();
  });
});
