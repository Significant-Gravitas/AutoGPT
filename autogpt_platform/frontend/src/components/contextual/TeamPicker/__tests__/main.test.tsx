import { useOrgTeamStore } from "@/services/org-team/store";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, it } from "vitest";

import { CreateSurface } from "../helpers";
import { TeamPicker } from "../TeamPicker";
import { useCreateTeamSelection } from "../useCreateTeamSelection";

const TEAM_A = {
  id: "team-a",
  name: "Growth",
  slug: "growth",
  isDefault: false,
  joinPolicy: "closed",
  orgId: "org-1",
};
const TEAM_B = {
  id: "team-b",
  name: "Platform",
  slug: "platform",
  isDefault: false,
  joinPolicy: "closed",
  orgId: "org-1",
};

function seedTeams(teams: (typeof TEAM_A)[]) {
  useOrgTeamStore.setState({
    activeOrgID: "org-1",
    activeTeamID: null,
    orgs: [],
    teams,
    isLoaded: true,
  });
}

// Harness wiring the picker to the real state hook, mirroring how create
// surfaces consume it.
function Harness({ surfaceKey }: { surfaceKey: string }) {
  const { teamId, setTeamId } = useCreateTeamSelection(surfaceKey);
  return (
    <>
      <TeamPicker surfaceKey={surfaceKey} value={teamId} onChange={setTeamId} />
      <div data-testid="current">{teamId ?? "org-home"}</div>
    </>
  );
}

function currentValue() {
  return screen.getByTestId("current").textContent;
}

beforeEach(() => {
  window.localStorage.clear();
  useOrgTeamStore.setState({
    activeOrgID: null,
    activeTeamID: null,
    orgs: [],
    teams: [],
    isLoaded: false,
  });
});

describe("TeamPicker", () => {
  it("renders nothing when the user has no teams", () => {
    seedTeams([]);
    render(<Harness surfaceKey={CreateSurface.BuilderSave} />);

    expect(screen.queryByRole("combobox")).toBeNull();
    expect(currentValue()).toBe("org-home");
  });

  it("defaults to Organization (org-home) when the surface was never used", () => {
    seedTeams([TEAM_A, TEAM_B]);
    render(<Harness surfaceKey={CreateSurface.BuilderSave} />);

    expect(
      screen.getByRole("combobox", { name: "Team" }).textContent,
    ).toContain("Organization");
    expect(currentValue()).toBe("org-home");
  });

  it("lists Organization plus every team as options", async () => {
    seedTeams([TEAM_A, TEAM_B]);
    render(<Harness surfaceKey={CreateSurface.BuilderSave} />);

    fireEvent.click(screen.getByRole("combobox", { name: "Team" }));

    expect(
      await screen.findByRole("option", { name: "Organization" }),
    ).toBeTruthy();
    expect(screen.getByRole("option", { name: "Growth" })).toBeTruthy();
    expect(screen.getByRole("option", { name: "Platform" })).toBeTruthy();
  });

  it("selecting a team updates the value and remembers it for the surface", async () => {
    seedTeams([TEAM_A, TEAM_B]);
    render(<Harness surfaceKey={CreateSurface.BuilderSave} />);

    fireEvent.click(screen.getByRole("combobox", { name: "Team" }));
    fireEvent.click(await screen.findByRole("option", { name: "Platform" }));

    await waitFor(() => {
      expect(currentValue()).toBe("team-b");
    });
    expect(
      JSON.parse(window.localStorage.getItem("create-surface-teams") ?? "{}"),
    ).toMatchObject({ [CreateSurface.BuilderSave]: "team-b" });
  });

  it("seeds the last-used team for that surface on mount", () => {
    window.localStorage.setItem(
      "create-surface-teams",
      JSON.stringify({ [CreateSurface.LibraryFolder]: "team-a" }),
    );
    seedTeams([TEAM_A, TEAM_B]);
    render(<Harness surfaceKey={CreateSurface.LibraryFolder} />);

    expect(currentValue()).toBe("team-a");
    expect(
      screen.getByRole("combobox", { name: "Team" }).textContent,
    ).toContain("Growth");
  });

  it("does not leak one surface's last-used team into another", () => {
    window.localStorage.setItem(
      "create-surface-teams",
      JSON.stringify({ [CreateSurface.LibraryFolder]: "team-a" }),
    );
    seedTeams([TEAM_A, TEAM_B]);
    render(<Harness surfaceKey={CreateSurface.BuilderSave} />);

    expect(currentValue()).toBe("org-home");
  });

  it("falls back to org-home when the last-used team no longer exists", async () => {
    window.localStorage.setItem(
      "create-surface-teams",
      JSON.stringify({ [CreateSurface.BuilderSave]: "deleted-team" }),
    );
    seedTeams([TEAM_A]);
    render(<Harness surfaceKey={CreateSurface.BuilderSave} />);

    await waitFor(() => {
      expect(currentValue()).toBe("org-home");
    });
  });
});
