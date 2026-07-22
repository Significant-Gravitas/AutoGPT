import { useOrgTeamStore } from "@/services/org-team/store";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, it } from "vitest";

import { TeamFilter } from "../TeamFilter";
import { useTeamFilter } from "../useTeamFilter";

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

const ROWS = [
  { id: "row-growth", teamId: "team-a" },
  { id: "row-orghome", teamId: null as string | null },
  { id: "row-platform", teamId: "team-b" },
];

function Harness() {
  const { value, setValue, matches } = useTeamFilter();
  const filtered = ROWS.filter((r) => matches(r.teamId));
  return (
    <>
      <TeamFilter value={value} onChange={setValue} />
      <ul>
        {filtered.map((r) => (
          <li key={r.id} data-testid="row">
            {r.id}
          </li>
        ))}
      </ul>
    </>
  );
}

function visibleRowIds() {
  return screen.getAllByTestId("row").map((el) => el.textContent);
}

function seedTeams(teams: (typeof TEAM_A)[]) {
  useOrgTeamStore.setState({
    activeOrgID: "org-1",
    activeTeamID: null,
    orgs: [],
    teams,
    isLoaded: true,
  });
}

beforeEach(() => {
  useOrgTeamStore.setState({
    activeOrgID: null,
    activeTeamID: null,
    orgs: [],
    teams: [],
    isLoaded: false,
  });
});

describe("TeamFilter", () => {
  it("renders nothing and shows all rows for solo users (no teams)", () => {
    seedTeams([]);
    render(<Harness />);

    expect(screen.queryByRole("combobox")).toBeNull();
    expect(visibleRowIds()).toHaveLength(3);
  });

  it("defaults to All and shows every row", () => {
    seedTeams([TEAM_A, TEAM_B]);
    render(<Harness />);

    expect(visibleRowIds()).toHaveLength(3);
  });

  it("narrows to a single team's rows", async () => {
    seedTeams([TEAM_A, TEAM_B]);
    render(<Harness />);

    fireEvent.click(screen.getByRole("combobox", { name: "Team" }));
    fireEvent.click(await screen.findByRole("option", { name: "Growth" }));

    await waitFor(() => {
      expect(visibleRowIds()).toEqual(["row-growth"]);
    });
  });

  it("narrows to org-home rows when filtering by Organization", async () => {
    seedTeams([TEAM_A, TEAM_B]);
    render(<Harness />);

    fireEvent.click(screen.getByRole("combobox", { name: "Team" }));
    fireEvent.click(
      await screen.findByRole("option", { name: "Organization" }),
    );

    await waitFor(() => {
      expect(visibleRowIds()).toEqual(["row-orghome"]);
    });
  });

  it("resets a team filter to All when the selected team disappears", async () => {
    seedTeams([TEAM_A, TEAM_B]);
    render(<Harness />);

    fireEvent.click(screen.getByRole("combobox", { name: "Team" }));
    fireEvent.click(await screen.findByRole("option", { name: "Growth" }));

    await waitFor(() => {
      expect(visibleRowIds()).toEqual(["row-growth"]);
    });

    // Team removed (deleted, or switched to an org without it).
    act(() => seedTeams([TEAM_B]));

    await waitFor(() => {
      // Filter fell back to All, so every row is visible again.
      expect(visibleRowIds()).toHaveLength(3);
    });
  });

  it("resets to All (and hides the control) when the team list empties", async () => {
    seedTeams([TEAM_A, TEAM_B]);
    render(<Harness />);

    fireEvent.click(screen.getByRole("combobox", { name: "Team" }));
    fireEvent.click(await screen.findByRole("option", { name: "Growth" }));

    await waitFor(() => {
      expect(visibleRowIds()).toEqual(["row-growth"]);
    });

    // Solo org / left the org: team list empties, control hides.
    act(() => seedTeams([]));

    await waitFor(() => {
      expect(screen.queryByRole("combobox")).toBeNull();
      expect(visibleRowIds()).toHaveLength(3);
    });
  });
});
