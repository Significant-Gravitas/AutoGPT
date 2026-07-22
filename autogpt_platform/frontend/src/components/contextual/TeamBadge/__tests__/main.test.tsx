import { useOrgTeamStore } from "@/services/org-team/store";
import { render, screen } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, it } from "vitest";

import { TeamBadge } from "../TeamBadge";

const TEAM_A = {
  id: "team-a",
  name: "Growth",
  slug: "growth",
  isDefault: false,
  joinPolicy: "closed",
  orgId: "org-1",
};

beforeEach(() => {
  useOrgTeamStore.setState({
    activeOrgID: "org-1",
    activeTeamID: null,
    orgs: [],
    teams: [TEAM_A],
    isLoaded: true,
  });
});

describe("TeamBadge", () => {
  it("renders the team name resolved from the store", () => {
    render(<TeamBadge teamId="team-a" />);
    expect(screen.getByText("Growth")).toBeTruthy();
  });

  it("renders nothing for an org-home row (null teamId)", () => {
    const { container } = render(<TeamBadge teamId={null} />);
    expect(container.textContent).toBe("");
  });

  it("renders nothing for a team the store does not know", () => {
    const { container } = render(<TeamBadge teamId="ghost-team" />);
    expect(container.textContent).toBe("");
  });
});
