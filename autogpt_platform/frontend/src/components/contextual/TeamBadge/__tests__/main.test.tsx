import { useOrgTeamStore } from "@/services/org-team/store";
import { render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

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
  process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS = "true";
  useOrgTeamStore.setState({
    activeOrgID: "org-1",
    activeTeamID: null,
    orgs: [],
    teams: [TEAM_A],
    isLoaded: true,
  });
});

afterEach(() => {
  delete process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS;
});

describe("TeamBadge", () => {
  it.each(["false", undefined])(
    "hides a known team when the flag is %s",
    (value) => {
      if (value === undefined) {
        delete process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS;
      } else {
        process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS = value;
      }
      const { container } = render(<TeamBadge teamId="team-a" />);
      expect(container.textContent).toBe("");
    },
  );
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
