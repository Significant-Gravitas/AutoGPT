import { describe, expect, it } from "vitest";
import {
  matchesTeamFilter,
  TEAM_FILTER_ALL,
  TEAM_FILTER_ORG_HOME,
} from "./helpers";

describe("matchesTeamFilter", () => {
  it("passes every row when filtering by All", () => {
    expect(matchesTeamFilter("team-a", TEAM_FILTER_ALL)).toBe(true);
    expect(matchesTeamFilter(null, TEAM_FILTER_ALL)).toBe(true);
    expect(matchesTeamFilter(undefined, TEAM_FILTER_ALL)).toBe(true);
  });

  it("passes only org-home rows when filtering by Organization", () => {
    expect(matchesTeamFilter(null, TEAM_FILTER_ORG_HOME)).toBe(true);
    expect(matchesTeamFilter(undefined, TEAM_FILTER_ORG_HOME)).toBe(true);
    expect(matchesTeamFilter("team-a", TEAM_FILTER_ORG_HOME)).toBe(false);
  });

  it("passes only the selected team's rows", () => {
    expect(matchesTeamFilter("team-a", "team-a")).toBe(true);
    expect(matchesTeamFilter("team-b", "team-a")).toBe(false);
    expect(matchesTeamFilter(null, "team-a")).toBe(false);
  });
});
