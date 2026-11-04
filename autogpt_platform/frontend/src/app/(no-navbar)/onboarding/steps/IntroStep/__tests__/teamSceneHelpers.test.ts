import { describe, expect, it } from "vitest";
import {
  TEAM,
  TEAM_PARENTS,
  revealDelay,
  teamBranch,
} from "../components/teamSceneHelpers";

const LABEL_BOTTOM = 6 + 40;

describe("teamBranch", () => {
  it.each(TEAM_PARENTS)("joins $name to each of their team", (parent) => {
    const branch = teamBranch(parent);
    const children = TEAM.filter((member) => member.parent === parent.asset);
    const parentBottom =
      parent.top + parent.size + (parent.parent ? LABEL_BOTTOM : 0);

    expect(branch.start).toBeGreaterThan(parentBottom);
    expect(branch.middle).toBeGreaterThan(branch.start);
    expect(branch.end).toBe(Math.min(...children.map((child) => child.top)));
    expect(branch.middle).toBeLessThan(branch.end);
    expect(branch.left).toBe(children[0].x);
    expect(branch.right).toBe(children[children.length - 1].x);
    expect(branch.left).toBeLessThanOrEqual(parent.x);
    expect(branch.right).toBeGreaterThanOrEqual(parent.x);
  });
});

describe("revealDelay", () => {
  it("brings in Otto first, then each row from left to right", () => {
    const delays = TEAM.map(revealDelay);
    expect(delays).toEqual([...delays].sort((a, b) => a - b));
  });
});
