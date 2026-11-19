import { describe, expect, it } from "vitest";
import {
  LABEL_GAP,
  LABEL_HEIGHT,
  TEAM,
  TEAM_PARENTS,
  revealDelay,
  teamBranch,
} from "../components/teamSceneHelpers";

const LABEL_BOTTOM = LABEL_GAP + LABEL_HEIGHT;
// The bar's rounded corners (rounded-t-xl) need this much height.
const CORNER = 12;

function member(asset: string) {
  const found = TEAM.find((m) => m.asset === asset);
  if (!found) throw new Error(`No ${asset} in TEAM`);
  return found;
}

describe("teamBranch", () => {
  it.each(TEAM_PARENTS)("joins $name to each of their team", (parent) => {
    const branch = teamBranch(parent);
    const children = TEAM.filter((member) => member.parent === parent.asset);
    const parentBottom =
      parent.top + parent.size + (parent.parent ? LABEL_BOTTOM : 0);

    expect(branch.start).toBeGreaterThan(parentBottom);
    expect(branch.end).toBe(Math.min(...children.map((child) => child.top)));
    expect((branch.end - branch.start) / 2).toBeGreaterThanOrEqual(CORNER);
    expect(branch.left).toBe(children[0].x);
    expect(branch.right).toBe(children[children.length - 1].x);
    expect(branch.innerDrops).toEqual(
      children.slice(1, -1).map((child) => child.x),
    );
    expect(branch.left).toBeLessThanOrEqual(parent.x);
    expect(branch.right).toBeGreaterThanOrEqual(parent.x);
  });

  it("groups each department under its head", () => {
    expect(TEAM.filter((m) => m.parent === "otto").map((m) => m.name)).toEqual([
      "Maria",
      "Max",
      "Mina",
    ]);
    expect(TEAM.filter((m) => m.parent === "maria").map((m) => m.name)).toEqual(
      ["Nadia", "Riley"],
    );
    expect(TEAM.filter((m) => m.parent === "mina").map((m) => m.name)).toEqual([
      "Frankie",
      "Devon",
    ]);
    expect(teamBranch(member("otto")).innerDrops).toEqual([300]);
    expect(teamBranch(member("maria"))).toMatchObject({ left: 75, right: 225 });
  });
});

describe("revealDelay", () => {
  it("brings in Otto first, then each row from left to right", () => {
    const delays = TEAM.map(revealDelay);
    expect(new Set(delays).size).toBe(TEAM.length);

    const rows = [...new Set(TEAM.map((m) => m.top))].sort((a, b) => a - b);
    let previous = -Infinity;
    for (const top of rows) {
      const row = TEAM.filter((m) => m.top === top).sort((a, b) => a.x - b.x);
      for (const m of row) {
        expect(revealDelay(m)).toBeGreaterThan(previous);
        previous = revealDelay(m);
      }
    }
  });
});
