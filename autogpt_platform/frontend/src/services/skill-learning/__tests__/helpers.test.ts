import { describe, expect, it } from "vitest";
import { lineDiff, skillDetailHref, stepChanges } from "../helpers";

describe("skill-learning helpers", () => {
  it("groups before/after by procedure step", () => {
    const before =
      "---\nname: x\n---\n\n## Steps\n1. Open the file\n2. Validate rows\n";
    const after =
      "## Steps\n1. Open the file as utf-8\n2. Validate rows\n3. Report the count\n";
    expect(stepChanges(before, after)).toEqual([
      {
        step: "Step 1",
        before: "Open the file",
        after: "Open the file as utf-8",
      },
      { step: "Step 3", before: null, after: "Report the count" },
    ]);
  });

  it("produces a raw line diff without frontmatter", () => {
    const diff = lineDiff("---\nname: x\n---\nline a\n", "line b\n");
    expect(diff).toBe("- line a\n+ line b");
  });

  it("builds detail links per scope", () => {
    expect(
      skillDetailHref({ expertId: "e1", skillName: "csv", versionId: "v1" }),
    ).toBe("/team/e1?tab=skills&skill=csv&version=v1");
    expect(
      skillDetailHref({ expertId: null, skillName: "csv", versionId: "v2" }),
    ).toBe("/settings/memory?skill=csv&version=v2");
  });
});
