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

describe("review accuracy", () => {
  it("retains the order and repeated lines in a raw diff", () => {
    expect(lineDiff("Validate\nImport", "Import\nValidate")).toBe(
      "- Validate\n- Import\n+ Import\n+ Validate",
    );
    expect(lineDiff("Check\nCheck\nImport", "Check\nImport")).toBe("- Check");
    expect(lineDiff("Check\nImport", "Check\n\nImport")).toBe("+ ");
  });

  it("does not let a later numbered section hide a procedure correction", () => {
    const before =
      "## Steps\n1. Import rows\n\n## Verification\n1. Check counts";
    const after =
      "## Steps\n1. Validate then import rows\n\n## Verification\n1. Check counts";
    expect(stepChanges(before, after)).toEqual([
      {
        step: "Step 1",
        before: "Import rows",
        after: "Validate then import rows",
      },
    ]);
  });

  it("keeps separate bullets and verification changes visible", () => {
    const before =
      "## Steps\n- Open\n- Import\n## Verification\n1. Check counts";
    const after =
      "## Steps\n- Open\n- Validate\n- Import\n## Verification\n1. Check types";
    expect(stepChanges(before, after)).toEqual([
      { step: "Step 2", before: "Import", after: "Validate" },
      {
        step: "Verification · Step 1",
        before: "Check counts",
        after: "Check types",
      },
      { step: "Step 3", before: null, after: "Import" },
    ]);
  });
});
